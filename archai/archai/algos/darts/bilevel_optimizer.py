# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterator, Mapping, Optional, Union
import copy

import torch
from torch import Tensor, nn, autograd
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from archai.common.config import Config
from archai.common import utils, ml_utils
from archai.nas.model import Model
from archai.common.common import logger
from archai.common.utils import zip_eq

def _get_loss(model:Model, lossfn, x, y):
    logits, *_ = model(x) # might also return aux tower logits
    return lossfn(logits, y)

def _get_alphas(model:Model)->Iterator[nn.Parameter]:
    return model.all_owned().param_by_kind('alphas')

class BilevelOptimizer:
    def __init__(self, conf_alpha_optim:Config, w_momentum: float, w_decay: float,
                 model: Model, lossfn: _Loss, device, batch_chunks:int) -> None:
        self._w_momentum = w_momentum  # momentum for w
        self._w_weight_decay = w_decay  # weight decay for w
        self._lossfn = lossfn
        self._model = model  # main model with respect to w and alpha
        self.batch_chunks = batch_chunks
        self.device = device

        # create a copy of model which we will use
        # to compute grads for alphas without disturbing
        # original weights
        self._vmodel = copy.deepcopy(model).to(device)

        self._alphas = list(_get_alphas(self._model))
        self._valphas = list(_get_alphas(self._vmodel))

        # this is the optimizer to optimize alphas parameter
        self._alpha_optim = ml_utils.create_optimizer(conf_alpha_optim, self._alphas)

    def state_dict(self)->dict:
        return {
            'alpha_optim': self._alpha_optim.state_dict(),
            'vmodel': self._vmodel.state_dict()
        }

    def load_state_dict(self, state_dict)->None:
        self._vmodel.load_state_dict(state_dict['vmodel'])
        self._alpha_optim.load_state_dict(state_dict['alpha_optim'])

    # NOTE: Original dart paper uses all paramaeters which includes ops weights
    # as well as stems and alphas however in theory it should only be using
    # ops weights. Below you can conduct experiment by replacing parameters()
    # with weights() but that tanks accuracy below 97.0 for cifar10
    def _model_params(self):
        return self._model.parameters()
        #return self._model.nonarch_params(recurse=True)
    def _vmodel_params(self):
        return self._vmodel.parameters()
        #return self._vmodel.nonarch_params(recurse=True)

    def _update_vmodel(self, x, y, lr: float, w_optim: Optimizer) -> None:
        """ Update vmodel with w' (main model has w) """

        # TODO: should this loss be stored for later use?
        loss = _get_loss(self._model, self._lossfn, x, y)
        gradients = autograd.grad(loss, self._model_params())

        """update weights in vmodel so we leave main model undisturbed
        The main technical difficulty computing w' without affecting alphas is
        that you can't simply do backward() and step() on loss because loss
        tracks alphas as well as w. So, we compute gradients using autograd and
        do manual sgd update."""
        # TODO: other alternative may be to (1) copy model
        #   (2) set require_grads = False on alphas
        #   (3) loss and step on vmodel (4) set back require_grads = True
        with torch.no_grad():  # no need to track gradient for these operations
            for w, vw, g in zip(
                    self._model_params(), self._vmodel_params(), gradients):
                # simulate momentum update on model but put this update in vmodel
                m = w_optim.state[w].get(
                    'momentum_buffer', 0.)*self._w_momentum
                vw.copy_(w - lr * (m + g + self._w_weight_decay*w))

            # synchronize alphas
            for a, va in zip_eq(self._alphas, self._valphas):
                va.copy_(a)

    def step(self, x_train: Tensor, y_train: Tensor, x_valid: Tensor, y_valid: Tensor,
             w_optim: Optimizer) -> None:
        # TODO: unlike darts paper, we get lr from optimizer insead of scheduler
        lr = w_optim.param_groups[0]['lr']
        self._alpha_optim.zero_grad()

        # divide batch in to chunks if needed so it fits in GPU RAM
        if self.batch_chunks > 1:
            xt_chunks, yt_chunks = torch.chunk(x_train, self.batch_chunks), torch.chunk(y_train, self.batch_chunks)
            xv_chunks, yv_chuncks = torch.chunk(x_valid, self.batch_chunks), torch.chunk(y_valid, self.batch_chunks)
        else:
            xt_chunks, yt_chunks = (x_train,), (y_train,)
            xv_chunks, yv_chuncks = (x_valid,), (y_valid,)

        for xtc, ytc, xvc, yvc in zip(xt_chunks, yt_chunks, xv_chunks, yv_chuncks):
            xtc, ytc = xtc.to(self.device), ytc.to(self.device, non_blocking=True)
            xvc, yvc = xvc.to(self.device), yvc.to(self.device, non_blocking=True)

            # compute the gradient and write it into tensor.grad
            # instead of generated by loss.backward()
            self._backward_bilevel(xtc, ytc, xvc, yvc,lr, w_optim)

        # at this point we should have model with updated gradients for w and alpha
        self._alpha_optim.step()

    def _backward_bilevel(self, x_train, y_train, x_valid, y_valid, lr, w_optim):
        """ Compute unrolled loss and backward its gradients """

        # update vmodel with w', but leave alphas as-is
        # w' = w - lr * grad
        self._update_vmodel(x_train, y_train, lr, w_optim)

        # compute loss on validation set for model with w'
        # wrt alphas. The autograd.grad is used instead of backward()
        # to avoid having to loop through params
        vloss = _get_loss(self._vmodel, self._lossfn, x_valid, y_valid)

        v_alphas = tuple(self._valphas)
        v_weights = tuple(self._vmodel_params())
        # TODO: if v_weights = all params then below does double counting of alpahs
        v_grads = autograd.grad(vloss, v_alphas + v_weights)

        # grad(L(w', a), a), part of Eq. 6
        dalpha = v_grads[:len(v_alphas)]
        # get grades for w' params which we will use it to compute w+ and w-
        dw = v_grads[len(v_alphas):]

        hessian = self._hessian_vector_product(dw, x_train, y_train)

        # dalpha we have is from the unrolled model so we need to
        # transfer those grades back to our main model
        # update final gradient = dalpha - xi*hessian
        # TODO: currently alphas lr is same as w lr
        with torch.no_grad():
            for alpha, da, h in zip(self._alphas, dalpha, hessian):
                alpha.grad = da - lr*h
        # now that model has both w and alpha grads,
        # we can run w_optim.step() to update the param values

    def _hessian_vector_product(self, dw, x, y, epsilon_unit=1e-2):
        """
        Implements equation 8

        dw = dw` {L_val(w`, alpha)}
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha {L_trn(w+, alpha)} -dalpha {L_trn(w-, alpha)})/(2*eps)
        eps = 0.01 / ||dw||
        """

        """scale epsilon with grad magnitude. The dw
        is a multiplier on RHS of eq 8. So this scalling is essential
        in making sure that finite differences approximation is not way off
        Below, we flatten each w, concate all and then take norm"""
        # TODO: is cat along dim 0 correct?
        dw_norm = torch.cat([w.view(-1) for w in dw]).norm()
        epsilon = epsilon_unit / dw_norm

        # w+ = w + epsilon * grad(w')
        with torch.no_grad():
            for p, v in zip(self._model_params(), dw):
                p += epsilon * v

        # Now that we have model with w+, we need to compute grads wrt alphas
        # This loss needs to be on train set, not validation set
        loss = _get_loss(self._model, self._lossfn, x, y)
        dalpha_plus = autograd.grad(
            loss, self._alphas)  # dalpha{L_trn(w+)}

        # get model with w- and then compute grads wrt alphas
        # w- = w - eps*dw`
        with torch.no_grad():
            for p, v in zip(self._model_params(), dw):
                # we had already added dw above so sutracting twice gives w-
                p -= 2. * epsilon * v

        # similarly get dalpha_minus
        loss = _get_loss(self._model, self._lossfn, x, y)
        dalpha_minus = autograd.grad(loss, self._alphas)

        # reset back params to original values by adding dw
        with torch.no_grad():
            for p, v in zip(self._model_params(), dw):
                p += epsilon * v

        # apply eq 8, final difference to compute hessian
        h = [(p - m) / (2. * epsilon)
             for p, m in zip(dalpha_plus, dalpha_minus)]
        return h
