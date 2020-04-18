from typing import Callable, Tuple, Optional

from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from overrides import EnforceOverrides

from .metrics import Metrics
from .tester import Tester
from .config import Config
from . import utils, ml_utils
from ..common.common import logger
from ..common.checkpoint import CheckPoint
from .apex_utils import Amp


class Trainer(EnforceOverrides):
    def __init__(self, conf_train:Config, model:nn.Module, device,
                 checkpoint:Optional[CheckPoint])->None:
        # region config vars
        conf_lossfn = conf_train['lossfn']
        self._apex = conf_train['apex']
        self._aux_weight = conf_train['aux_weight']
        self._grad_clip = conf_train['grad_clip']
        self._drop_path_prob = conf_train['drop_path_prob']
        self._logger_freq = conf_train['logger_freq']
        self._title = conf_train['title']
        self._epochs = conf_train['epochs']
        self._conf_optim = conf_train['optimizer']
        self._conf_sched = conf_train['lr_schedule']
        conf_validation = conf_train['validation']
        self._validation_freq = 0 if conf_validation is None else conf_validation['freq']
        # endregion

        self._checkpoint = checkpoint
        self.model = model
        self.device = device
        self._lossfn = ml_utils.get_lossfn(conf_lossfn).to(device)
        self._tester = Tester(conf_validation, model, device) \
                        if conf_validation else None
        self._metrics:Optional[Metrics] = None
        self._amp = Amp(self._apex)
        self._start_epoch = -1 # nothing is started yet

    def fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->Metrics:
        logger.pushd(self._title)

        self._metrics = Metrics(self._title, logger_freq=self._logger_freq)

        # optimizers, schedulers needs to be recreated for each fit call
        # as they have state specific to each run
        optim = self.create_optimizer()
        # create scheduler for optim before applying amp
        self._sched, self._sched_on_epoch = self._create_scheduler(optim, len(train_dl))
        # before checkpoint restore, convert to amp
        # TODO: see if original model gets lost after to_amp?
        self.model, self._optim = self._amp.to_amp(self.model, optim)

        self.pre_fit(train_dl, val_dl)

        # we need to restore checkpoint after all objects are created because
        # restoring checkpoint requires load_state_dict calls on these objects
        self._start_epoch = 0
        checkpoint_avail = self._checkpoint is not None
        checkpoint_val = checkpoint_avail and 'trainer' in self._checkpoint
        resumed = False
        if checkpoint_val:
            resumed = True
            self.restore_checkpoint()
        elif checkpoint_avail: # TODO: bad checkpoint?
            self._checkpoint.clear()
        logger.warn({'resumed': resumed, 'checkpoint_avail': checkpoint_avail,
                     'checkpoint_val': checkpoint_val,
                     'start_epoch': self._start_epoch,
                     'total_epochs': self._epochs})
        logger.info({'aux_weight': self._aux_weight,
                     'grad_clip': self._grad_clip,
                     'drop_path_prob': self._drop_path_prob,
                     'validation_freq': self._validation_freq})

        if self._start_epoch >= self._epochs:
            logger.warn(f'fit done because start_epoch {self._start_epoch}>={self._epochs}')
            return self.get_metrics() # we already finished the run, we might be checkpointed

        logger.pushd('epochs')
        for epoch in range(self._start_epoch, self._epochs):
            logger.pushd(epoch)
            self._set_drop_path(epoch, self._epochs)

            self.pre_epoch(train_dl, val_dl)
            self._train_epoch(train_dl)
            self.post_epoch(train_dl, val_dl)

            logger.popd()
        logger.popd()
        self.post_fit(train_dl, val_dl)

        # make sure we don't keep references to the graph
        del self._optim
        del self._sched


        logger.popd()
        return self.get_metrics()

    def create_optimizer(self)->Optimizer:
        return ml_utils.create_optimizer(self._conf_optim, self.model.parameters())

    def _create_scheduler(self, optim:Optimizer, steps_per_epoch:int) \
            ->Tuple[Optional[_LRScheduler],bool]:
        return ml_utils.create_lr_scheduler(self._conf_sched, self._epochs,
            optim, steps_per_epoch)

    def get_optimizer(self)->Optimizer:
        return self._optim
    def get_scheduler(self)->Optional[_LRScheduler]:
        return self._sched

    def get_metrics(self)->Metrics:
        return self._metrics

    #########################  hooks #########################
    def pre_fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        self._metrics.pre_run()

    def post_fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        self._metrics.post_run()

    def pre_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        self._metrics.pre_epoch(lr=self._optim.param_groups[0]['lr'])

    def post_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        val_metrics = None
        # first run test before checkpointing, otherwise we won't have val metrics
        if val_dl and self._tester and self._validation_freq > 0:
            if self._metrics.epochs() % self._validation_freq == 0 or \
                    self._metrics.epochs() >= self._epochs:
                val_metrics = self._tester.test(val_dl)

        # update val metrics
        self._metrics.post_epoch(val_metrics, lr=self._optim.param_groups[0]['lr'])

        # checkpoint if enabled with given freq or if this is the last epoch
        if self._checkpoint is not None and self._checkpoint.freq > 0 and \
                (self._metrics.epochs() % self._checkpoint.freq == 0 or \
                    self._metrics.epochs() >= self._epochs):
            self._checkpoint.new()
            self.update_checkpoint(self._checkpoint)
            self._checkpoint.commit()

    def pre_step(self, x:Tensor, y:Tensor)->None:
        self._metrics.pre_step(x, y)

    def post_step(self, x:Tensor, y:Tensor, logits:Tensor, loss:Tensor,
                  steps:int)->None:
        self._metrics.post_step(x, y, logits, loss, steps)
    #########################  hooks #########################

    def restore_checkpoint(self)->None:
        state = self._checkpoint['trainer']
        last_epoch = state['last_epoch']
        assert last_epoch >= 0 and last_epoch < self._epochs

        self._metrics.load_state_dict(state['metrics'])
        assert self._metrics.epochs() == last_epoch+1
        self._amp.load_state_dict(state['amp'])
        self.model.load_state_dict(state['model'])
        self._optim.load_state_dict(state['optim'])
        if self._sched:
            self._sched.load_state_dict(state['sched'])
        else:
            assert state['sched'] is None

        self._start_epoch = last_epoch + 1

    def update_checkpoint(self, checkpoint:CheckPoint)->None:
        # save all necessory state
        state = {
            'last_epoch': self._metrics.epochs()-1,
            'metrics': self._metrics.state_dict(),
            'model': self.model.state_dict(),
            'optim': self._optim.state_dict(),
            'sched': self._sched.state_dict() if self._sched else None,
            'amp': self._amp.state_dict()
        }
        self._checkpoint['trainer'] = state

    def _train_epoch(self, train_dl: DataLoader)->None:
        steps = len(train_dl)
        self.model.train()

        logger.pushd('steps')
        for step, (x, y) in enumerate(train_dl):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            logger.pushd(step)
            assert self.model.training # derived class might alter the mode

            self.pre_step(x, y)

            self._optim.zero_grad()

            logits, aux_logits = self.model(x), None
            tupled_out = isinstance(logits, Tuple) and len(logits) >=2
            if self._aux_weight:
                assert tupled_out, "aux_logits cannot be None unless aux tower is disabled"
            if tupled_out: # then we are using model created by desc
                logits, aux_logits = logits[0], logits[1]
            loss = self.compute_loss(self._lossfn, x, y, logits,
                                    self._aux_weight, aux_logits)

            self._amp.backward(loss, self._optim)

            # TODO: original darts clips alphas as well but pt.darts doesn't
            self._amp.clip_grad(self._grad_clip, self.model, self._optim)

            self._optim.step()
            if self._sched and not self._sched_on_epoch:
                self._sched.step()

            self.post_step(x, y, logits, loss, steps)
            logger.popd()

        if self._sched and self._sched_on_epoch:
            self._sched.step()
        logger.popd()

    def compute_loss(self, lossfn:Callable,
                     x:Tensor, y:Tensor, logits:Tensor,
                     aux_weight:float, aux_logits:Optional[Tensor])->Tensor:
        loss = lossfn(logits, y)
        if aux_weight > 0.0 and  aux_logits is not None:
            loss += aux_weight * lossfn(aux_logits, y)
        return loss

    def _set_drop_path(self, epoch:int, epochs:int)->None:
        if self._drop_path_prob:
            drop_prob = self._drop_path_prob * epoch / epochs
            # set value as property in model (it will be used by forward())
            # this is necessory when using DataParallel(model)
            # https://github.com/pytorch/pytorch/issues/16885
            m = self.model
            if hasattr(self.model, 'module'): # for data parallel model
                m = self.model.module
            if hasattr(m, 'drop_path_prob'):
                m.drop_path_prob(drop_prob)
            else:
                raise RuntimeError('Drop path value {} was specified but model'
                                   ' does not have drop_path_prob() method'\
                                       .format(self._drop_path_prob))
