from typing import Tuple

from torch.optim.optimizer import Optimizer
from torch import Tensor, nn

from .common import logger

class Amp:
    _warning_shown = False
    def __init__(self, use_amp:bool)->None:
        self._use_amp = use_amp
        self._amp = None

        if self._use_amp:
            try:
                from apex import amp
                self._amp = amp
                logger.warn({'apex': True})
            except ModuleNotFoundError:
                if not Amp._warning_shown:
                    logger.warn({'apex': False})
                    Amp._warning_shown = True
                self._amp = None
        else:
            pass # do not disable if already enabled as other callers may be using it

    def available(self)->bool:
        return self._amp is not None

    def backward(self, loss:Tensor, optim:Optimizer)->None:
        if self._amp:
            with self._amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def to_amp(self, model:nn.Module, optim:Optimizer, opt_level="O2",
            keep_batchnorm_fp32=True, loss_scale="dynamic")\
                ->Tuple[nn.Module, Optimizer]:
        if self._amp:
            model, optim = self._amp.initialize(
                model, optim, opt_level=opt_level,
                keep_batchnorm_fp32=keep_batchnorm_fp32, loss_scale=loss_scale
            )
        return model, optim

    def clip_grad(self, clip:float, model:nn.Module, optim:Optimizer)->None:
        if clip > 0.0:
            if self._amp:
                nn.utils.clip_grad_norm_(self._amp.master_params(optim), clip)
            else:
                nn.utils.clip_grad_norm_(model.parameters(), clip)

    def state_dict(self):
        if self._amp:
            return self._amp.state_dict()
        else:
            return None

    def load_state_dict(self, state_dict):
        if self._amp:
            if state_dict is None:
                raise RuntimeError('checkpoint state_dict is None but Nvidia apex (amp) is enabled')
            self._amp.load_state_dict()
        else:
            if state_dict is not None:
                raise RuntimeError('checkpoint state_dict is not None but Nvidia apex (amp) is not enabled')
            else:
                pass

