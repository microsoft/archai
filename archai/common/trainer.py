from typing import Callable, Tuple, Optional

from torch import nn, Tensor, torch
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
from ..common.apex_utils import ApexUtils


class Trainer(EnforceOverrides):
    def __init__(self, conf_train:Config, model:nn.Module,
                 checkpoint:Optional[CheckPoint])->None:
        # region config vars
        conf_lossfn = conf_train['lossfn']
        self._aux_weight = conf_train['aux_weight']
        self._grad_clip = conf_train['grad_clip']
        self._drop_path_prob = conf_train['drop_path_prob']
        self._logger_freq = conf_train['logger_freq']
        self._title = conf_train['title']
        self._epochs = conf_train['epochs']
        self._conf_optim = conf_train['optimizer']
        self._conf_sched = conf_train['lr_schedule']
        self.batch_chunks = conf_train['batch_chunks']
        conf_validation = conf_train['validation']
        conf_apex = conf_train['apex']
        self._validation_freq = 0 if conf_validation is None else conf_validation['freq']
        # endregion

        self._apex = ApexUtils(conf_apex, logger)

        self._checkpoint = checkpoint
        self.model = model

        self._lossfn = ml_utils.get_lossfn(conf_lossfn)
        # using separate apex for Tester is not possible because we must use
        # same distributed model as Trainer and hence they must share apex
        self._tester = Tester(conf_validation, model, self._apex) \
                        if conf_validation else None
        self._metrics:Optional[Metrics] = None

        self._droppath_module = self._get_droppath_module()
        if self._droppath_module is None and self._drop_path_prob > 0.0:
            logger.warn({'droppath_module': None})

        self._start_epoch = -1 # nothing is started yet

    def fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->Metrics:
        logger.pushd(self._title)

        self._metrics = Metrics(self._title, self._apex, logger_freq=self._logger_freq)

        # optimizers, schedulers needs to be recreated for each fit call
        # as they have state specific to each run
        optim = self.create_optimizer()
        # create scheduler for optim before applying amp
        self._sched, self._sched_on_epoch = self._create_scheduler(optim, len(train_dl))
        # before checkpoint restore, convert to amp
        self.model, self._optim = self._apex.to_amp(self.model, optim,
                                                          batch_size=train_dl.batch_size)

        self._lossfn = self._lossfn.to(self.get_device())

        self.pre_fit(train_dl, val_dl)

        # we need to restore checkpoint after all objects are created because
        # restoring checkpoint requires load_state_dict calls on these objects
        self._start_epoch = 0
        # do we have a checkpoint
        checkpoint_avail = self._checkpoint is not None
        checkpoint_val = checkpoint_avail and 'trainer' in self._checkpoint
        resumed = False
        if checkpoint_val:
            # restore checkpoint
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
                     'validation_freq': self._validation_freq,
                     'batch_chunks': self.batch_chunks})

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
        optim = ml_utils.create_optimizer(self._conf_optim, self.model.parameters())
        logger.info({'conf_optim': self._conf_optim})
        return optim

    def _create_scheduler(self, optim:Optimizer, steps_per_epoch:int) \
            ->Tuple[Optional[_LRScheduler],bool]:

        logger.info({'steps_per_epoch': steps_per_epoch,
                     'scheduler': self._conf_sched.to_dict()})

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
        if self._checkpoint is not None and self._apex.is_master() and \
            self._checkpoint.freq > 0 and (self._metrics.epochs() % self._checkpoint.freq == 0 or \
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

    def get_device(self):
        return self._apex.device

    def restore_checkpoint(self)->None:
        state = self._checkpoint['trainer']
        last_epoch = state['last_epoch']
        assert last_epoch >= 0 and last_epoch < self._epochs

        self._metrics.load_state_dict(state['metrics'])
        assert self._metrics.epochs() == last_epoch+1
        self._apex.load_state_dict(state['amp'])
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
            'amp': self._apex.state_dict()
        }
        self._checkpoint['trainer'] = state

    def _train_epoch(self, train_dl: DataLoader)->None:
        steps = len(train_dl)
        self.model.train()

        logger.pushd('steps')
        for step, (x, y) in enumerate(train_dl):
            logger.pushd(step)
            assert self.model.training # derived class might alter the mode

            self.pre_step(x, y)

            self._optim.zero_grad()

            # divide batch in to chunks if needed so it fits in GPU RAM
            if self.batch_chunks > 1:
                x_chunks, y_chunks = torch.chunk(x, self.batch_chunks), torch.chunk(y, self.batch_chunks)
            else:
                x_chunks, y_chunks = (x,), (y,)

            logits_chunks = []
            loss_sum, loss_count = 0.0, 0
            for xc, yc in zip(x_chunks, y_chunks):
                xc, yc = xc.to(self.get_device(), non_blocking=True), yc.to(self.get_device(), non_blocking=True)

                logits_c, aux_logits = self.model(xc), None
                tupled_out = isinstance(logits_c, Tuple) and len(logits_c) >=2
                # if self._aux_weight: # TODO: some other way to validate?
                #     assert tupled_out, "aux_logits cannot be None unless aux tower is disabled"
                if tupled_out: # then we are using model created by desc
                    logits_c, aux_logits = logits_c[0], logits_c[1]
                loss_c = self.compute_loss(self._lossfn, yc, logits_c,
                                        self._aux_weight, aux_logits)

                self._apex.backward(loss_c, self._optim)

                loss_sum += loss_c.item() * len(logits_c)
                loss_count += len(logits_c)
                logits_chunks.append(logits_c.detach().cpu())

            # TODO: original darts clips alphas as well but pt.darts doesn't
            self._apex.clip_grad(self._grad_clip, self.model, self._optim)

            self._optim.step()

            # TODO: we possibly need to sync so all replicas are upto date
            self._apex.sync_devices()

            if self._sched and not self._sched_on_epoch:
                self._sched.step()

            self.post_step(x, y,
                           ml_utils.join_chunks(logits_chunks),
                           torch.tensor(loss_sum/loss_count),
                           steps)
            logger.popd()

            # end of step

        if self._sched and self._sched_on_epoch:
            self._sched.step()
        logger.popd()

    def compute_loss(self, lossfn:Callable, y:Tensor, logits:Tensor,
                     aux_weight:float, aux_logits:Optional[Tensor])->Tensor:
        loss = lossfn(logits, y)
        if aux_weight > 0.0 and  aux_logits is not None:
            loss += aux_weight * lossfn(aux_logits, y)
        return loss

    def _get_droppath_module(self)->Optional[nn.Module]:
        m = self.model
        if hasattr(self.model, 'module'): # for data parallel model
            m = self.model.module
        if hasattr(m, 'drop_path_prob'):
            return m
        return None

    def _set_drop_path(self, epoch:int, epochs:int)->None:
        if self._drop_path_prob and self._droppath_module is not None:
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
