from typing import List, Union
import random
from pathlib import Path

import ray

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
from archai.algos.evolution_pareto_image_seg.model import SegmentationNasModel
from archai.algos.evolution_pareto_image_seg.face_synthetics_data import FaceSynthetics
import segmentation_models_pytorch as smp


def get_custom_overall_metrics(tp, fp, fn, tn, stage):
    gt_pos = (tp + fn).sum(axis=0)
    pd_pos = (tp + fp).sum(axis=0)

    tp_diag = tp.sum(axis=0)
    f1 = 2 * tp_diag / torch.maximum(torch.ones_like(gt_pos), gt_pos + pd_pos)
    iou = tp_diag / torch.maximum(torch.ones_like(gt_pos), gt_pos + pd_pos - tp_diag)

    weight = 1 / torch.sqrt(gt_pos[1:18])
    overall_f1 = torch.sum(f1[1:18] * weight) / torch.sum(weight)
    overall_iou = torch.sum(iou[1:18] * weight) / torch.sum(weight)

    return {
        f'{stage}_overall_f1': overall_f1,
        f'{stage}_overall_iou': overall_iou
    }


class LightningModelWrapper(pl.LightningModule):
    def __init__(self,
                 model: SegmentationNasModel,
                 criterion_name: str = 'ce',
                 lr: float = 2e-4,
                 exponential_decay_lr: bool = True,
                 img_size: int = 256):

        super().__init__()

        self.model = model
        self.lr = lr
        self.exponential_decay_lr = exponential_decay_lr
        self.latency = None
        self.img_size = img_size
        
        self.set_loss(criterion_name)
        self.save_hyperparameters()

    def set_loss(self, criterion_name):
        if criterion_name == 'ce':
            self.loss_fn = smp.losses.SoftCrossEntropyLoss(ignore_index=255, smooth_factor=0)
        elif criterion_name == 'dice':
            self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True, ignore_index=255)
        elif criterion_name == 'lovasz':
            self.loss_fn = smp.losses.LovaszLoss(smp.losses.MULTICLASS_MODE, ignore_index=255, from_logits=True)

    def forward(self, image):
        return self.model(image)

    def shared_step(self, batch, stage='train'):
        image = batch['image']

        assert image.ndim == 4

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch['mask']
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        pred_classes = logits_mask.argmax(axis=1)

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_classes, mask.long(), mode='multiclass',
            num_classes=self.model.nb_classes, ignore_index=255
        )

        metrics_result = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'loss': loss
        }

        return metrics_result

    def training_step(self, batch, batch_idx):
        results = self.shared_step(batch, stage='train')
        # the sync_dist may have performance impact with ddp
        self.log_dict({'training_loss': results['loss']}, sync_dist=True)

        return results

    def predict(self, image):
        with torch.no_grad():
            return self.model.predict(image)

    def validation_step(self, batch, batch_idx):
        results = self.shared_step(batch, stage='validation')
        return results

    def validation_epoch_end(self, outputs):
        self.shared_epoch_end(outputs, stage='validation')

    def training_epoch_end(self, outputs):
        self.shared_epoch_end(outputs, stage='train')

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x['tp'] for x in outputs])
        fp = torch.cat([x['fp'] for x in outputs])
        fn = torch.cat([x['fn'] for x in outputs])
        tn = torch.cat([x['tn'] for x in outputs])
        avg_loss = torch.tensor([x['loss'] for x in outputs]).mean()

        results = get_custom_overall_metrics(tp, fp, fn, tn, stage=stage)
        results[f'{stage}_loss'] = avg_loss

        # TODO: enabling this causes error in lightning
        # when calling validate independently
        self.log_dict(results, sync_dist=True)
        return results

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # TODO: these magic values should get set as arguments
        # which come from config
        if self.exponential_decay_lr:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.973435286)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10000, eta_min=2e-7, verbose=False)

        scheduler = {
            'scheduler': scheduler,
            'interval': 'epoch'
        }

        return [optimizer], [scheduler]

    # def on_train_start(self) -> None:
    #     # These two lines generate a trace
    #     sample = torch.randn((1, 3, self.img_size, self.img_size)).to(self.device)
    #     self.logger.experiment.add_graph(self.model, sample)

    #     if self.latency:
    #         self.logger.log_metrics({'latency': self.latency})

    #     return

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Model parameters')
        parser.add_argument(
            '--arch', type=lambda x: Path(x) if x else x, default=None,
            help='Path to YAML architecture config file. If not provided, generates a random architecture.'
        )
        parser.add_argument('--nb_layers', type=int, default=12,
                            help='Number of layers in the network (used if no architecture is provided).')
        parser.add_argument('--max_downsample_factor', type=int, default=16,
                            help='Max downsample factor (used if no architecture is provided).')
        parser.add_argument('--no_skip_connections', action='store_false', default=True,
                            help='Whether to use skip connections (used if no architecture is provided).')
        parser.add_argument('--max_skip_connection_lenght', type=int, default=3,
                            help='Max skip connection length (used if no architecture is provided).')
        parser.add_argument('--operation_subset', nargs='*', type=str,
                            help='Subset of allowed operations (used if no architecture is provided).')
        parser.add_argument('--max_scale_delta', nargs='*', type=str,
                            help='Max scale diference between consecutive layers (used if no architecture is provided).')

        return parent_parser

@ray.remote(num_gpus=0.2)
class SegmentationTrainer():

    def __init__(self, model: SegmentationNasModel, dataset_dir: str,
                 max_steps: int = 12000, val_size: int = 2000,
                 img_size: int = 256,
                 augmentation: str = 'none', batch_size: int = 16,
                 lr: float = 2e-4, criterion_name: str = 'ce', gpus: int = 1,
                 val_check_interval: Union[int, float] = 0.25, seed: int = 1):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(int(seed))

        self.max_steps = max_steps
        self.val_check_interval = val_check_interval
        self.data_dir = Path(dataset_dir)
        self.tr_dataset = FaceSynthetics(self.data_dir, subset='train', val_size=val_size,
                                         img_size=(img_size, img_size), augmentation=augmentation)
        self.val_dataset = FaceSynthetics(self.data_dir, subset='validation', val_size=val_size,
                                          img_size=(img_size, img_size), augmentation=augmentation)

        self.tr_dataloader = DataLoader(self.tr_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

        self.model = LightningModelWrapper(model, criterion_name=criterion_name, lr=lr,
                                           exponential_decay_lr=True, img_size=img_size)
        self.img_size = img_size
        self.gpus = gpus

    def get_training_callbacks(self, run_dir: Path) -> List[pl.callbacks.Callback]:
        return [pl.callbacks.ModelCheckpoint(
            dirpath=str(run_dir / 'best_model'),
            mode='max', save_top_k=1, verbose=True,
            monitor='validation_overall_f1',
            filename='{epoch}-{step}-{validation_overall_f1:.2f}'
        )]

    def fit(self, run_path: str):
        run_path = Path(run_path)

        trainer = pl.Trainer(
            max_steps=self.max_steps,
            default_root_dir=run_path,
            gpus=self.gpus,
            callbacks=self.get_training_callbacks(run_path),
            val_check_interval=self.val_check_interval
        )

        trainer.fit(self.model, self.tr_dataloader, self.val_dataloader)

    def fit_and_validate(self, run_path: str)->float:
        # fit
        self.fit(run_path)

        # validate
        # TODO: am I doing this unnecessarily?
        # if I use validation_end_epoch I get error with self.log_dict
        outputs = []
        with torch.no_grad():
            for bi, b in enumerate(self.val_dataloader): 
                b['image'] = b['image'].to('cuda')
                b['mask'] = b['mask'].to('cuda')
                self.model.to('cuda')
                outputs.append(self.model.validation_step(b, bi))

        tp = torch.cat([x['tp'] for x in outputs])
        fp = torch.cat([x['fp'] for x in outputs])
        fn = torch.cat([x['fn'] for x in outputs])
        tn = torch.cat([x['tn'] for x in outputs])
        avg_loss = torch.tensor([x['loss'] for x in outputs]).mean()

        results = get_custom_overall_metrics(tp, fp, fn, tn, stage='validation')
        
        f1 = results['validation_overall_f1']
        return f1.to('cpu').item()
        


