from pathlib import Path
from typing import Optional

from overrides import overrides
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from archai.discrete_search.api import ModelEvaluator, DatasetProvider, ArchaiModel
from .pl_trainer import SegmentationTrainingLoop


class PartialTrainingValIOU(ModelEvaluator):
    def __init__(self, output_dir: str, tr_epochs: float = 1.0,
                 batch_size: int = 16, lr: float = 2e-4,
                 tr_dl_workers: int = 8, val_dl_workers: int = 8,
                 val_check_interval: float = 0.1):
        self.output_dir = Path(output_dir)
        self.tr_epochs = tr_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.tr_dl_workers = tr_dl_workers
        self.val_dl_workers = val_dl_workers
        self.val_check_interval = val_check_interval

    @overrides
    def evaluate(self, model: ArchaiModel, dataset_provider: DatasetProvider, 
                 budget: Optional[float] = None) -> float:
        
        tr_dataset = dataset_provider.get_train_dataset()
        val_dataset = dataset_provider.get_val_dataset()

        tr_dataloader = DataLoader(
            tr_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.tr_dl_workers
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.val_dl_workers
        )

        trainer = Trainer(
            default_root_dir=str(self.output_dir), gpus=1, 
            val_check_interval=int(self.val_check_interval * len(tr_dataloader)),
            max_steps=int(self.tr_epochs * len(tr_dataloader)),
        )

        trainer.fit(
            SegmentationTrainingLoop(model.arch, lr=self.lr),
            tr_dataloader, val_dataloader,
        )

        return trainer.validate(trainer.model, val_dataloader)[0]['validation_mIOU']
