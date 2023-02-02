import torch
from pytorch_lightning import Trainer

from archai.datasets.cv.face_synthetics import FaceSyntheticsDatasetProvider

from search_space.hgnet import HgnetSegmentationSearchSpace
from training.pl_trainer import SegmentationTrainingLoop

ss = HgnetSegmentationSearchSpace(18, image_size=(256, 256), num_blocks=6)
model = ss.random_sample()

pl_model = SegmentationTrainingLoop(model.arch, (256, 256))
dataset_prov = FaceSyntheticsDatasetProvider('/data/face_synthetics/')

if __name__ == '__main__':
    tr_dl = torch.utils.data.DataLoader(
        dataset_prov.get_train_dataset(), batch_size=32, num_workers=8
    )

    val_dl = torch.utils.data.DataLoader(
        dataset_prov.get_val_dataset(), batch_size=32, num_workers=8
    )

    trainer = Trainer(default_root_dir='/tmp/test', gpus=1, val_check_interval=0.1)
    trainer.fit(pl_model, train_dataloaders=tr_dl, val_dataloaders=val_dl)
