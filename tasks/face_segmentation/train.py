from pathlib import Path
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer

from archai.datasets.cv.face_synthetics import FaceSyntheticsDatasetProvider
from archai.discrete_search.search_spaces.config import ArchConfig
from search_space.hgnet import StackedHourglass
from training.pl_trainer import SegmentationTrainingLoop


parser = ArgumentParser()
parser.add_argument('arch', type=Path)
parser.add_argument('--dataset_dir', type=Path, help='Face Synthetics dataset directory.', required=True)
parser.add_argument('--output_dir', type=Path, help='Output directory.', required=True)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--val_check_interval', type=float, default=0.1)


if __name__ == '__main__':
    args = parser.parse_args()

    arch_config = ArchConfig.from_file(args.arch)
    model = StackedHourglass(arch_config, num_classes=18)

    pl_model = SegmentationTrainingLoop(model, lr=args.lr)
    dataset_prov = FaceSyntheticsDatasetProvider(args.dataset_dir)

    tr_dl = torch.utils.data.DataLoader(
        dataset_prov.get_train_dataset(), batch_size=args.batch_size, num_workers=8,
        shuffle=True
    )

    val_dl = torch.utils.data.DataLoader(
        dataset_prov.get_val_dataset(), batch_size=args.batch_size, num_workers=8
    )

    trainer = Trainer(
        default_root_dir=str(args.output_dir), gpus=1, 
        val_check_interval=args.val_check_interval,
        max_epochs=args.epochs
    )

    trainer.fit(pl_model, tr_dl, val_dl)

    val_result = trainer.validate(trainer.model, val_dl)
    print(val_result)

    trainer.save_checkpoint(args.output_dir / 'final_model.ckpt')
