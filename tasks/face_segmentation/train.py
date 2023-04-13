# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from argparse import ArgumentParser
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from archai.datasets.cv.face_synthetics import FaceSyntheticsDatasetProvider
from archai.discrete_search.search_spaces.config import ArchConfig
from search_space.hgnet import StackedHourglass
from training.pl_trainer import SegmentationTrainingLoop
from archai.common.store import ArchaiStore


def main():
    parser = ArgumentParser()
    parser.add_argument('arch', type=Path)
    parser.add_argument('--dataset_dir', type=Path, help='Face Synthetics dataset directory.', required=True)
    parser.add_argument('--output_dir', type=Path, help='Output directory.', required=True)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    parser.add_argument('--model_id', type=str, default=None)
    args = parser.parse_args()

    model_id = args.model_id
    store = None

    experiment_name = os.getenv("EXPERIMENT_NAME", "facesynthetics")
    con_str = os.getenv('MODEL_STORAGE_CONNECTION_STRING')
    if con_str is not None:
        storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
        store = ArchaiStore(storage_account_name, storage_account_key, table_name=experiment_name)

    storing = model_id is not None and store is not None:
    if storing:
        e = store.lock(model_id, 'training')
        pipeline_id = os.getenv('AZUREML_ROOT_RUN_ID')
        if pipeline_id is not None:
            e['pipeline_id'] = pipeline_id
            store.merge_status_entity(e)

    try:
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

        callbacks = [
            ModelCheckpoint(
                dirpath=str(args.output_dir / 'checkpoints'),
                monitor='validation_loss', mode='min',
                save_last=True, save_top_k=1, verbose=True,
                filename='{epoch}-{step}-{validation_loss:.2f}'
            )
        ]

        trainer = Trainer(
            default_root_dir=str(args.output_dir), accelerator='gpu',
            val_check_interval=args.val_check_interval,
            max_epochs=args.epochs,
            callbacks=callbacks
        )

        trainer.fit(pl_model, tr_dl, val_dl)

        val_result = trainer.validate(trainer.model, val_dl)
        print(val_result)

        trainer.save_checkpoint(args.output_dir / 'final_model.ckpt')

        # Save onnx model.
        input_shape = (1, 3, 256, 256)
        rand_range = (0.0, 1.0)
        export_kwargs = {'opset_version': 11}
        rand_min, rand_max = rand_range
        sample_input = ((rand_max - rand_min) * torch.rand(*input_shape) + rand_min).type("torch.FloatTensor")
        onnx_file = str(args.output_dir / 'final_model.onnx')
        torch.onnx.export(model, (sample_input,), onnx_file, input_names=["input_0"], **export_kwargs, )

        # post updated progress to our unified status table.
        if storing:
            metric = val_result[0]['validation_mIOU']
            e = store.get_status(model_id)
            e['val_iou'] = float(metric)
            e['status'] = 'completed'
            store.unlock_entity(e)

    except Exception as ex:
        # record failed state.
        if storing:
            e['status'] = 'failed'
            e['error'] = str(ex)
            store.unlock_entity(e)


if __name__ == '__main__':
    main()