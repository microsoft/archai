# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from argparse import ArgumentParser
import os
import time
import torch
import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from archai.datasets.cv.face_synthetics import FaceSyntheticsDatasetProvider
from archai.discrete_search.search_spaces.config import ArchConfig
from search_space.hgnet import StackedHourglass
from training.pl_trainer import SegmentationTrainingLoop
from archai.common.store import ArchaiStore
from archai.common.config import Config


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in mlflow.MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


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
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--register', help="Specify whether to register the trained model with your mlflow workspace", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    store: ArchaiStore = None
    epochs = 1 if args.epochs < 1 else args.epochs

    start_time = time.time()
    storing = False
    config = args.config
    experiment_name = None
    if config and config.is_file():
        config = Config(str(config))
        if 'aml' in config:
            # we are running in azure ml.
            aml_config = config['aml']
            metric_key = config['training'].get('metric_key')
            connection_str = aml_config['connection_str']
            experiment_name = aml_config['experiment_name']
            storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(connection_str)
            store = ArchaiStore(storage_account_name, storage_account_key, table_name=experiment_name)
            storing = True

    try:
        if storing:
            print(f'Locking entity {model_id}')
            e = store.lock(model_id, 'training')
            if e is None:
                # force the reset of this lock so the training job can take it!
                # might be a left over from previous failed job.
                store.unlock_entity(store.get_status(model_id))
                e = store.lock(model_id, 'training')

            pipeline_id = os.getenv('AZUREML_ROOT_RUN_ID')
            if pipeline_id is not None:
                e['pipeline_id'] = pipeline_id
                store.merge_status_entity(e)

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
            max_epochs=epochs,
            callbacks=callbacks
        )

        mlflow.pytorch.autolog(log_models=args.register, registered_model_name=model_id)
        with mlflow.start_run() as run:
            trainer.fit(pl_model, tr_dl, val_dl)
            print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

        val_result = trainer.validate(trainer.model, val_dl)
        print(val_result)
        end_time = time.time()

        if storing:
            # post updated progress to our unified status table and unlock the row.
            metric = float(val_result[0]['validation_mIOU'])
            print(f"Storing {metric_key}={metric} for model {model_id}")
            e = store.get_status(model_id)
            e[metric_key] = metric
            e['status'] = 'complete'
            e['training_time'] = end_time - start_time
            store.unlock_entity(e)

        trainer.save_checkpoint(args.output_dir / 'model.ckpt')

        # Save onnx model.
        input_shape = (1, 3, 256, 256)
        rand_range = (0.0, 1.0)
        export_kwargs = {'opset_version': 11}
        rand_min, rand_max = rand_range
        sample_input = ((rand_max - rand_min) * torch.rand(*input_shape) + rand_min).type("torch.FloatTensor")
        onnx_file = str(args.output_dir / 'model.onnx')
        torch.onnx.export(model, (sample_input,), onnx_file, input_names=["input_0"], **export_kwargs, )

    except Exception as ex:
        # record failed state.
        if storing:
            e['status'] = 'failed'
            e['error'] = str(ex)
            store.unlock_entity(e)


if __name__ == '__main__':
    main()
