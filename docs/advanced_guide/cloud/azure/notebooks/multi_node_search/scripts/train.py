# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
import json
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from model import MyModel
from mnist_data_module import MNistDataModule
from archai.common.store import ArchaiStore
import mlflow
from shutil import copyfile


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in mlflow.MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


def main():
    """ This program trains a model, exports the model as onnx and updates the status of this
    training job in an Azure storage table. """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, type=str, help="The globally unique name of this model")
    parser.add_argument("--storage_account_key", required=True, type=str, help="Azure model store key")
    parser.add_argument("--storage_account_name", required=True, type=str, help="Azure model store name")
    parser.add_argument("--save_models", action='store_true', help="save models to azure storage")
    parser.add_argument("--model_params", type=str, help="json string containing model parameters")
    parser.add_argument("--data_dir", type=str, help="location of dataset", default='dataset')
    parser.add_argument('--epochs', type=float, help='number of epochs to train', default=0.001)
    parser.add_argument("--output", type=str, help="place to write the results", default='output')

    pipeline_id = os.getenv('AZUREML_ROOT_RUN_ID')

    args = parser.parse_args()

    save_models = args.save_models
    output_folder = args.output

    if save_models and not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    print(f'Training model: {args.name} with architecture {args.model_params}')

    name = args.name
    store = ArchaiStore(args.storage_account_name, args.storage_account_key)
    e = store.lock(name, 'training')

    epochs = args.epochs

    try:
        model = MyModel.from_archid(args.model_params)
        if model is None:
            e['status'] = 'failed'
            e['error'] = 'invalid model parameters'
            store.merge_status_entity(e)
            return

        e['nb_layers'] = model.nb_layers
        e['kernel_size'] = model.kernel_size
        e['hidden_dim'] = model.hidden_dim
        e['epochs'] = epochs
        if pipeline_id is not None:
            e['pipeline_id'] = pipeline_id

        store.merge_status_entity(e)

        data = MNistDataModule(args.data_dir)
        trainer = Trainer(accelerator='gpu', max_epochs=1, callbacks=[TQDMProgressBar(refresh_rate=100)])
        mlflow.pytorch.autolog(log_models=save_models, registered_model_name=name)
        with mlflow.start_run() as run:
            trainer.fit(model, data)
            print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

        result = trainer.validate(model, data)
        val_acc = result[0]['accuracy']

        if save_models:
            # this writes the results to the output folder.
            model.export_onnx(data.input_shape, os.path.join(output_folder, 'model.onnx'))
            config = {
                'name': name,
                'vac_acc': val_acc,
                'epochs': epochs,
                'nb_layers': model.nb_layers,
                'kernel_size': model.kernel_size,
                'hidden_dim': model.hidden_dim,
            }

            json_file = os.path.join(output_folder, 'results.json')
            with open(json_file, 'w') as fp:
                json.dump(config, fp)

        # post updated progress to our unified status table.
        e['val_acc'] = float(val_acc)
        e['status'] = 'completed'
        store.merge_status_entity(e)
        store.unlock(name)

        if os.path.isfile('model_summary.txt'):
            copyfile('model_summary.txt', os.path.join(output_folder, 'model_summary.txt'))

        print(f"Training job completed successfully with validation accuracy {val_acc}")
    except Exception as ex:
        print(f"Training job failed with err {str(ex)}")
        e['status'] = 'failed'
        e['error'] = str(ex)
        store.merge_status_entity(e)
        store.unlock(name)
        sys.exit(1)


if __name__ == "__main__":
    main()
