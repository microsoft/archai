# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
import json
import pytorch_lightning as pl
from model import MyModel
from mnist_data_module import MNistDataModule
from store import ArchaiStore


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
    parser.add_argument("--subscription", type=str, help="subscription of workspace")
    parser.add_argument("--resource_group", type=str, help="resource group of workspace")
    parser.add_argument("--workspace", type=str, help="the workspace name")
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
    e = store.update_status(name, 'training')

    epochs = args.epochs

    try:
        model = MyModel.from_archid(args.model_params)
        if model is None:
            e['status'] = 'failed'
            e['error'] = 'invalid model parameters'
            store.update_status_entity(e)
            return

        e['nb_layers'] = model.nb_layers
        e['kernel_size'] = model.kernel_size
        e['hidden_dim'] = model.hidden_dim
        e['epochs'] = epochs
        if pipeline_id is not None:
            e['pipeline_id'] = pipeline_id

        store.update_status_entity(e)

        data = MNistDataModule(args.data_dir)
        logger = pl.loggers.TensorBoardLogger('logs', name='mnist')
        trainer = pl.Trainer(accelerator='gpu', max_epochs=1, logger=logger, log_every_n_steps=1)
        trainer.fit(model, data)
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
        store.update_status_entity(e)
        print(f"Training job completed successfully with validation accuracy {val_acc}")
    except Exception as ex:
        print(f"Training job failed with err {str(ex)}")
        e['status'] = 'failed'
        e['error'] = str(ex)
        store.update_status_entity(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
