# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
import json
import math
import torch
from torch import nn
from model import MyModel
from archai.datasets.cv.mnist_dataset_provider import MnistDatasetProvider
from store import ArchaiStore


class Trainer:
    def __init__(self, training_epochs: float = 1.0, lr: float = 1e-4, device: str = 'cpu'):
        self.training_epochs = training_epochs
        self.device = device
        self.lr = lr
        self.model = None
        self.val_acc = None
        self.input_shape = None

    def train(self, model, dataset_provider, progress_bar=False) -> float:
        # Loads the dataset
        tr_data = dataset_provider.get_train_dataset()
        val_data = dataset_provider.get_val_dataset()

        self.input_shape = tr_data.data[0].shape

        tr_dl = torch.utils.data.DataLoader(tr_data, batch_size=16, shuffle=True, num_workers=4)
        val_dl = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4)

        # Training settings
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        model.to(self.device)

        # Partial training
        epoch_iter = range(math.ceil(self.training_epochs))
        if progress_bar:
            from tqdm import tqdm
            epoch_iter = tqdm(epoch_iter, desc=f'Training model {model.get_archid()}')

        for epoch_nb in epoch_iter:
            # Early stops for fractional values of training epochs (e.g, 0.2)
            early_stop = len(tr_dl) + 1
            if 0 < (self.training_epochs - epoch_nb) < 1:
                early_stop = int((self.training_epochs - epoch_nb) * len(tr_dl))

            for i, (x, y) in enumerate(tr_dl):
                if i >= early_stop:
                    break

                optimizer.zero_grad()

                pred = model(x.to(self.device))
                loss = criterion(pred, y.to(self.device))

                loss.backward()
                optimizer.step()

        # Evaluates final model
        model.eval()

        with torch.no_grad():
            val_pred, val_target = [], []

            for x, y in val_dl:
                val_pred.append(model(x.to(self.device)).argmax(axis=1).to('cpu'))
                val_target.append(y.to('cpu'))

            val_pred, val_target = torch.cat(val_pred, axis=0), torch.cat(val_target, axis=0)
            val_acc = (val_pred.squeeze() == val_target.squeeze()).numpy().mean()

        # Returns model to cpu
        model.cpu()

        self.model = model
        self.val_acc = val_acc
        return val_acc


def main():
    # input and output arguments
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

    cmd_line = '\n'.join(sys.argv)
    print(f'Training arguments: {cmd_line}')

    args = parser.parse_args()

    save_models = args.save_models
    output_folder = args.output

    if save_models and not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

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
        store.update_status_entity(e)

        trainer = Trainer(training_epochs=epochs, lr=1e-4, device='cuda')
        dataset_provider = MnistDatasetProvider(root=args.data_dir)
        val_acc = trainer.train(model, dataset_provider, progress_bar=True)

        shape = trainer.input_shape
        # add batch and channel dimensions
        shape = [1, 1] + list(shape)

        if save_models:
            # this writes the results to the output folder.
            model.export_onnx(shape, os.path.join(output_folder, 'model.onnx'))

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
