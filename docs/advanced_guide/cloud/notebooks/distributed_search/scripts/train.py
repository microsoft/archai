import argparse
import json
import math
import torch
from torch import nn
from model import MyModel
from archai.datasets.cv.mnist_dataset_provider import MnistDatasetProvider
from store import ArchaiStore
import onnx


class Trainer:
    def __init__(self, training_epochs: float = 1.0, lr: float = 1e-4, device: str = 'cpu'):
        self.training_epochs = training_epochs
        self.device = device
        self.lr = lr
        self.model = None
        self.val_acc = None

    def evaluate(self, model, dataset_provider, store: ArchaiStore) -> float:
        # Loads the dataset
        tr_data = dataset_provider.get_train_dataset()
        val_data = dataset_provider.get_val_dataset()

        tr_dl = torch.utils.data.DataLoader(tr_data, batch_size=16, shuffle=True, num_workers=4)
        val_dl = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4)

        # Training settings
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        model.to(self.device)

        # Partial training
        epoch_iter = range(math.ceil(self.training_epochs))

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

    def upload_results(self, store: ArchaiStore):

        store.upload_model(self.model)
        store.upload_val_acc(self.val_acc


def main():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_key", required=True, type=str, help="Azure model store key")
    parser.add_argument("--storage_account_name", required=True, type=str, help="Azure model store name")
    parser.add_argument("--model_params", type=str, help="json string containing model parameters")
    parser.add_argument("--data_dir", type=str, help="location of dataset")
    parser.add_argument('--epochs', type=float, help='number of epochs to train', default=0.001)
    parser.add_argument("--output", type=str, help="place to write the results")

    args = parser.parse_args()

    store = ArchaiStore(args.storage_account_name, args.storage_key)
    model = MyModel.from_json(args.model_params)
    evaluator = Trainer(training_epochs=args.epochs, lr=1e-4, device='cuda')
    dataset_provider = MnistDatasetProvider(args.data_dir)
    val_acc = evaluator.evaluate(model, dataset_provider)

    config = json.load(open(args.model_params))
    config['vac_acc'] = val_acc
    config['epochs'] = args.epochs
    with open(args.output, 'w') as fp:
        json.dump(config, fp)

if __name__ == "__main__":
    main()
