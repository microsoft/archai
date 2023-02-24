import argparse
from tqdm import tqdm
import math
import torch
from overrides import overrides
from pathlib import Path
from model import MyModel
from archai.datasets.cv.mnist_dataset_provider import MnistDatasetProvider
from archai.discrete_search.api.model_evaluator import ModelEvaluator
from archai.discrete_search.api.archai_model import ArchaiModel


class PartialTrainingValAccuracy(ModelEvaluator):
    def __init__(self, training_epochs: float = 1.0, lr: float = 1e-4, device: str = 'cpu',
                 progress_bar: bool = False):
        self.training_epochs = training_epochs
        self.device = device
        self.lr = lr
        self.progress_bar = progress_bar

    @overrides
    def evaluate(self, model, dataset_provider, budget = None) -> float:
        # Loads the dataset
        tr_data = dataset_provider.get_train_dataset()
        val_data = dataset_provider.get_val_dataset()

        tr_dl = torch.utils.data.DataLoader(tr_data, batch_size=16, shuffle=True, num_workers=4)
        val_dl = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4)

        # Training settings
        optimizer = torch.optim.Adam(model.arch.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        model.arch.train()
        model.arch.to(self.device)

        # Partial training
        epoch_iter = range(math.ceil(self.training_epochs))
        if self.progress_bar:
            epoch_iter = tqdm(epoch_iter, desc=f'Training model {model.archid}')

        for epoch_nb in epoch_iter:
            # Early stops for fractional values of training epochs (e.g, 0.2)
            early_stop = len(tr_dl) + 1
            if 0 < (self.training_epochs - epoch_nb) < 1:
                early_stop = int((self.training_epochs - epoch_nb) * len(tr_dl))

            for i, (x, y) in enumerate(tr_dl):
                if i >= early_stop:
                    break

                optimizer.zero_grad()

                pred = model.arch(x.to(self.device))
                loss = criterion(pred, y.to(self.device))

                loss.backward()
                optimizer.step()

        # Evaluates final model
        model.arch.eval()

        with torch.no_grad():
            val_pred, val_target = [], []

            for x, y in val_dl:
                val_pred.append(model.arch(x.to(self.device)).argmax(axis=1).to('cpu'))
                val_target.append(y.to('cpu'))

            val_pred, val_target = torch.cat(val_pred, axis=0), torch.cat(val_target, axis=0)
            val_acc = (val_pred.squeeze() == val_target.squeeze()).numpy().mean()

        # Returns model to cpu
        model.arch.cpu()

        return val_acc


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_params", type=str, help="encoded model parameters like '[5,3,32]'")
    parser.add_argument("--output", type=str, help="place to write the results")
    args = parser.parse_args()

    nb_layers, kernel_size, hidden_dim = eval(args.model_params)

    model = MyModel(nb_layers, kernel_size, hidden_dim)
    archai_model = ArchaiModel(arch=model, archid=ArchaiModel.get_archid())

    evaluator = PartialTrainingValAccuracy(training_epochs=0.2, lr=1e-4, device='cuda',))
    dataset_provider = MnistDatasetProvider()
    val_acc = evaluator.evaluate(archai_model, dataset_provider)

    with open(args.output, 'w') as f:
        f.write(f'{val_acc}')


