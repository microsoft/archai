from torch.utils.data import DataLoader

class ListDataset:
    def __init__(self, loader:DataLoader, batch_size) -> None:
        self._list_inputs = None
        self._list_targets = None

        for inputs, targets in loader:
            self._list_inputs.append(inputs.tolist())
            self._list_targets.append(targets)
