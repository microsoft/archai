from torch.utils.data import Dataset

class LMDataset(Dataset):
    """ Implements a simple Pytorch Dataset the takes in a rank-1 Tensor (input_ids) with all
    tokens ids and produces one sample at a time with the sequence length of bptt.

    It is a simpler alternative to archai.nlp.datasets.LMOrderedIterator to be used
    only with diffp or where additional LMOrderedIterator parameters like 
    mem_len, ext_len, warm_up have no impact. This was needed to be compatible with
    Opacus diffp implementation and allows for Poisson sampling.
    """

    def __init__(self, input_ids, bptt):
        self.bptt = bptt

        # Work out how cleanly we can divide the dataset into bptt parts.
        self.n_samples = (input_ids.size(0) // bptt) - 1

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        # +1 to include the label
        self.input_ids = input_ids[:self.n_samples * bptt + 1].pin_memory()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        beg_idx = idx * self.bptt
        end_idx = beg_idx + self.bptt

        input_ids = self.input_ids[beg_idx:end_idx]
        labels = self.input_ids[beg_idx+1:end_idx+1]
        return input_ids, labels, self.bptt, True