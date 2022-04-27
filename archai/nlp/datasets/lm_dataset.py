from torch.utils.data import Dataset

class LMDataset(Dataset):
    def __init__(self, input_ids, bsz, bptt, device='cpu', mem_len=None, ext_len=None, warmup=False):
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0
        self.mem_len = mem_len
        self.warmup = warmup
        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
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