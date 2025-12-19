import os
import numpy as np
import torch
from fairseq.data import FairseqDataset


class DepBiasDataset(FairseqDataset):
    """
    Wrap a base dataset; each sample adds dep_bias matrix from dep_dir/{idx}.npy
    dep_bias is aligned to BPE token length (src side).
    """

    def __init__(self, base_dataset: FairseqDataset, dep_dir: str):
        self.base = base_dataset
        self.dep_dir = dep_dir

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        bias_path = os.path.join(self.dep_dir, f"{idx}.npy")
        bias = np.load(bias_path).astype("float32")  # [T,T]
        item["dep_bias"] = torch.from_numpy(bias)
        return item

    def collater(self, samples):
        batch = self.base.collater(samples)
        if len(samples) == 0:
            return batch

        max_len = batch["net_input"]["src_tokens"].size(1)
        bsz = len(samples)

        # pad area fill with 1.0 (mul 中为中性；log 中 log(1)=0)
        dep = torch.ones(bsz, max_len, max_len, dtype=torch.float32)
        for i, s in enumerate(samples):
            m = s["dep_bias"]
            t = min(m.size(0), max_len)
            dep[i, :t, :t] = m[:t, :t]
        batch["net_input"]["dep_bias"] = dep
        return batch

    @property
    def sizes(self):
        return self.base.sizes

    def size(self, index):
        return self.base.size(index)

    def num_tokens(self, index):
        return self.base.num_tokens(index)

    def ordered_indices(self):
        return self.base.ordered_indices()

    def supports_prefetch(self):
        return self.base.supports_prefetch()

    def prefetch(self, indices):
        if self.supports_prefetch():
            self.base.prefetch(indices)