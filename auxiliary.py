__author__ = "Jantina Schakel, Marieke Weultjes, Leon Wetzel," \
             " and Dion Theodoridis"
__copyright__ = "Copyright 2021, Jantina Schakel, Marieke Weultjes," \
                " Leon Wetzel, and Dion Theodoridis"
__credits__ = ["Jantina Schakel", "Marieke Weultjes", "Leon Wetzel",
                    "Dion Theodoridis"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Leon Wetzel"
__email__ = "l.f.a.wetzel@student.rug.nl"
__status__ = "Development"

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class SentenceDataset(Dataset):
    def __init__(self, data_file, tokenizer):
        super().__init__()

        self.corpus_frame = pd.read_csv(data_file)
        # TODO: juiste gegevens inladen

        # TODO: onderscheid maken in data en labels
        self.data = np.array()
        self.labels = np.array()

    def __len__(self):
        return len(self.corpus_frame)

    def __getitem__(self, index):
        # TODO: ervoor zorgen dat een Tensor-object geretourneerd wordt
        # voorbeeld: torch.Tensor(self.data[index]).long(), torch.Tensor(self.labels[index]).long()
        return index


def padding_collate_fn(batch):
    """
    Pads data with zeros to size of longest sentence in batch.

    Parameters
    ----------
    batch : tuple
        Part of the data that should be padded.

    Returns
    -------
    padded_data :
        asdladjkasd
    padded_labels :
        asdlsadjsadsad

    """
    data, labels = zip(*batch)
    largest_sample = max([len(d) for d in data])
    padded_data = torch.zeros((len(data), largest_sample), dtype=torch.long)
    padded_labels = torch.full_like(padded_data, -100)  # TODO: replace by valid ignore index
    for i, sample in enumerate(data):
        padded_data[i, :len(sample)] = sample
        padded_labels[i, :len(sample)] = labels[i]

    return padded_data, padded_labels
