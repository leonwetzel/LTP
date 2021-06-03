# dataloading.py
# author: Lukas Edman
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset 

IDX2POS = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ",
    "NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ",
    "SYM","VERB","X"]

POS2IDX = {pos:idx for idx, pos in enumerate(IDX2POS)}

IGNORE_IDX = -100

class POSDataset(Dataset):
    def __init__(self, data_filepath, labels_filepath, tokenizer):
        super().__init__()

        data_file = open(data_filepath, 'r', encoding='utf8')
        if labels_filepath is not None:
            labels_file = open(labels_filepath, 'r', encoding='utf8')

        data = []
        labels = []
        

        for i, line in enumerate(data_file):
            tokens = tokenizer.tokenize(line)
            data_idxs = tokenizer.encode(line)
            if len(data_idxs) == 2:
                if labels_filepath is not None:
                    labels_file.readline()
                continue

            data.append(data_idxs)

            labels_idxs = [IGNORE_IDX]  # bos token
            if labels_filepath is not None:
                labels_line = labels_file.readline().split()
                idx = 0
                for token in tokens:
                    if "##" in token:  # e.g. "hel ##lo wo ##rld"
                        labels_idxs.append(IGNORE_IDX)
                    else:
                        labels_idxs.append(POS2IDX[labels_line[idx]])
                        idx += 1
            else:
                for token in tokens:
                    if "##" in token:
                        labels_idxs.append(IGNORE_IDX)
                    else:
                        labels_idxs.append(100)  # doesn't matter, as long as it's not IGNORE_IDX
                
            labels_idxs.append(IGNORE_IDX) # eos token
            # assert len(labels_idxs) == len(data_idxs), str(i) + " " + str(len(labels_idxs)) + " " + str(len(data_idxs))
            labels.append(labels_idxs)

        self.data = np.array(data)
        self.labels = np.array(labels)


    def __getitem__(self, index):
        return torch.Tensor(self.data[index]).long(), torch.Tensor(self.labels[index]).long()

    def __len__(self):
        return len(self.data)

def padding_collate_fn(batch):
    """ Pads data with zeros to size of longest sentence in batch. """
    data, labels = zip(*batch)
    largest_sample = max([len(d) for d in data])
    padded_data = torch.zeros((len(data), largest_sample), dtype=torch.long)
    padded_labels = torch.full_like(padded_data, IGNORE_IDX)
    for i, sample in enumerate(data):
        padded_data[i, :len(sample)] = sample
        padded_labels[i, :len(sample)] = labels[i]

    return padded_data, padded_labels
