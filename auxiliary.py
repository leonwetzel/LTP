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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

class SentenceDataset(Dataset):
    def __init__(self, data_file, country, tokenizer):
        super().__init__()
        # load dataset
        df = pd.read_csv(data_file, sep=',', quotechar='"')

        # get data from given country parameter
        df = df.loc[df['Country'] == country]

        # clean dataframe from missing values
        df = df[df['text'].notnull()]
        df = df[df['Category'].notnull()]

        # get data and labels
        data, labels = self.extract(df)

        self.data = np.array(data)
        self.labels = np.array(labels)


    def extract(self, df):
        # obtain right columns from the dataset
        data = [tokenizer.tokenize(i) for i in df['text']]
        indexes = [tokenizer.encode(i) for i in df['text']] # moet dit dan data zijn? en wordt tokenizer hier dan wel meegenomen?
        labels = [i for i in df['Category']]

        all_labels = ['None', 'Anti-immigrant', 'Anti-muslim', 'Anti-semitic', 'Sexist', 'Homophobic', 'Other']
        label2idx = {label: i for i, label in enumerate(all_labels)}

        # transform labels from dataset to right format
        one_hot_multi_label = self.convert_to_one_hot(labels,label2idx,len(all_labels))

        return indexes, one_hot_multi_label

    def convert_to_one_hot(self, Y, label2idx, label_size):
        out = []
        for instance in Y:
            multi_label = instance.split(" ")
            one_hot = np.zeros(label_size, dtype=int)
            for l in multi_label:
                one_hot[label2idx[l]] = 1
            out.append(one_hot)
        return np.array(out)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = torch.Tensor(self.data[index]).long(), torch.Tensor(self.labels[index]).long()
        return item


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


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

fr_data = SentenceDataset('PSP_data.csv', 'France', tokenizer)
gr_data = SentenceDataset('PSP_data.csv', 'Germany', tokenizer)

fr_train, fr_rest = train_test_split(fr_data, test_size=0.2)
fr_dev, fr_test = train_test_split(fr_rest, test_size=0.5)

gr_train, gr_rest = train_test_split(gr_data, test_size=0.2)
gr_dev, gr_test = train_test_split(gr_rest, test_size=0.5)

fr_train_loader = DataLoader(fr_train, shuffle=False, batch_size=64)
fr_dev_loader = DataLoader(fr_dev, shuffle=False, batch_size=64)
fr_test_loader = DataLoader(fr_test, shuffle=False, batch_size=64)

gr_train_loader = DataLoader(gr_train, shuffle=False, batch_size=64)
gr_dev_loader = DataLoader(gr_dev, shuffle=False, batch_size=64)
gr_test_loader = DataLoader(gr_test, shuffle=False, batch_size=64)