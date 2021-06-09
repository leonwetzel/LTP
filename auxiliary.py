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
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

DATA_PATH = "data/PSP_data.csv"

IDX2LABEL = ['None', 'Anti-immigrant', 'Anti-muslim', 'Anti-semitic', 'Sexist', 'Homophobic', 'Other']
LABEL2IDX = {label: i for i, label in enumerate(IDX2LABEL)}

IGNORE_IDX = -100
VALID_IDX = 100


class SentenceDataset(Dataset):
    def __init__(self, data_file, country, tokenizer):
        super().__init__()
        # load tokenizer
        self.tokenizer = tokenizer

        # load dataset
        df = pd.read_csv(data_file, sep=',', quotechar='"')

        # get data from given country parameter
        df = df.loc[df['Country'] == country]

        # clean dataframe from missing values
        df = df[df['text'].notnull()]
        df = df[df['Category'].notnull()]

        # get data and labels
        data, labels = self._extract(df)

        self.data = np.array(data)
        self.labels = np.array(labels)

    def _extract(self, df):
        # obtain right columns from the dataset
        largest_sample = max([len(self.tokenizer.tokenize(i)) for i in df['text']])

        if largest_sample > 512:
            max_length = 512
        else:
            max_length = largest_sample

        data = [
            self.tokenizer.encode(i, padding='max_length', max_length=max_length,
                                  truncation=True) for i in df['text']
        ]

        # data = [self.tokenizer.tokenize(i) for i in df['text']]
        # indexes = [self.tokenizer.encode(i) for i in df['text']]  # moet dit dan data zijn?
        labels = [i for i in df['Category']]

        # transform labels from dataset to right format
        one_hot_multi_label = convert_to_one_hot(labels, len(IDX2LABEL))

        return data, one_hot_multi_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.Tensor(self.data[index]).long(), torch.Tensor(self.labels[index]).long()


def tensor_desc(x):
    """ Inspects a tensor: prints its type, shape and content"""
    try:
        print("Type:   {}".format(x.type()))
    except AttributeError:
        print("Type:   {}".format(type(x)))
    try:
        print("Size:   {}".format(x.size()))
    except AttributeError:
        print("Size:   {}".format(len(x)))
    print("Values: {}".format(x.data[0]))


def convert_to_one_hot(Y, label_size):
    out = []
    for instance in Y:
        multi_label = instance.split(" ")
        one_hot = np.zeros(label_size, dtype=int)
        for l in multi_label:
            one_hot[LABEL2IDX[l]] = 1
        out.append(one_hot)
    return np.array(out)
