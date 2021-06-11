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
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

DATA_PATH = "data/PSP_data.csv"

IDX2LABEL = ['Non-offensive', 'Offensive']
LABEL2IDX = {label: i for i, label in enumerate(IDX2LABEL)}


def preprocessing_dataset(data_file):
    # load dataset
    df = pd.read_csv(data_file, sep=',', quotechar='"')

    # clean dataframe from missing values
    df = df[df['text'].notnull()]
    df = df[df['Category'].notnull()]

    # convert multi class labels to binairy labels
    df.loc[df['Category'] != "None", "Category"] = 'Offensive'
    df.loc[df['Category'] == "None", "Category"] = 'Non-offensive'

    return df


class SentenceDataset(Dataset):
    def __init__(self, data_frame, tokenizer):
        super().__init__()
        print(f"Loading data..")
        # load tokenizer
        self.tokenizer = tokenizer

        # get data and labels
        data, labels = self._extract(data_frame)

        self.data = np.array(data)
        self.labels = np.array(labels)

    def _extract(self, df):
        # obtain right columns from the dataset
        largest_sample = max([len(self.tokenizer.tokenize(i)) for i in df['text']])

        if largest_sample > 512:
            max_length = 512
        else:
            max_length = largest_sample

        print(f"Encoding data...")
        data = [
            self.tokenizer.encode(i, padding='max_length', max_length=max_length,
                                  truncation=True) for i in df['text']
        ]

        labels = [i for i in df['Category']]

        # transform labels from dataset to right format
        one_hot_labels = convert_to_one_hot(labels, len(IDX2LABEL))

        return data, one_hot_labels

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
    print("Values: {}".format(x.data))


def convert_to_one_hot(Y, label_size):
    out = []
    for instance in Y:
        multi_label = instance.split(" ")
        one_hot = np.zeros(label_size, dtype=int)
        for l in multi_label:
            one_hot[LABEL2IDX[l]] = 1
        out.append(one_hot)
    return np.array(out)


def baseline_data(dataframe, tokenizer):
    largest_sample = max([len(tokenizer.tokenize(i)) for i in dataframe['text']])

    if largest_sample > 512:
        max_length = 512
    else:
        max_length = largest_sample

    data = [
        tokenizer.encode(i, padding='max_length', max_length=max_length,
                         truncation=True) for i in dataframe['text']
    ]

    labels = []
    for i in dataframe['Category']:
        if i == "Non-offensive":
            labels.append(0)
        else:
            labels.append(1)

    print(labels)
    data = np.array(data)
    labels = np.array(labels)
    print(labels)

    return data, labels
