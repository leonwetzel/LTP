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
from torch.utils.data import Dataset

DATA_PATH = "data/PSP_data.csv"
IDX2LABEL = ['Non-offensive', 'Offensive']
LABEL2IDX = {label: i for i, label in enumerate(IDX2LABEL)}
OFFENSIVE = torch.tensor([0, 1], dtype=torch.int64)
NOT_OFFENSIVE = torch.tensor([1, 0], dtype=torch.int64)


def preprocessing_dataset(data_file):
    # load dataset
    df = pd.read_csv(data_file, sep=',', quotechar='"')

    # clean dataframe from missing values
    df = df[df['text'].notnull()]
    df = df[df['Category'].notnull()]

    # identify not retrievable tweets
    indices = []
    for index, row in df.iterrows():
        if "The full tweet text was not retrievable." in row['text']:
            indices.append(index)

    # drop rows that contain not retrievable tweets
    df = df.drop(indices)

    # convert multi class labels to binairy labels
    df.loc[df['Category'] != "None", "Category"] = 'Offensive'
    df.loc[df['Category'] == "None", "Category"] = 'Non-offensive'

    return df


def dividing_dataset(dataframe, sep_test_sets=False, undersampling=0):
    # Split the data per country
    fr_df = dataframe.loc[dataframe['Country'] == 'France']
    it_df = dataframe.loc[dataframe['Country'] == 'Italy']
    de_df = dataframe.loc[dataframe['Country'] == 'Germany']
    ch_df = dataframe.loc[dataframe['Country'] == 'Switzerland']

    # check for undersampling and change de dataframes accordingly
    if undersampling == 1:
        fr_df = undersample(fr_df)
        it_df = undersample(it_df)
        de_df = undersample(de_df)
        ch_df = undersample(ch_df)

    # For each country, split the data in train (70%), dev (20%), test (10%)
    # Calculations:
    # - train:
    #   1 - 0.3 = 0.7 (70%)
    # - dev:
    #   1 - 0.7 = 0.3
    #   0.3 * (2/3) = 0.2 (20%)
    # - test:
    #   1 - 0.7 = 0.3
    #   0.3 * (1/3) = 0.1 (10%)

    # Splitting data from France:
    fr_train, fr_rest = train_test_split(fr_df, test_size=0.3, random_state=42)  # lists []
    fr_dev, fr_test = train_test_split(fr_rest, test_size=0.33, random_state=42)

    # Splitting data from Italy:
    it_train, it_rest = train_test_split(it_df, test_size=0.3, random_state=42)  # lists []
    it_dev, it_test = train_test_split(it_rest, test_size=0.33, random_state=42)

    # Splitting data from Germany:
    de_train, de_rest = train_test_split(de_df, test_size=0.3, random_state=42)  # lists []
    de_dev, de_test = train_test_split(de_rest, test_size=0.33, random_state=42)

    # Splitting data from Switzerland:
    ch_train, ch_rest = train_test_split(ch_df, test_size=0.3, random_state=42)  # lists []
    ch_dev, ch_test = train_test_split(ch_rest, test_size=0.33, random_state=42)

    """
    print("France:")
    print(fr_train['Category'].value_counts())
    print(fr_dev['Category'].value_counts())
    print(fr_test['Category'].value_counts())

    print("Italy:")
    print(it_train['Category'].value_counts())
    print(it_dev['Category'].value_counts())
    print(it_test['Category'].value_counts())

    print("Germany:")
    print(de_train['Category'].value_counts())
    print(de_dev['Category'].value_counts())
    print(de_test['Category'].value_counts())

    print("Swiss:")
    print(ch_train['Category'].value_counts())
    print(ch_dev['Category'].value_counts())
    print(ch_test['Category'].value_counts())
    """

    # Concatenate train sets:
    train = fr_train.append(it_train).append(de_train).append(ch_train)

    # Concatenate dev sets:
    dev = fr_dev.append(it_dev).append(de_dev).append(ch_dev)

    if sep_test_sets:
        return train, dev, fr_test, it_test, de_test, ch_test

    else:
        # Concatenate test sets:
        test = fr_test.append(it_test).append(de_test).append(ch_test)

        return train, dev, test


def undersample(df):
    off_df = df.loc[df['Category'] == 'Offensive']
    not_df = df.loc[df['Category'] == 'Non-offensive']
    off_count = len(off_df)

    not_count = (off_count // 3) * 7

    not_sample = not_df.sample(n=not_count, random_state=42)
    new_df = off_df.append(not_sample)

    return new_df


class SentenceDataset(Dataset):
    def __init__(self, data_frame, tokenizer):
        super().__init__()
        print(f"Loading data..")
        # load tokenizer
        self.tokenizer = tokenizer

        messages = [i for i in data_frame["text"]]
        labels = [i for i in data_frame['Category']]

        # transform data and labels from dataset to right format
        data = [self.tokenizer.encode(message) for message in messages]
        one_hot_labels = convert_to_one_hot(labels, len(IDX2LABEL))

        self.data = np.array(data)
        self.labels = np.array(one_hot_labels)

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
    # TODO: updaten ahv functionaliteit in constructor
    """
    largest_sample = max([len(tokenizer.tokenize(i)) for i in dataframe['text']])
    max_length = 512
    # if largest_sample > 512:
    #    max_length = 512
    # else:
    #    max_length = largest_sample

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

    data = np.array(data)
    labels = np.array(labels)
    """
    messages = [i for i in dataframe["text"]]

    data = [tokenizer.encode(message, padding='max_length', max_length=512, truncation=True) for message in messages]

    labels = []
    for i in dataframe['Category']:
        if i == "Non-offensive":
            labels.append(0)
        else:
            labels.append(1)


    return data, labels


def padding_collate_fn(batch):
    """
    Pads data with zeros to size of longest sentence in batch.
    """
    data, labels = zip(*batch)
    maximum = max([len(d) for d in data])
    if maximum > 512:
        largest_sample = 512
    else:
        largest_sample = maximum
    padded_data = torch.zeros((len(data), largest_sample), dtype=torch.long)
    padded_labels = torch.zeros((len(labels), 2), dtype=torch.long)
    counts = 0
    for i, sample in enumerate(data):
        sample_length = len(sample)
        if sample_length > 512:
            sample_length = 512
            sample = sample[:512]
            counts += 1
        padded_data[i, :sample_length] = sample
        padded_labels[i] = labels[i]
    print(f"{counts} case(s) of exceeding 512 limit (actual max size: {maximum})")
    return padded_data, padded_labels
