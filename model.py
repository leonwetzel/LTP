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

"""
USEFUL WORK:
- BERT fine-tune on fake news detection: https://colab.research.google.com/drive/1P4Hq0btDUDOTGkCHGzZbAx1lb0bTzMMa?usp=sharing
- Preprocessing of Fake News Dataset: https://colab.research.google.com/drive/1xqkvuNDg0Opk-aZpicTVPNY4wUdC7Y7v?usp=sharing
- Summary of work above: https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b

"""


import time
import argparse

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from sklearn.model_selection import train_test_split

from auxiliary import SentenceDataset, DATA_PATH, IDX2LABEL, tensor_desc



parser = argparse.ArgumentParser(description="POS tagging")
parser.add_argument("--reload_model", type=str, help="Path of model to reload")
parser.add_argument("--save_model", type=str, help="Path for saving the model")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--num_hidden_layers", type=int, default=1)
parser.add_argument("--num_attn_heads", type=int, default=1)
parser.add_argument("--output_file", type=str, help="Path for writing to a file", default='output.txt')


def train(model, train_loader, valid_loader, test_loader, epochs=3):
    """
    Train the model, given various data splits and an epoch count.

    Parameters
    ----------
    model
    train_loader
    valid_loader
    test_loader
    epochs

    Returns
    -------
    None

    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # TODO: change/upgrade if needed
    criterion = nn.BCELoss(reduction='mean')

    print("Commencing training...")

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        total_sentences = 0.0
        for index, batch in enumerate(train_loader):
            data, labels = batch
            total_sentences += data.numel()

            # i. zero gradients
            optimizer.zero_grad()
            # ii. do forward pass
            # tensor_desc(data)
            # print()
            # tensor_desc(labels)

            y_pred = model(input_ids=data)
            # iii. get loss
            loss = criterion(y_pred, labels)
            # add loss to total_loss
            total_loss += loss.item()
            # iv. do backward pass
            loss.backward()
            # v. take an optimization step
            optimizer.step()

            if index % 5 == 0 and index > 0:
                avg_loss = total_loss / 5.0
                sentences_per_sec = total_sentences / (time.time() - start_time)
                print("[Epoch %d, Iter %d] loss: %.4f, toks/sec: %d" % \
                    (epoch, index, avg_loss, sentences_per_sec))
                start_time = time.time()
                total_loss = 0.0
                total_sentences = 0.0

    print("############## END OF TRAINING ##############")
    acc = evaluate(model, valid_loader)
    print("Final Acc (valid): %.4f" % (acc))
    acc = evaluate(model, test_loader)
    print("Final Acc (test): %.4f" % (acc))


def evaluate(model, loader):
    """
    Evaluate model, using data provided by a DataLoader object.

    Parameters
    ----------
    model
    loader

    Returns
    -------

    """
    model.eval()
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for batch in loader:
            # batch size = 64
            data, labels = batch
            output = model(input_ids=data)
            predictions = torch.argmax(output.logits, dim=-1)

            for i, pred in enumerate(predictions):
                result = torch.zeros(len(labels[i]), dtype=torch.int64)
                result[pred] = 1

                if torch.equal(result, labels[i]):
                    correct += 1

            total += len(data)

    print(f"Accuracy: {correct / total}")
    model.train()
    return correct / total


def write_to_file(model, loader, output_file):
    model.eval()
    with torch.no_grad():
        outputs = []
        for batch in loader:
            data, labels = batch
            out = model(input_ids=data)
            preds = out.logits.argmax(dim=-1)

            for b in range(preds.size(0)):
                sent = []
                for i in range(preds.size(1)):
                    pass
                    # if labels[b, i] != IGNORE_IDX:
                    #    sent.append(IDX2POS[preds[b, i]])
                outputs.append(" ".join(sent))

    with open(output_file, 'w', encoding='utf8') as file:
        for line in outputs:
            file.write(line + "\n")
    model.train()


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using {} device'.format(device))

    # For more info, see https://huggingface.co/bert-base-multilingual-cased
    # - uncased variant: https://huggingface.co/bert-base-multilingual-uncased
    pretrained = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    tokenizer.do_basic_tokenize = False  # TODO: change if needed

    model = BertForSequenceClassification.from_pretrained(pretrained)
    model.num_labels = len(IDX2LABEL)

    # TODO: add more countries for comparisons
    all_data = SentenceDataset(DATA_PATH, tokenizer)

    # Split the data in train, dev, test
    train, rest = train_test_split(all_data, test_size=0.2)
    dev, test = train_test_split(rest, test_size=0.5)

    # load the datasets
    train_loader = DataLoader(train, shuffle=False, batch_size=64)
    dev_loader = DataLoader(dev, shuffle=False, batch_size=64)
    test_loader = DataLoader(test, shuffle=False, batch_size=64)

    print("no errors")

    # Load model weights from a file
    if args.reload_model:
        model.load_state_dict(torch.load(args.reload_model))

    evaluate(model, dev_loader)
    train(model, train_loader, dev_loader, test_loader, args.epochs)

    # Write output to file
    if args.output_file:
        print("Writing output to file...")
        write_to_file(model, dev_loader, args.output_file)

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
