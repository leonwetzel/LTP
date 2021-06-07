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

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch
from torch.utils.data import DataLoader

import numpy as np

from auxiliary import SentenceDataset, padding_collate_fn


parser = argparse.ArgumentParser(description="POS tagging")
parser.add_argument("--reload_model", type=str, help="Path of model to reload")
parser.add_argument("--save_model", type=str, help="Path for saving the model")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--num_hidden_layers", type=int, default=1)
parser.add_argument("--num_attn_heads", type=int, default=1)
parser.add_argument("--output_file", type=str, help="Path for writing to a file", default='output_s3284174.txt')


def train(model, train_loader, valid_loader, test_loader, epochs=5):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        total_sentences = 0.0
        for index, batch in enumerate(train_loader):
            data, _ = batch
            total_sentences += data.numel()

            # i. zero gradients
            optimizer.zero_grad()
            # ii. do forward pass
            y_pred = model(data, labels=[])  # TODO: get labels somewhere
            # iii. get loss
            loss = y_pred.loss
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
    # TODO: look for good/better ways to evaluate particular task
    model.eval()
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for batch in loader:
            data, labels = batch
            out = model(input_ids=data)
            preds = out.logits.argmax(dim=-1)

            mask = labels != -100  # TODO: change or remove mask
            correct += (preds[mask] == labels[mask]).sum().item()
            total += labels[mask].numel()

    print(correct, total)
    model.train()
    return correct/total


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
    tokenizer.do_basic_tokenize = False  # TODO: check if this is really needed

    tokenizer = BertTokenizer.from_pretrained(pretrained)
    model = BertForSequenceClassification.from_pretrained(pretrained)
    model.to(device)

    # TODO: replace old data loading by functionality in auxiliary.py
    # TODO: check if collate function actually makes sense
    train_dataset = SentenceDataset("data/train.en", tokenizer)
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              collate_fn=padding_collate_fn,
                              batch_size=args.batch_size)
    valid_dataset = SentenceDataset("data/valid.en", tokenizer)
    valid_loader = DataLoader(valid_dataset,
                              collate_fn=padding_collate_fn,
                            batch_size=args.batch_size)
    test_dataset = SentenceDataset("data/test.en", tokenizer)
    test_loader = DataLoader(test_dataset,
                             collate_fn=padding_collate_fn,
                             batch_size=args.batch_size)

    config = BertConfig.from_pretrained(pretrained)
    # config.num_labels = len(IDX2POS)  # TODO: specify amount of possible labels
    # TODO: check if num_labels also applies to multi-label rows
    config.num_hidden_layers = args.num_hidden_layers
    config.num_attention_heads = args.num_attn_heads
    config.output_attentions = True

    # Load an untrained model
    model = BertForSequenceClassification(config)

    # Load model weights from a file
    if args.reload_model:
        model.load_state_dict(torch.load(args.reload_model))

    evaluate(model, valid_loader)
    train(model, train_loader, valid_loader, test_loader, args.epochs)

    # Write output to file
    if args.output_file:
        print("Writing output to file...")
        write_to_file(model, test_loader, args.output_file)

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)