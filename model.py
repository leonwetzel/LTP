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
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC, LinearSVC

from auxiliary import SentenceDataset, DATA_PATH, IDX2LABEL, tensor_desc, baseline_data, preprocessing_dataset, dividing_dataset

parser = argparse.ArgumentParser(description="POS tagging")
parser.add_argument("--reload_model", type=str, help="Path of model to reload")
parser.add_argument("--save_model", type=str, help="Path for saving the model")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--num_hidden_layers", type=int, default=1)
parser.add_argument("--num_attn_heads", type=int, default=1)
parser.add_argument("--output_file", type=str, help="Path for writing to a file", default='output.txt')
parser.add_argument("--undersampling", type=int, help="Set the use of undersampling the non-offensive class", default=0)
parser.add_argument("--sep_test_sets", type=int, default="0")


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
    # criterion = nn.BCELoss(reduction='mean')  # TODO: change/upgrade if needed

    print("Commencing training...")

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        total_sentences = 0.0
        for index, batch in enumerate(train_loader):
            print(f"Processing batch {index} of epoch {epoch}...")
            data, labels = batch
            total_sentences += data.numel()

            # i. zero gradients
            optimizer.zero_grad()
            # ii. do forward pass
            y_pred = model(input_ids=data)
            # iii. get loss
            loss = F.binary_cross_entropy_with_logits(y_pred.logits, labels.float())
            # add loss to total_loss
            total_loss += loss.item()
            # iv. do backward pass
            loss.backward()
            # v. take an optimization step
            optimizer.step()

            if index % 5 == 0 and index > 0:
                avg_loss = total_loss / 5.0
                sentences_per_sec = total_sentences / (time.time() - start_time)
                print("[Epoch %d, Iter %d] loss: %.4f, sentences/sec: %d" % \
                    (epoch, index, avg_loss, sentences_per_sec))
                start_time = time.time()
                total_loss = 0.0
                total_sentences = 0.0

    print("############## END OF TRAINING ##############")
    acc = evaluate(model, valid_loader)
    print("Final Acc (valid): %.4f" % (acc))

    if type(test_loader) is list:
        print("France:")
        acc = evaluate(model, test_loader[0])
        print("Final Acc (test): %.4f" % (acc))

        print("Italy:")
        acc = evaluate(model, test_loader[1])
        print("Final Acc (test): %.4f" % (acc))

        print("Germany:")
        acc = evaluate(model, test_loader[2])
        print("Final Acc (test): %.4f" % (acc))

        print("Switzerland:")
        acc = evaluate(model, test_loader[3])
        print("Final Acc (test): %.4f" % (acc))

    else:
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
    print("Evaluating...")
    model.eval()
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for batch in loader:
            # batch size = 64
            data, labels = batch
            output = model(input_ids=data)
            predictions = torch.argmax(output.logits, dim=-1)

            # TODO: optimize for performance (speed)
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
    # TODO fix
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

    #print(f"Available devices: {torch.cuda.device_count()} ({torch.cuda.get_device_name(0)})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #"cuda"0
    print('Using {} device'.format(device))

    # For more info, see https://huggingface.co/bert-base-multilingual-cased
    # - uncased variant: https://huggingface.co/bert-base-multilingual-uncased
    pretrained = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    tokenizer.do_basic_tokenize = False  # TODO: change if needed

    model = BertForSequenceClassification.from_pretrained(pretrained)
    model.num_labels = len(IDX2LABEL)

    # clean up the dataset
    DATA_FRAME = preprocessing_dataset(DATA_PATH)

    # Split the data in train, dev, test (old way)
    # train, rest = train_test_split(DATA_FRAME, test_size=0.2)  # lists []
    # dev, test = train_test_split(rest, test_size=0.5)

    # Check whether test sets should be separated per country
    separated = False if args.sep_test_sets == 0 else True
    undersampling = args.undersampling
    # Split the data in train (70%), dev (20%) and test (10%) taking into account
    # that the data from different countries is evenly divided over the three sets
    if separated == True:
        train, dev, fr_test, it_test, de_test, ch_test = dividing_dataset(DATA_FRAME,sep_test_sets=separated, undersampling=undersampling)

        # use the train, dev and test datasets for different dataformats for the different experiments
        svm_X_train, svm_y_train = baseline_data(train, tokenizer)
        svm_X_test_fr, svm_y_test_fr = baseline_data(fr_test, tokenizer)
        svm_X_test_it, svm_y_test_it = baseline_data(it_test, tokenizer)
        svm_X_test_de, svm_y_test_de = baseline_data(de_test, tokenizer)
        svm_X_test_ch, svm_y_test_ch = baseline_data(ch_test, tokenizer)

        bert_data_train = SentenceDataset(train, tokenizer)
        bert_data_dev = SentenceDataset(dev, tokenizer)
        bert_data_test_fr = SentenceDataset(fr_test, tokenizer)
        bert_data_test_it = SentenceDataset(it_test, tokenizer)
        bert_data_test_de = SentenceDataset(de_test, tokenizer)
        bert_data_test_ch = SentenceDataset(ch_test, tokenizer)

        # Baseline model: SVM
        baseline_clf = LinearSVC()  #C=1, class_weight={1: 10}?
        baseline_clf.fit(svm_X_train, svm_y_train)

        # France:
        print("Baseline model for France:")
        baseline_pred_fr = baseline_clf.predict(svm_X_test_fr)
        # print(baseline_pred_fr)
        print("Baseline Accuracy:", accuracy_score(svm_y_test_fr, baseline_pred_fr))
        print('Baseline F1 score:', f1_score(svm_y_test_fr, baseline_pred_fr, average='weighted'))
        print()

        # Italy:
        print("Baseline model for Italy:")
        baseline_pred_it = baseline_clf.predict(svm_X_test_it)
        # print(baseline_pred_it)
        print("Baseline Accuracy:", accuracy_score(svm_y_test_it, baseline_pred_it))
        print('Baseline F1 score:', f1_score(svm_y_test_it, baseline_pred_it, average='weighted'))
        print()

        # Germany:
        print("Baseline model for Germany:")
        baseline_pred_de = baseline_clf.predict(svm_X_test_de)
        # print(baseline_pred_de)
        print("Baseline Accuracy:", accuracy_score(svm_y_test_de, baseline_pred_de))
        print('Baseline F1 score:', f1_score(svm_y_test_de, baseline_pred_de, average='weighted'))
        print()

        # Switzerland:
        print("Baseline model for Switzerland:")
        baseline_pred_ch = baseline_clf.predict(svm_X_test_ch)
        # print(baseline_pred_ch)
        print("Baseline Accuracy:", accuracy_score(svm_y_test_ch, baseline_pred_ch))
        print('Baseline F1 score:', f1_score(svm_y_test_ch, baseline_pred_ch, average='weighted'))
        print()
        print()

        # load the datasets
        train_loader = DataLoader(bert_data_train, shuffle=False, batch_size=64)
        dev_loader = DataLoader(bert_data_dev, shuffle=False, batch_size=64)
        test_loader_fr = DataLoader(bert_data_test_fr, shuffle=False, batch_size=64)
        test_loader_it = DataLoader(bert_data_test_it, shuffle=False, batch_size=64)
        test_loader_de = DataLoader(bert_data_test_de, shuffle=False, batch_size=64)
        test_loader_ch = DataLoader(bert_data_test_ch, shuffle=False, batch_size=64)
        test_loader = [test_loader_fr, test_loader_it, test_loader_de, test_loader_ch]

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
    else:
        train, dev, test = dividing_dataset(DATA_FRAME, undersampling=undersampling)


        # use the train, dev and test datasets for different dataformats for the different experiments
        svm_X_train, svm_y_train = baseline_data(train, tokenizer)
        svm_X_test, svm_y_test = baseline_data(test, tokenizer)

        bert_data_train = SentenceDataset(train, tokenizer)
        bert_data_dev = SentenceDataset(dev, tokenizer)
        bert_data_test = SentenceDataset(test, tokenizer)

        # Baseline model: most frequent with f1-score
        #dummy_clf = DummyClassifier(strategy="most_frequent")
        #dummy_clf.fit(svm_X_train, svm_y_train)
        #print(len(svm_X_test))
        #print(len(svm_y_test))
        #dummy_pred = dummy_clf.predict(svm_X_train)
        #print("Dummy Accuracy Score: ", dummy_clf.score(svm_X_train, svm_y_test))
        #print('Dummy F1 score:', f1_score(svm_y_train, dummy_pred, average='weighted'))

        # Baseline model: SVM
        baseline_clf = LinearSVC()  #C=1, class_weight={1: 10}?
        baseline_clf.fit(svm_X_train, svm_y_train)
        baseline_pred = baseline_clf.predict(svm_X_test)
        # print(baseline_pred)
        print("Baseline Accuracy:", accuracy_score(svm_y_test, baseline_pred))
        print('Baseline F1 score:', f1_score(svm_y_test, baseline_pred, average='weighted'))

        # load the datasets
        train_loader = DataLoader(bert_data_train, shuffle=False, batch_size=64)
        dev_loader = DataLoader(bert_data_dev, shuffle=False, batch_size=64)
        test_loader = DataLoader(bert_data_test, shuffle=False, batch_size=64)

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
