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
from torch.utils.data import DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from auxiliary import SentenceDataset, DATA_PATH, IDX2LABEL, tensor_desc, baseline_data, preprocessing_dataset, \
    dividing_dataset, padding_collate_fn, OFFENSIVE, NOT_OFFENSIVE

parser = argparse.ArgumentParser(description="POS tagging")
parser.add_argument("--reload_model", type=str, help="Path of model to reload")
parser.add_argument("--save_model", type=str, help="Path for saving the model", default='model.pt')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--num_hidden_layers", type=int, default=1)
parser.add_argument("--num_attn_heads", type=int, default=1)
parser.add_argument("--output_file", type=str, help="Path for writing to a file", default='output.txt')
parser.add_argument("--undersampling", type=int, help="Set the use of undersampling the non-offensive class", default=0)
parser.add_argument("--sep_test_sets", type=int, default=1)


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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    print("Commencing training...")

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        total_sentences = 0.0
        for index, batch in enumerate(train_loader):
            print(f"(train) Processing batch {index} of epoch {epoch}...")
            data, labels = batch
            total_sentences += data.numel()

            # i. zero gradients
            optimizer.zero_grad()
            # ii. do forward pass
            y_pred = model(input_ids=data)
            # iii. get loss
            loss = criterion(y_pred.logits, labels.float())
            print(f"Loss: {loss}")
            # add loss to total_loss
            total_loss += loss.item()
            # iv. do backward pass
            loss.backward()
            # v. take an optimization step
            optimizer.step()

            if index % 5 == 0 and index > 0:
                avg_loss = total_loss / 5.0
                sentences_per_sec = total_sentences / (time.time() - start_time)
                print("[Epoch %d, Iter %d] avg loss: %.4f, sentences/sec: %d" % \
                    (epoch, index, avg_loss, sentences_per_sec))
                start_time = time.time()
                total_loss = 0.0
                total_sentences = 0.0

    print("############## END OF TRAINING ##############")
    accuracy, f1 = evaluate(model, valid_loader)
    print("Final Acc (valid): %.4f" % (accuracy))
    print("Final F1 (valid: %.4f" % (f1))

    if type(test_loader) is list:
        print("France:")
        accuracy, f1 = evaluate(model, test_loader[0])
        print("Final Acc (test): %.4f" % (accuracy))
        print("Final F1 (test): %.4f" % (f1))

        print("Italy:")
        accuracy, f1 = evaluate(model, test_loader[1])
        print("Final Acc (test): %.4f" % (accuracy))
        print("Final F1 (test): %.4f" % (f1))

        print("Germany:")
        accuracy, f1 = evaluate(model, test_loader[2])
        print("Final Acc (test): %.4f" % (accuracy))
        print("Final F1 (test): %.4f" % (f1))

        print("Switzerland:")
        accuracy, f1 = evaluate(model, test_loader[3])
        print("Final Acc (test): %.4f" % (accuracy))
        print("Final F1 (test): %.4f" % (f1))

    else:
        accuracy, f1 = evaluate(model, test_loader)
        print("Final Acc (test): %.4f" % (accuracy))
        print("Final F1 (test): %.4f" % (f1))


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
        total = 0.0
        tp, fp, tn, fn = 0, 0, 0, 0
        for index, batch in enumerate(loader):
            data, labels = batch
            output = model(input_ids=data)
            predictions = torch.argmax(output.logits, dim=-1)

            for i, pred in enumerate(predictions):
                label = labels[i]
                result = torch.zeros(2, dtype=torch.int64)
                print("pred: ", pred)
                print()
                result[pred] = 1
                print("result + label: ", result, label)
                print()
                if torch.equal(result, OFFENSIVE) and torch.equal(label, OFFENSIVE):
                    tp += 1
                elif torch.equal(result, NOT_OFFENSIVE) and torch.equal(label, NOT_OFFENSIVE):
                    tn += 1
                elif torch.equal(result, OFFENSIVE) and torch.equal(label, NOT_OFFENSIVE):
                    fn += 1
                elif torch.equal(result, NOT_OFFENSIVE) and torch.equal(label, OFFENSIVE):
                    fp += 1
                total += 1

            print(f"(evaluate) Processed batch {index} (tp={tp}, tn={tn}, fn={fn}, fp={fp})...")

    accuracy = (tp + tn) / total
    print(f"Accuracy: {accuracy}")
    precision = tp / (tp + fp)
    print(f"Precision: {precision}")
    recall = tp / (tp + fn)
    print(f"Recall: {recall}")
    f1 = 2 * precision * recall / (precision + recall)
    print(f"F1 score: {f1}")

    model.train()
    return accuracy, f1


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
    tokenizer.do_basic_tokenize = True  # TODO: change if needed

    model = BertForSequenceClassification.from_pretrained(pretrained)
    model.num_labels = len(IDX2LABEL)
    model.num_hidden_layers = 12
    model.hidden_act = "relu"
    model.use_cache = False

    # clean up the dataset
    DATA_FRAME = preprocessing_dataset(DATA_PATH)

    # Split the data in train, dev, test (old way)
    # train, rest = train_test_split(DATA_FRAME, test_size=0.2)  # lists []
    # dev, test = train_test_split(rest, test_size=0.5)

    # Check whether test sets should be separated per country
    separated = False if args.sep_test_sets == 0 else True
    # Split the data in train (70%), dev (20%) and test (10%) taking into account
    # that the data from different countries is evenly divided over the three sets
    if separated:
        training, dev, fr_test, it_test, de_test, ch_test = dividing_dataset(DATA_FRAME,
                                                                          sep_test_sets=separated,
                                                                          undersampling=args.undersampling)

        # Get the train and test data prepared for the baselines
        X_train, y_train = baseline_data(training, tokenizer)
        X_test_fr, y_test_fr = baseline_data(fr_test, tokenizer)
        X_test_it, y_test_it = baseline_data(it_test, tokenizer)
        X_test_de, y_test_de = baseline_data(de_test, tokenizer)
        X_test_ch, y_test_ch = baseline_data(ch_test, tokenizer)

        # Get the data prepared for the BERT model
        bert_data_train = SentenceDataset(training, tokenizer)
        bert_data_dev = SentenceDataset(dev, tokenizer)
        bert_data_test_fr = SentenceDataset(fr_test, tokenizer)
        bert_data_test_it = SentenceDataset(it_test, tokenizer)
        bert_data_test_de = SentenceDataset(de_test, tokenizer)
        bert_data_test_ch = SentenceDataset(ch_test, tokenizer)

        # Baseline model: most frequent with f1-score
        dummy_clf = DummyClassifier(strategy="most_frequent")
        dummy_clf.fit(X_train, y_train)
        dummy_pred_fr = dummy_clf.predict(X_test_fr)
        dummy_pred_it = dummy_clf.predict(X_test_it)
        dummy_pred_de = dummy_clf.predict(X_test_de)
        dummy_pred_ch = dummy_clf.predict(X_test_ch)

        print("Dummy Accuracy Score (fr): ", round(dummy_clf.score(X_test_fr, y_test_fr), 3))
        print('Dummy F1 score (fr):', round(f1_score(y_test_fr, dummy_pred_fr, average='weighted'), 3))

        print("Dummy Accuracy Score (it): ", round(dummy_clf.score(X_test_it, y_test_it), 3))
        print('Dummy F1 score (it):', round(f1_score(y_test_it, dummy_pred_it, average='weighted'), 3))

        print("Dummy Accuracy Score (de): ", round(dummy_clf.score(X_test_de, y_test_de), 3))
        print('Dummy F1 score (de):', round(f1_score(y_test_de, dummy_pred_de, average='weighted'), 3))

        print("Dummy Accuracy Score (ch): ", round(dummy_clf.score(X_test_ch, y_test_ch), 3))
        print('Dummy F1 score (ch):', round(f1_score(y_test_ch, dummy_pred_ch, average='weighted'), 3))


        # Baseline model: SVM
        baseline_clf = LinearSVC(max_iter=1000)
        X_train = np.array(X_train)

        baseline_clf.fit(X_train, y_train)
        # # France:
        print("Baseline model for France:")
        baseline_pred_fr = baseline_clf.predict(X_test_fr)
        print("Baseline Accuracy:", round(accuracy_score(y_test_fr, baseline_pred_fr),3))
        print('Baseline F1 score:', round(f1_score(y_test_fr, baseline_pred_fr, average='weighted'),3))
        print()
        #
        # # Italy:
        print("Baseline model for Italy:")
        baseline_pred_it = baseline_clf.predict(X_test_it)
        # # print(baseline_pred_it)
        print("Baseline Accuracy:", round(accuracy_score(y_test_it, baseline_pred_it),3))
        print('Baseline F1 score:', round(f1_score(y_test_it, baseline_pred_it, average='weighted'),3))
        print()
        #
        # # Germany:
        print("Baseline model for Germany:")
        baseline_pred_de = baseline_clf.predict(X_test_de)
        # # print(baseline_pred_de)
        print("Baseline Accuracy:", round(accuracy_score(y_test_de, baseline_pred_de),3))
        print('Baseline F1 score:', round(f1_score(y_test_de, baseline_pred_de, average='weighted'),3))
        print()
        #
        # # Switzerland:
        print("Baseline model for Switzerland:")
        baseline_pred_ch = baseline_clf.predict(X_test_ch)
        # # print(baseline_pred_ch)
        print("Baseline Accuracy:", round(accuracy_score(y_test_ch, baseline_pred_ch),3))
        print('Baseline F1 score:', round(f1_score(y_test_ch, baseline_pred_ch, average='weighted'),3))
        print()
        # print()

        # load the datasets
        train_loader = DataLoader(bert_data_train, shuffle=False,
                                  batch_size=64, collate_fn=padding_collate_fn)
        dev_loader = DataLoader(bert_data_dev, shuffle=False,
                                batch_size=64, collate_fn=padding_collate_fn)
        test_loader_fr = DataLoader(bert_data_test_fr, shuffle=False,
                                    batch_size=64, collate_fn=padding_collate_fn)
        test_loader_it = DataLoader(bert_data_test_it, shuffle=False,
                                    batch_size=64, collate_fn=padding_collate_fn)
        test_loader_de = DataLoader(bert_data_test_de, shuffle=False,
                                    batch_size=64, collate_fn=padding_collate_fn)
        test_loader_ch = DataLoader(bert_data_test_ch, shuffle=False,
                                    batch_size=64, collate_fn=padding_collate_fn)
        test_loader = [test_loader_fr, test_loader_it, test_loader_de, test_loader_ch]
    else:
        training, dev, test = dividing_dataset(DATA_FRAME, undersampling=args.undersampling)
        print()

        # use the train, dev and test datasets for different dataformats for the different experiments
        X_train, y_train = baseline_data(training, tokenizer)
        X_test, y_test = baseline_data(test, tokenizer)

        bert_data_train = SentenceDataset(training, tokenizer)
        bert_data_dev = SentenceDataset(dev, tokenizer)
        bert_data_test = SentenceDataset(test, tokenizer)

        # Baseline model: most frequent with f1-score
        dummy_clf = DummyClassifier(strategy="most_frequent")
        dummy_clf.fit(X_train, y_train)

        dummy_pred = dummy_clf.predict(X_test)
        print("Dummy Accuracy Score: ", round(dummy_clf.score(X_test, y_test),3))
        print('Dummy F1 score:', round(f1_score(y_test, dummy_pred, average='weighted'),3))

        # Baseline model: SVM
        baseline_clf = LinearSVC()  #C=1, class_weight={1: 10}?
        baseline_clf.fit(X_train, y_train)
        baseline_pred = baseline_clf.predict(X_test)
        # print(baseline_pred)
        print("Baseline Accuracy:", round(accuracy_score(y_test, baseline_pred),3))
        print('Baseline F1 score:', round(f1_score(y_test, baseline_pred, average='weighted'),3))

        # load the datasets
        train_loader = DataLoader(bert_data_train, shuffle=False,
                                  batch_size=64, collate_fn=padding_collate_fn)
        dev_loader = DataLoader(bert_data_dev, shuffle=False,
                                batch_size=64, collate_fn=padding_collate_fn)
        test_loader = DataLoader(bert_data_test, shuffle=False,
                                 batch_size=64, collate_fn=padding_collate_fn)

    # Load model weights from a file
    if args.reload_model:
        model.load_state_dict(torch.load(args.reload_model))

    # evaluate(model, dev_loader)
    train(model, train_loader, dev_loader, test_loader, args.epochs)

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
