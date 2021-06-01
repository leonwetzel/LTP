# bert_pos.py
# author: Lukas Edman
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from dataloading import POSDataset, padding_collate_fn, IDX2POS, POS2IDX, IGNORE_IDX
from torch.utils.data import DataLoader 
from transformers import SqueezeBertConfig, SqueezeBertTokenizer, SqueezeBertForTokenClassification

parser = argparse.ArgumentParser(description="POS tagging")
parser.add_argument("--reload_model", type=str, help="Path of model to reload")
parser.add_argument("--save_model", type=str, help="Path for saving the model")
parser.add_argument("--output_file", type=str, help="Path for writing to a file")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--num_hidden_layers", type=int, default=1)
parser.add_argument("--num_attn_heads", type=int, default=1)

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
                    if labels[b, i] != IGNORE_IDX:
                        sent.append(IDX2POS[preds[b, i]])
                outputs.append(" ".join(sent))
            
    with open(output_file, 'w', encoding='utf8') as file:
        for line in outputs:
            file.write(line + "\n")
    model.train()


def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        correct = 0.0 
        total = 0.0
        for batch in loader:
            data, labels = batch
            out = model(input_ids=data)
            preds = out.logits.argmax(dim=-1)

            mask = labels != IGNORE_IDX
            correct += (preds[mask] == labels[mask]).sum().item()
            total += labels[mask].numel()


    print(correct, total)
    model.train()
    return correct/total

def train(model, train_loader, valid_loader, test_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        total_tokens = 0.0
        for i, batch in enumerate(train_loader):

            if i % 5 == 0 and i > 0:
                tps = total_tokens/ (time.time()-start_time)
                print("[Epoch %d, Iter %d] loss: %.4f, toks/sec: %d" % (epoch, i, total_loss/5, tps))
                start_time = time.time()
                total_loss = 0.0
                total_tokens = 0.0

            data, labels = batch
            optimizer.zero_grad()
            out = model(input_ids=data, labels=labels)
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()
            total_tokens += data.numel()

        acc = evaluate(model, valid_loader)
        print("[Epoch %d] Acc (valid): %.4f" % (epoch, acc))
        acc = evaluate(model, test_loader)
        print("[Epoch %d] Acc (test): %.4f" % (epoch, acc))
        start_time = time.time() # so tps stays consistent

    print("############## END OF TRAINING ##############")
    acc = evaluate(model, valid_loader)
    print("Final Acc (valid): %.4f" % (acc))
    acc = evaluate(model, test_loader)
    print("Final Acc (test): %.4f" % (acc))

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    args = parser.parse_args()
    args.save_model = "model.pt"
    
    pretrained = 'squeezebert/squeezebert-uncased'
    tokenizer = SqueezeBertTokenizer.from_pretrained(pretrained)
    tokenizer.do_basic_tokenize = False
    train_dataset = POSDataset("data/train.en", "data/train.en.label", tokenizer)
    train_loader = DataLoader(train_dataset,
        shuffle=True,
        collate_fn=padding_collate_fn,
        batch_size=args.batch_size)
    valid_dataset = POSDataset("data/valid.en", "data/valid.en.label", tokenizer)
    valid_loader = DataLoader(valid_dataset,
        collate_fn=padding_collate_fn,
        batch_size=args.batch_size)
    # test_dataset = POSDataset("data/test.en", "data/test.en.label", tokenizer)
    test_dataset = POSDataset("data/test.en", None, tokenizer)
    test_loader = DataLoader(test_dataset,
        collate_fn=padding_collate_fn,
        batch_size=args.batch_size)
    
    config = SqueezeBertConfig.from_pretrained(pretrained)
    config.num_labels = len(IDX2POS)
    config.num_hidden_layers = args.num_hidden_layers
    config.num_attention_heads = args.num_attn_heads
    config.output_attentions = True

    # Load an untrained model
    # model = SqueezeBertForTokenClassification(config)

    # Load a pretrained model
    model = SqueezeBertForTokenClassification.from_pretrained(
        pretrained, 
        num_labels=len(IDX2POS), 
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attn_heads,
        output_attentions=True)

    # Load model weights from a file
    args.reload_model = "model.pt"
    if args.reload_model:
        model.load_state_dict(torch.load(args.reload_model))

    evaluate(model, valid_loader)
    train(model, train_loader, valid_loader, test_loader, args.epochs)

    # args.output_file = "output.txt"
    # if args.output_file:
    #     write_to_file(model, test_loader, args.output_file)

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)