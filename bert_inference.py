import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

from dataset import TextDataset


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_set = TextDataset(path="dataset/split_train.csv")
    val_set = TextDataset(path="dataset/split_val.csv")
    test_set = TextDataset(path="dataset/test.csv", test=True)

    print(f"Train set size: {len(train_set)}")
    print(f"Val set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )
    print(f"Train loader size: {len(train_loader)}")
    print(f"Val loader size: {len(val_loader)}")
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )
    print(f"Test loader size: {len(test_loader)}")

    if args.model == "bert-base-uncased" or args.model == "bert-base-cased":
        tokenizer = BertTokenizer.from_pretrained(args.model)
        model = BertModel.from_pretrained(args.model).to(device)
    elif args.model == "roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained(args.model)
        model = RobertaModel.from_pretrained(args.model).to(device)
    train_embeddings = np.zeros((len(train_set), 768), dtype=np.float32)
    val_embeddings = np.zeros((len(val_set), 768), dtype=np.float32)
    test_embeddings = np.zeros((len(test_set), 768), dtype=np.float32)

    with torch.no_grad():
        model.eval()
        print("Train set")
        total = 0
        for i, (x, y) in enumerate(train_loader):
            B = len(x["string"])
            total += B
            print(i, total - B, total, flush=True)
            inp = tokenizer(
                x["string"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            inp = {k: v.to(device) for k, v in inp.items()}
            output = model(**inp)
            train_embeddings[total - B : total, :] = output[0][:, 0, :].detach().cpu().numpy()

            if i % 400 == 0:
                with open(
                    f"dataset/{args.model.replace('-', '_')}_train_embeddings_{i}.npy", "wb"
                ) as f:
                    np.save(f, train_embeddings)
                ## remove previous file
                try:
                    os.remove(
                        f"dataset/{args.model.replace('-', '_')}_train_embeddings_{i-400}.npy"
                    )
                except:
                    pass

        with open(f"dataset/{args.model.replace('-', '_')}_train_embeddings.npy", "wb") as f:
            np.save(f, train_embeddings)

        print("Val set")
        total = 0
        for i, (x, y) in enumerate(val_loader):
            B = len(x["string"])
            total += B
            print(i, total - B, total, flush=True)
            inp = tokenizer(
                x["string"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            inp = {k: v.to(device) for k, v in inp.items()}
            output = model(**inp)
            val_embeddings[total - B : total, :] = output[0][:, 0, :].detach().cpu().numpy()

            if i % 400 == 0:
                with open(
                    f"dataset/{args.model.replace('-', '_')}_val_embeddings_{i}.npy", "wb"
                ) as f:
                    np.save(f, val_embeddings)
                ## remove previous file
                try:
                    os.remove(f"dataset/{args.model.replace('-', '_')}_val_embeddings_{i-400}.npy")
                except:
                    pass

        with open(f"dataset/{args.model.replace('-', '_')}_val_embeddings.npy", "wb") as f:
            np.save(f, val_embeddings)

        print("Test set")
        total = 0
        for i, x in enumerate(test_loader):
            B = len(x["string"])
            total += B
            print(i, total - B, total, flush=True)
            inp = tokenizer(
                x["string"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            inp = {k: v.to(device) for k, v in inp.items()}
            output = model(**inp)
            test_embeddings[total - B : total, :] = output[0][:, 0, :].detach().cpu().numpy()

            if i % 400 == 0:
                with open(
                    f"dataset/{args.model.replace('-', '_')}_test_embeddings_{i}.npy", "wb"
                ) as f:
                    np.save(f, test_embeddings)
                ## remove previous file
                try:
                    os.remove(f"dataset/{args.model.replace('-', '_')}_test_embeddings_{i-400}.npy")
                except:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="roberta-base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
