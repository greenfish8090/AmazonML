import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from dataset import EmbeddingDataset
from model import Regressor


class Trainer(nn.Module):
    def __init__(self, args, pred=False):
        super(Trainer, self).__init__()
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_model(self.args.features)  # Creating Regressor
        if not pred:
            self.get_data(args)  # Creating Dataloaders for Train and Test
            self.get_training_utils()  # Creating Optimizer

        # Variables to track training
        self.loss = None
        self.mape = None
        self.best_mape = None
        self.last_mape = None

        if not os.path.exists(self.args.save):
            os.mkdir(self.args.save)

    # Creating regressor
    def get_model(self, features=768):
        self.regressor = Regressor(features).to(self.device)

    # Creating Dataloaders and Datasets
    def get_data(self, args):
        self.train_loader = DataLoader(
            EmbeddingDataset(args.train_csv, args.train_emb),
            batch_size=args.batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            EmbeddingDataset(self.args.val_csv, self.args.val_emb),
            batch_size=args.batch_size,
            shuffle=False
        )

    # Creating Optimizer
    def get_training_utils(self):
        self.optimizer = optim.Adam(self.regressor.parameters(), lr=self.args.lr, amsgrad=True)

    # Forward pass
    def forward(self, x):
        return self.regressor(x)

    # Training loop for one epoch using complete trainset
    def train_epoch(self, epoch):
        self.regressor.train()
        loss_fn = nn.MSELoss()
        losses = []
        for batch_idx, (emb, trg) in enumerate(self.train_loader):
            emb = emb.to(self.device)
            trg = trg.to(self.device)

            self.optimizer.zero_grad()
            output = self(emb)
            loss = loss_fn(output, trg)
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            if batch_idx % self.args.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(emb),
                        len(self.train_loader.dataset),
                        100.0 * batch_idx / len(self.train_loader),
                        loss.item(),
                    )
                )
        return np.mean(losses)

    # Testing loop for one epoch using complete val set
    def test(self, epoch):
        self.regressor.eval()
        mape = 0
        for _, (emb, trg) in enumerate(self.val_loader):
            emb = emb.to(self.device)
            trg = trg.to(self.device)
            output = self(emb)
            mape += torch.sum(torch.abs(output - trg) / torch.max(torch.abs(trg), 1e-7))

        print("MAPE Loss at epoch {} is {}%".format(epoch, mape))
        return mape

    # Complete Training Loop
    def train(self):
        self.losses = []
        self.mapes = []
        self.best_mape = 0
        for epoch in range(1, self.args.epochs + 1):
            loss = self.train_epoch(epoch)
            self.losses.append(loss)
            if epoch % self.args.test_interval == 0:
                mape = self.test(epoch)
                self.mapes.append(mape)
                if mape < self.best_mape:
                    self.save("best", epoch)
                    self.best_mape = mape

        self.last_mape = self.test("INTMAX")
        self.save("last", epoch)

    # Saving Model State
    def save(self, star, epoch):
        save_dict = {
            "model": self.regressor.state_dict(),
            "loss": self.loss,
            "mape": self.mape,
            "best_mape": self.best_mape,
            "last_mape": self.last_mape,
            "epoch": epoch,
        }
        torch.save(save_dict, "{}/{}.pt".format(self.args.save, star))
        print("{} model saved".format(star))

    # Loading Model State
    def load(self, path):
        self.regressor.load_state_dict(
            torch.load(path, map_location=torch.device(self.device))["model"]
        )


def main():
    parser = argparse.ArgumentParser(description="Null")
    parser.add_argument("--train_csv", default="dataset/split_train.csv", type=str)
    parser.add_argument(
        "--train_emb", default="dataset/bert_base_uncased_train_embeddings.npy", type=str
    )
    parser.add_argument("--val_csv", default="dataset/split_val.csv", type=str)
    parser.add_argument(
        "--val_emb", default="dataset/bert_base_uncased_val_embeddings.npy", type=str
    )
    parser.add_argument("--epochs", "-e", default=100, type=int)
    parser.add_argument("--lr", "-l", default=0.01, type=float)
    parser.add_argument("--batch_size", "-b", default=8, type=int)
    parser.add_argument("--features", "-f", default=384, type=int)
    parser.add_argument("--seed", "-r", default=421, type=int)
    parser.add_argument("--log_interval", "-q", default=5, type=int)
    parser.add_argument("--test_interval", "-t", default=1, type=int)
    parser.add_argument("--save", "-s", default="./", type=str)
    args = parser.parse_args()

    # python train.py -dtr dataset/bert_base_uncased_train_embeddings.npy -dte dataset/bert_base_uncased_train_embeddings.npy -e 200 -b 64 -f 768 -q 10000 -t 5 -s model/
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
