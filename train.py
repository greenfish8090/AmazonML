import torch
from torch import nn
from model import Regressor
from torch.utils.data import DataLoader, Dataset
from torch import optim
import argparse
import numpy as np
import pandas as pd
import os
from dataset import EmbeddingDataset


class Trainer(nn.Module):
    def __init__(self, args, pred=False):
        super(Trainer, self).__init__()
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_model(self.args.features) # Creating Regressor
        if not pred:
            self.get_data(self.args.seed, self.args.batch_size) # Creating Dataloaders for Train and val
            self.get_training_utils() # Creating Optimizer

        if not os.path.exists(self.args.save):
            os.mkdir(self.args.save)

    # Creating regressor
    def get_model(self, features=768):
        self.regressor = Regressor(features).to(self.device, torch.float64)

    # Creating Dataloaders and Datasets
    def get_data(self, seed, batch_size):
        self.trainloader = DataLoader(EmbeddingDataset(self.args.train_data_path, self.args.train_csv_path), batch_size=batch_size, shuffle=True)
        self.valloader = DataLoader(EmbeddingDataset(self.args.val_data_path, self.args.val_csv_path), batch_size=batch_size, shuffle=False)

    # Creating Optimizer
    def get_training_utils(self):
        self.optimizer = optim.Adam(self.regressor.parameters(), lr=self.args.lr, amsgrad=True)

    # Forward pass
    def forward(self, x):
        return self.regressor(x)

    # Training loop for one epoch using complete trainset
    def train_epoch(self, epoch):
        self.regressor.train()
        epoch_loss = 0
        for batch_idx, (emb, val) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            output = self(emb.to(self.device, torch.float64)).squeeze(1)
            loss = nn.MSELoss()(output, val.to(self.device, torch.float64))
            epoch_loss += loss.item() 
            loss.backward()
            self.optimizer.step()
        print('Train Epoch: {} Loss: {:.6f}'.format(epoch, epoch_loss /(batch_idx + 1) ), flush=True)
        return epoch_loss / (batch_idx + 1)

    # valing loop for one epoch using complete valset
    def val(self, epoch):
        self.regressor.eval()
        mape = 0
        mse = 0
        for _, (emb, val) in enumerate(self.valloader):
            emb = emb.to(self.device, torch.float64)
            val = val.to(self.device, torch.float64)
            output = self(emb).squeeze(1)
            mape += torch.sum(torch.abs(output-val) / (torch.abs(val) + 1e-8)).item()
            mse += nn.MSELoss(reduction='sum')(output, val).item()
        print("MAPE Loss at epoch {} is {}% and MSE Loss is {}".format(epoch, 100 * mape/len(self.valloader.dataset), mse/len(self.valloader.dataset)), flush=True)
        return 100 * mape/len(self.valloader.dataset), mse/len(self.valloader.dataset)

    # Complete Training Loop
    def train(self):
        self.loss = []
        self.mape = []
        self.mse = []
        self.best_mse= 100000000000000
        for epoch in range(1, self.args.epochs + 1):
            loss = self.train_epoch(epoch)
            self.loss.append(loss)
            if epoch % self.args.val_interval == 0:
                mape, mse = self.val(epoch)
                self.mape.append(mape)
                self.mse.append(mse)
                if mse < self.best_mse:
                    self.save('best', epoch)
                    self.best_mse = mse

        self.last_mape, self.last_mse = self.val('INTMAX')
        self.save('last', epoch)

    # Saving Model State
    def save(self, star, epoch):
        save_dict = {
            'model':self.regressor.state_dict(), \
            'loss':self.loss, \
            'mape':self.mape, \
            'mse': self.mse, \
            'best_mse': self.best_mse, \
            'epoch': epoch,
            }
        torch.save(save_dict, "{}/{}.pt".format(self.args.save, star))
        print("{} model saved".format(star), flush=True)

    # Loading Model State
    def load(self, path):
        self.regressor.load_state_dict(torch.load(path, map_location=torch.device(self.device))['model'])

def main():
    parser = argparse.ArgumentParser(description='Null')
    parser.add_argument('--train_data_path', '-dtr', default='/home/pbalaji/AmazonML/dataset/bert_base_uncased_train_embeddings.npy', type=str)
    parser.add_argument('--val_data_path', '-dte', default='/home/pbalaji/AmazonML/dataset/bert_base_uncased_val_embeddings.npy', type=str)
    parser.add_argument('--train_csv_path', '-ctr', default='/home/pbalaji/AmazonML/dataset/split_train.csv', type=str)
    parser.add_argument('--val_csv_path', '-cte', default='/home/pbalaji/AmazonML/dataset/split_val.csv', type=str)
    parser.add_argument('--epochs', '-e', default=500, type=int)
    parser.add_argument('--lr', '-l', default=0.1, type=float)
    parser.add_argument('--batch_size', '-b', default=128, type=int)
    parser.add_argument('--features', '-f', default=768, type=int)
    parser.add_argument('--seed', '-r', default=421, type=int)
    parser.add_argument('--log_interval', '-q', default=10000, type=int)
    parser.add_argument('--val_interval', '-t', default=5, type=int)
    parser.add_argument('--save', '-s', default='models/', type=str)
    args = parser.parse_args()

    # python train.py -dtr dataset/bert_base_uncased_train_embeddings.npy -dte dataset/bert_base_uncased_train_embeddings.npy -e 200 -b 64 -f 768 -q 10000 -t 5 -s model/
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
