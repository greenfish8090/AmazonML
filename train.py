import torch
from torch import nn
from model import Regressor
from torch.utils.data import DataLoader, Dataset
from torch import optim
import argparse
import numpy as np
import pandas as pd
import os


class TDataset(Dataset):
    def __init__(self, path):
        self.data = np.load(path)
        self.values = pd.read_csv('dataset/train.csv')['PRODUCT_LENGTH'].values().tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.values[idx]

class Trainer(nn.Module):
    def __init__(self, args, pred=False):
        super(Trainer, self).__init__()
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        self.get_model(self.args.features) # Creating Regressor
        if not pred:
            self.get_data(self.args.seed, self.args.batch_size) # Creating Dataloaders for Train and Test
            self.get_training_utils() # Creating Optimizer

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
    def get_data(self, seed, batch_size):
        self.trainloader = DataLoader(TDataset(self.args.train_data_path), batch_size=batch_size, shuffle=True)
        self.testloader = DataLoader(TDataset(self.args.test_data_path), batch_size=batch_size, shuffle=False)

    # Creating Optimizer
    def get_training_utils(self):
        self.optimizer = optim.Adam(self.regressor.parameters(), lr=self.args.lr, amsgrad=True)

    # Forward pass 
    def forward(self, x):
        return self.regressor(x)

    # Training loop for one epoch using complete trainset
    def train_epoch(self, epoch):
        self.regressor.train()
        for batch_idx, (emb, val) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            output = self(emb)
            loss = nn.MSELoss()(output, val.to(self.device))
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(emb), len(self.trainloader.dataset),
                    100. * batch_idx / len(self.trainloader), loss.item()))
        return loss

    # Testing loop for one epoch using complete testset
    def test(self, epoch):
        self.regressor.eval()
        mape = 0
        for _, (emb, val) in enumerate(self.testloader):
            output = self(emb)
            mape += torch.sum(torch.abs(output-val) / torch.max(torch.abs(val), 1e-7))
            
        print("MAPE Loss at epoch {} is {}%".format(epoch, mape))
        return mape

    # Complete Training Loop
    def train(self):
        self.loss = []
        self.mape = []
        self.best_mape = 0
        for epoch in range(1, self.args.epochs + 1):
            loss = self.train_epoch(epoch)
            self.loss.append(loss)
            if epoch % self.args.test_interval == 0:
                mape = self.test(epoch)
                self.mape.append(mape)
                if mape < self.best_mape:
                    self.save('best', epoch)
                    self.best_mape = mape
            
        self.last_mape = self.test('INTMAX')
        self.save('last', epoch)

    # Saving Model State
    def save(self, star, epoch):
        save_dict = {
            'model':self.regressor.state_dict(), \
            'loss':self.loss, \
            'mape':self.mape, \
            'best_mape': self.best_mape, \
            'last_mape':self.last_mape, \
            'epoch': epoch,
            }
        torch.save(save_dict, "{}/{}.pt".format(self.args.save, star))
        print("{} model saved".format(star))
    
    # Loading Model State
    def load(self, path):
        self.regressor.load_state_dict(torch.load(path, map_location=torch.device(self.device))['model'])

def main():
    parser = argparse.ArgumentParser(description='Null')
    parser.add_argument('--train_data_path', '-dtr', default='', type=str)
    parser.add_argument('--test_data_path', '-dte', default='', type=str)
    parser.add_argument('--epochs', '-e', default=100, type=int)
    parser.add_argument('--lr', '-l', default=0.01, type=float)
    parser.add_argument('--batch_size', '-b', default=8, type=int)
    parser.add_argument('--features', '-f', default=384, type=int)
    parser.add_argument('--seed', '-r', default=421, type=int)
    parser.add_argument('--log_interval', '-q', default=5, type=int)
    parser.add_argument('--test_interval', '-t', default=1, type=int)
    parser.add_argument('--save', '-s', default='./', type=str)
    args = parser.parse_args()
    
    # python train.py -dtr dataset/bert_base_uncased_train_embeddings.npy -dte dataset/bert_base_uncased_train_embeddings.npy -e 200 -b 64 -f 768 -q 10000 -t 5 -s model/
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
