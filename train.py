import torch
from torch import nn
from model import Regressor
from transformers import AutoTokenizer, AutoModel
from dataset import EmbeddingDataset
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import argparse


class Trainer(nn.Module):
    def __init__(self, args, pred=False):
        super(Trainer, self).__init__()
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        self.get_model(self.args.features) # Creating Regressor
        if not pred:
            self.get_data(self.args.seed, self.args.batch_size) # Creating Dataloaders for Train and Val
            self.get_training_utils() # Creating Optimizer

        # Variables to track training
        self.loss = None
        self.mape = None 
        self.best_mape = None
        self.last_mape = None
        
    # Creating regressor
    def get_model(self, features=None):
        self.regressor = Regressor(features).to(self.device) 

    # Creating Dataloaders and Datasets
    def get_data(self, seed, batch_size):
        self.train_set = EmbeddingDataset(csv_path=self.args.train_data_path)
        self.val_set = EmbeddingDataset(csv_path=self.args.val_data_path)
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False)

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
        for batch_idx, (emb, trg) in enumerate(self.train_loader):
            emb = emb.to(self.device)
            trg = trg.to(self.device)

            output = self(emb)
            loss = loss_fn(output, trg)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(emb), len(self.trainloader.dataset),
                    100. * batch_idx / len(self.trainloader), loss.item()))
        return loss

    # Testing loop for one epoch using complete val_set
    def val(self, epoch):
        self.regressor.eval()
        mape = 0
        for _, (emb, val) in enumerate(self.val_loader):
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
            if epoch % self.args.val_interval == 0:
                mape = self.val(epoch)
                self.mape.append(mape)
                if mape < self.best_mape:
                    self.save('best')
                    self.best_mape = mape
            
        self.last_mape = self.val('INTMAX')
        self.save('last')

    # Saving Model State
    def save(self, star):
        save_dict = {
            'model':self.regressor.state_dict(), \
            'loss':self.loss, \
            'mape':self.mape, \
            'best_mape': self.best_mape, \
            'last_mape':self.last_mape, 
            }
        torch.save(save_dict, "{}_{}.pt".format(self.args.save, star))
        print("{} model saved".format(star))
    
    # Loading Model State
    def load(self, path):
        self.regressor.load_state_dict(torch.load(path, map_location=torch.device(self.device))['model'])

def main():
    parser = argparse.ArgumentParser(description='Null')
    parser.add_argument('--train_data_path', '-d', default='', type=str)
    parser.add_argument('--val_data_path', '-d', default='', type=str)
    parser.add_argument('--epochs', '-e', default=100, type=int)
    parser.add_argument('--lr', '-l', default=1e-3, type=float)
    parser.add_argument('--batch_size', '-b', default=8, type=int)
    parser.add_argument('--features', '-f', default=768, type=int)
    parser.add_argument('--seed', '-r', default=421, type=int)
    parser.add_argument('--log_interval', '-q', default=5, type=int)
    parser.add_argument('--val_interval', '-t', default=1, type=int)
    parser.add_argument('--save', '-s', default='./', type=str)
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()