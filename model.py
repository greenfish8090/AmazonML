from torch import nn

class Regressor(nn.Module):
    def __init__(self, num_feature):
        super(Regressor, self).__init__()
        
        self.layer_1 = nn.Sequential(nn.Linear(num_feature, 256))
        self.reg = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.batchnorm = nn.BatchNorm1d(256)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return self.reg(x)
