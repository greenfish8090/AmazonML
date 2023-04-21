from torch import nn
from transformers import BertModel, BertTokenizer


class Regressor(nn.Module):
    def __init__(self, num_feature):
        super(Regressor, self).__init__()

        self.layer_1 = nn.Sequential(nn.Linear(num_feature, 512), nn.Linear(512, 512))
        self.layer_2 = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 256))
        self.layer_3 = nn.Sequential(nn.Linear(256, 128), nn.Linear(128, 128))
        self.layer_4 = nn.Sequential(nn.Linear(128, 64), nn.Linear(64, 32))

        self.reg = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(254)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.batchnorm4 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.dropout(x)

        return self.reg(x)


class TransformerRegressor(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        if transformer == "bert-base-uncased":
            self.transformer = BertModel.from_pretrained(transformer)
            self.tokenizer = BertTokenizer.from_pretrained(transformer)
            self.num_feature = 768
        else:
            raise NotImplementedError()

        self.regressor = Regressor(self.num_feature)

    def forward(self, x, device):
        inp = self.tokenizer(
            x["string"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        inp = {k: v.to(device) for k, v in inp.items()}
        output = self.transformer(**inp)
        cls = output[0][:, 0, :]
        x = self.regressor(cls)
        return x
