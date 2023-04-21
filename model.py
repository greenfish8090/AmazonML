from torch import nn
from transformers import BertModel, BertTokenizer


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
