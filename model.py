from torch import nn
import torch
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer


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


class Regressor2(nn.Module):
    def __init__(self, num_feature):
        super(Regressor2, self).__init__()

        self.layer_1 = nn.Sequential(nn.Linear(num_feature, 512))
        self.layer_2 = nn.Sequential(nn.Linear(512, 256))
        self.reg = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.reg(x)


class EntityEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_embeddings):
        super(EntityEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.regressor = Regressor(self.embedding_dim)

    def forward(self, x):
        e = self.embedding(x)
        return self.regressor(e)


class TransformerRegressor(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        if transformer == "bert-base-uncased":
            self.transformer = BertModel.from_pretrained(transformer)
            self.tokenizer = BertTokenizer.from_pretrained(transformer)
            self.num_feature = 768
        elif transformer == "roberta-base":
            self.transformer = RobertaModel.from_pretrained(transformer)
            self.tokenizer = RobertaTokenizer.from_pretrained(transformer)
            self.num_feature = 768

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


class TransformerEntityRegressorOld(nn.Module):
    def __init__(self, transformer, embedding_dim, num_embeddings):
        super().__init__()
        if transformer == "bert-base-uncased":
            self.transformer = BertModel.from_pretrained(transformer)
            self.tokenizer = BertTokenizer.from_pretrained(transformer)
            self.num_feature = 768
        elif transformer == "roberta-base":
            self.transformer = RobertaModel.from_pretrained(transformer)
            self.tokenizer = RobertaTokenizer.from_pretrained(transformer)
            self.num_feature = 768

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.regressor = Regressor(self.num_feature + self.embedding_dim)

    def forward(self, string, type_id, device):
        inp = self.tokenizer(
            string,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        inp = {k: v.to(device) for k, v in inp.items()}
        output = self.transformer(**inp)
        cls = output[0][:, 0, :]
        x = self.embedding(type_id.to(device))
        x = self.regressor(torch.cat([cls, x], dim=1))
        return x


class TransformerEntityRegressor(nn.Module):
    def __init__(self, transformer, embedding_dim, num_embeddings):
        super().__init__()
        if transformer == "bert-base-uncased":
            self.transformer = BertModel.from_pretrained(transformer)
            self.tokenizer = BertTokenizer.from_pretrained(transformer)
            self.num_feature = 768
        elif transformer == "roberta-base":
            self.transformer = RobertaModel.from_pretrained(transformer)
            self.tokenizer = RobertaTokenizer.from_pretrained(transformer)
            self.num_feature = 768

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.regressor = Regressor2(self.num_feature + self.embedding_dim)

    def forward(self, inp, type_id):
        output = self.transformer(**inp)
        cls = output[0][:, 0, :]
        x = self.embedding(type_id)
        x = self.regressor(torch.cat([cls, x], dim=1))
        return x
