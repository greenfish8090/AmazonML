import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import math


class TextDataset(Dataset):
    def __init__(self, path="dataset/train.csv", test=False, seed=421):
        # Reading and preprocessing dataset
        self.test = test
        print("========> LOADING DATASET <========")
        self.data = pd.read_csv(path).reset_index()
        self.mean = 6.5502
        self.std = 0.9601

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.test:
            _, product_id, title, bullet_points, description, type_id, length = self.data.iloc[idx]
            string = f"Title: {title}, Bullet Points: {bullet_points}, Description: {description}"

            x = {"string": string, "type_id": type_id}

            length = (min(math.log(length), 12) - self.mean) / self.std
            return x, np.float32(length)

        _, product_id, title, bullet_points, description, type_id = self.data.iloc[idx]
        string = f"Title: {title}, Bullet Points: {bullet_points}, Description: {description}"

        x = {"string": string, "type_id": type_id}

        return x


class EEDataset(Dataset):
    def __init__(self, path, id_to_ind, default_ind, drop_big=False, test=False, seed=421):
        # Reading and preprocessing dataset
        self.test = test
        print("========> LOADING DATASET <========")
        self.data = pd.read_csv(path)
        if drop_big:
            self.data = self.data[self.data["PRODUCT_LENGTH"] < 1e4]
        self.data = self.data.reset_index()
        self.mean = 6.5502
        self.std = 0.9601
        vc = dict(self.data["PRODUCT_TYPE_ID"].value_counts())
        self.id_to_ind = id_to_ind
        self.default_ind = default_ind

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.test:
            _, product_id, title, bullet_points, description, type_id, length = self.data.iloc[idx]
            string = f"Title: {title}, Bullet Points: {bullet_points}, Description: {description}"

            # length = (min(math.log(length), 12) - self.mean) / self.std
            if type_id in self.id_to_ind:
                return self.id_to_ind[type_id], np.float32(length)
            return self.default_ind, np.float32(length)

        _, product_id, title, bullet_points, description, type_id = self.data.iloc[idx]
        string = f"Title: {title}, Bullet Points: {bullet_points}, Description: {description}"

        x = {"string": string, "type_id": type_id}

        return x


class TextEEDataset(Dataset):
    def __init__(self, path, id_to_ind, default_ind, transform=False, test=False, seed=421):
        # Reading and preprocessing dataset
        self.transform = transform
        self.test = test
        print("========> LOADING DATASET <========")
        self.data = pd.read_csv(path)
        self.data = self.data.reset_index()
        self.mean = 6.5502
        self.std = 0.9601
        vc = dict(self.data["PRODUCT_TYPE_ID"].value_counts())
        self.id_to_ind = id_to_ind
        self.default_ind = default_ind

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.test:
            _, product_id, title, bullet_points, description, type_id, length = self.data.iloc[idx]
            string = f"Title: {title}, Bullet Points: {bullet_points}, Description: {description}"

            if self.transform:
                length = (min(math.log(length), 12) - self.mean) / self.std
            if type_id in self.id_to_ind:
                return string, self.id_to_ind[type_id], np.float32(length)
            return string, self.default_ind, np.float32(length)

        _, product_id, title, bullet_points, description, type_id = self.data.iloc[idx]
        string = f"Title: {title}, Bullet Points: {bullet_points}, Description: {description}"

        if type_id in self.id_to_ind:
            return string, self.id_to_ind[type_id]
        return string, self.default_ind


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_path, csv_path, seed=421):
        # Reading and preprocessing dataset
        print("========> LOADING DATASET <========", flush=True)
        self.targets = pd.read_csv(csv_path, usecols=["PRODUCT_LENGTH"]).reset_index()
        self.embeddings = np.load(embeddings_path)
        self.mean = 6.5502
        self.std = 0.9601

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        _, length = self.targets.iloc[idx]
        length = (min(math.log(length), 12) - self.mean) / self.std
        embedding = self.embeddings[idx]

        return embedding, length
