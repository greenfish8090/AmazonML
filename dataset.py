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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.test:
            _, product_id, title, bullet_points, description, type_id, length = self.data.iloc[idx]
            string = f"Title: {title}, Bullet Points: {bullet_points}, Description: {description}"

            x = {"string": string, "type_id": type_id}

            return x, np.float32(length)

        _, product_id, title, bullet_points, description, type_id = self.data.iloc[idx]
        string = f"Title: {title}, Bullet Points: {bullet_points}, Description: {description}"

        x = {"string": string, "type_id": type_id}

        return x


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
        length = (min(math.log(length), 12) - self.mean ) / self.std
        embedding = self.embeddings[idx]

        return embedding, length
