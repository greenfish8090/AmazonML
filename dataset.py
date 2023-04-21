import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, path="dataset/train.csv", seed=421):
        # Reading and preprocessing dataset
        print("========> LOADING DATASET <========")
        self.data = pd.read_csv(path).reset_index()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _, product_id, title, bullet_points, description, type_id, length = self.data.iloc[idx]
        string = f"Title: {title}, Bullet Points: {bullet_points}, Description: {description}"

        sample = {
            "string": string,
            "type_id": type_id,
            "length": length,
        }

        return sample
