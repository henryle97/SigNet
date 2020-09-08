import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def get_data_loader(data_dir, is_train, batch_size, image_transform, num_workers):

    data = SigDataset(is_train, data_dir, image_transform)
    is_shuffle = is_train
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_workers, pin_memory=True)

    return data_loader


class SigDataset(Dataset):
    def __init__(self, is_train: bool, data_dir: str, image_transform=None):
        if not os.path.exists(os.path.join(data_dir, 'train.csv')) or not os.path.exists(
                os.path.join(data_dir, 'test.csv')):
            print('Not found train/test splits, run create_annotation first')
        else:
            print('Use existed train/test splits')

        if is_train:
            self.df = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None)
        else:
            self.df = pd.read_csv(os.path.join(data_dir, 'test.csv'), header=None)

        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x1, x2, y = self.df.iloc[index]

        x1 = Image.open(x1).convert('L')
        x2 = Image.open(x2).convert('L')

        if self.image_transform:
            x1 = self.image_transform(x1)
            x2 = self.image_transform(x2)

        return x1, x2, y
