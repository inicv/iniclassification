import os
import torch
import config
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform_train = transforms.Compose([
    transforms.Resize(
        [int(config.size_train_image * config.factor_train), int(config.size_train_image * config.factor_train)]),
    transforms.RandomRotation(15),
    transforms.RandomChoice([
        transforms.Resize([config.size_train_image, config.size_train_image]),
        transforms.CenterCrop([config.size_train_image, config.size_train_image])
    ]),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3),
    normalize,
])

transform_valid = transforms.Compose([
    transforms.Resize([config.size_valid_image, config.size_valid_image]),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.Resize([config.size_test_image, config.size_test_image]),
    transforms.ToTensor(),
    normalize,
])


def label2int(label):
    dict_label = {
        'mountain': 0,
        'street': 1,
        'buildings': 2,
        'sea': 3,
        'forest': 4,
        'glacier': 5
    }
    return dict_label[label]

def int2label(label):
    dict_label = {
        0: 'mountain',
        1: 'street',
        2: 'buildings',
        3: 'sea',
        4: 'forest',
        5: 'glacier'
    }
    return dict_label[label]

class MyDataSet(Dataset):
    def __init__(self, df, transform, mode='train'):
        self.df = df
        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        if (self.mode == 'train'):
            img = Image.open(self.df['filename'].iloc[index]).convert('RGB')
            img = self.transform(img)
            return img, torch.from_numpy(np.array(self.df['label'].iloc[index]))
        else:
            img = Image.open(self.df[index]).convert('RGB')
            img = self.transform(img)
            return img, torch.from_numpy(np.array(0))

    def __len__(self):
        return len(self.df)


def get_trainval_dataloader():
    train_df = pd.read_csv(config.dir_csv_train)
    print(train_df['label'].value_counts())
    print(set(train_df['label']))
    train_df['filename'] = train_df['filename'].apply(
        lambda x: os.path.join(config.dir_raw_train, x)
    )
    train_df['label'] = train_df['label'].apply(
        lambda x: label2int(x)
    )

    train_df, valid_df = train_test_split(
        train_df, shuffle=True, test_size=0.1)
    train_data = MyDataSet(train_df, transform_train)
    valid_data = MyDataSet(valid_df, transform_valid)

    train_loader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(
        valid_data, batch_size=config.batch_size, shuffle=False)
    return train_loader, valid_loader


def get_test_dataloader():
    data_len = len(os.listdir(config.dir_raw_test))
    test_path_list = [
        os.path.join(config.dir_raw_test, f'{x}.jpg') for x in range(0, data_len)]
    test_data = np.array(test_path_list)
    test_dataset = MyDataSet(test_data, transform_test, 'test')
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False)
    return test_loader
