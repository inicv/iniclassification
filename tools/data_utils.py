from sklearn.model_selection import train_test_split, StratifiedKFold
from mmcv.utils import Config
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyse_class_num():
    raw_train_df = pd.read_csv(cfg.path_raw_train_csv)
    raw_train_df.columns = ['filenames', 'label']

    train_data, val_data = train_test_split(
        raw_train_df, shuffle=True, test_size=cfg.size_valid)
    print('train:{}, val:{}'.format(
        train_data.shape[0], val_data.shape[0]))

    print(raw_train_df['label'].value_counts())

    data = raw_train_df['label'].value_counts().values
    plt.hist(data, bins=30, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("区间")
    plt.ylabel("频数/频率")
    plt.title("频数/频率分布直方图")
    plt.show()

def generate_classmap():
    raw_train_df = pd.read_csv(cfg.path_raw_train_csv)

    print(raw_train_df['label'].value_counts())
    class_names = raw_train_df['label'].value_counts().index
    with open(cfg.path_save_classmap_txt, 'w') as f:
        for name in class_names:
            f.write(name + '\n')


def split_train_val():
    raw_train_df = pd.read_csv(cfg.path_raw_train_csv)
    label_index_map = {}
    with open(cfg.path_save_classmap_txt, 'r') as f:
        classnames = f.readlines()
        for idx, name in enumerate(classnames):
            name = name.strip()
            label_index_map[name] = idx


    raw_train_df.columns = ['filename', 'label']

    raw_train_df['filename'] = raw_train_df['filename'].apply(
        lambda item: os.path.join(cfg.path_train_img, item.split('/')[-1])
    )
    raw_train_df['label'] = raw_train_df['label'].apply(
        lambda item: label_index_map[item]
    )

    x = raw_train_df['filename'].values
    y = raw_train_df['label'].values
    if cfg.num_KFold > 1:
        skf = StratifiedKFold(
            n_splits=cfg.num_KFold,
            random_state=cfg.seed_random,
            shuffle=True
        )
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y)):
            fold_train = raw_train_df.iloc[train_idx]
            fold_valid = raw_train_df.iloc[val_idx]
            fold_train.to_csv(os.path.join(cfg.path_save_trainval_csv, f'train_fold{fold_idx}.csv'))
            fold_valid.to_csv(os.path.join(cfg.path_save_trainval_csv, f'valid_fold{fold_idx}.csv'))
            print(f'train_fold{fold_idx}: {fold_train.shape[0]}, valid_fold{fold_idx}: {fold_valid.shape[0]}')
    else:
        train_data, valid_data = train_test_split(
            raw_train_df, shuffle=True, test_size=cfg.size_valid, random_state=cfg.seed_random)
        train_data.to_csv(os.path.join(cfg.path_save_trainval_csv, f'train.csv'))
        valid_data.to_csv(os.path.join(cfg.path_save_trainval_csv, f'valid.csv'))
        print(f'train:{train_data.shape[0]}, valid:{valid_data.shape[0]}')

def generate_test_csv():
    raw_test_df = pd.read_csv(cfg.path_raw_test_csv)
    raw_test_df.columns = ['filename']

    raw_test_df['filename'] = raw_test_df['filename'].apply(
        lambda item: os.path.join(cfg.path_train_img, item.split('/')[-1])
    )
    raw_test_df.to_csv(os.path.join(cfg.path_save_test_csv, f'test.csv'))

if __name__ == '__main__':
    cfg = Config.fromfile('config_data_utils_test.py')
    # analyse_class_num()
    # generate_classmap()
    # split_train_val()
    generate_test_csv()