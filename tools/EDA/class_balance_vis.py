import argparse
import hashlib
import os

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pandas as pd
from albumentations import (
    HorizontalFlip, VerticalFlip, ShiftScaleRotate, CLAHE,
    Transpose, Blur, GridDistortion, HueSaturationValue, GaussNoise, CoarseDropout
)
from mmcv import Config


def get_hash(image):
    md5 = hashlib.md5()
    md5.update(np.array(image).tobytes())
    return md5.hexdigest()


def get_meta_data(df):
    df['hash'] = 0
    for i in range(len(df)):
        image = mmcv.imread(df['filename'][i])
        image_hash = get_hash(image)
        df.loc['hash', i] = image_hash
    return df


def get_duplicate(dt):
    dt['dup_number'] = 0
    dup = dt.groupby(by='hash')[['dup_number']].count().reset_index()
    dup.reset_index(drop=True, inplace=True)
    dup = dup.merge(dt[['filename', 'hash']], on='hash')
    return dup


def plot_aug(img_src, method):
    img = mmcv.imread(img_src, channel_order='rgb')
    augmented_img = method(p=1)(image=img)['image']
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    ax[0].imshow(img)
    ax[0].set_title('Original Image', fontsize=24)
    ax[1].imshow(augmented_img)
    ax[1].set_title(method.__name__, fontsize=24)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a models')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    df_train = pd.read_csv(os.path.join(cfg.data.train.data_prefix, 'raw_train.csv'))
    df_test = pd.read_csv(os.path.join(cfg.data.val.data_prefix, 'test.csv'))

    print(df_train.sample(10))
    print(df_test.sample(10))
    print(df_train.shape)
    print(df_test.shape)
    print(df_train.label.value_counts())

    # display a single image
    mmcv.imshow(mmcv.imread(df_train.sample(1)['filename'].values[0]))

    # show the class distribution
    data = df_train['label'].value_counts().values
    plt.hist(data, bins=30, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("区间")
    plt.ylabel("频数/频率")
    plt.title("频数/频率分布直方图")
    plt.show()

    # duplication in data
    # meta_train = get_meta_data(df_train)
    # meta_train.to_csv('hash.csv')
    # dup_train = get_duplicate(meta_train)
    # print(dup_train.head())

    # augmentation visulization
    # plot_aug('/home/muyun99/data/dataset/competition_data/kaggle_classify_leaves/classify-leaves/images/14.jpg', HorizontalFlip)
    # plot_aug('/home/muyun99/data/dataset/competition_data/kaggle_classify_leaves/classify-leaves/images/14.jpg', VerticalFlip)
    plot_aug('/home/muyun99/data/dataset/competition_data/kaggle_classify_leaves/classify-leaves/images/14.jpg', Transpose)
    plot_aug('/home/muyun99/data/dataset/competition_data/kaggle_classify_leaves/classify-leaves/images/14.jpg', ShiftScaleRotate)
    plot_aug('/home/muyun99/data/dataset/competition_data/kaggle_classify_leaves/classify-leaves/images/14.jpg', Blur)
    plot_aug('/home/muyun99/data/dataset/competition_data/kaggle_classify_leaves/classify-leaves/images/14.jpg', GaussNoise)
    # plot_aug('/home/muyun99/data/dataset/competition_data/kaggle_classify_leaves/classify-leaves/images/14.jpg', HueSaturationValue)
    # plot_aug('/home/muyun99/data/dataset/competition_data/kaggle_classify_leaves/classify-leaves/images/14.jpg', CoarseDropout)
    # plot_aug('/home/muyun99/data/dataset/competition_data/kaggle_classify_leaves/classify-leaves/images/14.jpg', GridDistortion)
    # plot_aug('/home/muyun99/data/dataset/competition_data/kaggle_classify_leaves/classify-leaves/images/14.jpg', CLAHE)
