from mmcv.utils import Config
import os
import pandas as pd

def pseudo_label(csv_original_train, csv_pseudo_train, csv_pseudo_test):
    df_train_original = pd.read_csv(csv_original_train)
    df_test_pseudo = pd.read_csv(csv_pseudo_test)

    df_train_pseudo = pd.DataFrame()
    df_train_pseudo['filename'] = pd.Series(list(df_train_original['filename'])+(list(df_test_pseudo['image_raw'])))
    df_train_pseudo['label'] = pd.Series(list(df_train_original['label'])+(list(df_test_pseudo['label_raw'])))
    df_train_pseudo.to_csv(csv_pseudo_train, index=False)
if __name__ == '__main__':
    cfg = Config.fromfile('config_data_utils_test.py')
    csv_pseudo_test = '/home/muyun99/data/dataset/competition_data/xunfei_face_reco/test_pseudo.csv'
    for fold_idx in range(cfg.num_KFold):
        csv_original_train = os.path.join(cfg.path_save_trainval_csv, f'train_fold{fold_idx}.csv')
        csv_pseudo_train = os.path.join(cfg.path_save_trainval_csv, f'train_fold{fold_idx}_pseudo.csv')
        pseudo_label(csv_original_train=csv_original_train, csv_pseudo_train=csv_pseudo_train, csv_pseudo_test=csv_pseudo_test)