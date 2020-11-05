import gc
import os
import time
import torch
import config
import pandas as pd
import torch.nn as nn
from model import Net
from ranger import Ranger
from utils import accuracy
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from data import MyDataSet, transform_train, transform_valid, label2int

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_data, val_data, fold_idx=None):
    train_data = MyDataSet(train_data, transform_train)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_data = MyDataSet(val_data, transform_valid)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = Net(config.model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Ranger(model.parameters(), lr=1e-3, weight_decay=0.0005)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_val_acc = 0
    best_val_loss = 0
    last_improved_epoch = 0
    if fold_idx is None:
        print('start')
        model_save_path = os.path.join(config.dir_weight, '{}.bin'.format(config.save_model_name))
    else:
        print('start fold: {}'.format(fold_idx + 1))
        model_save_path = os.path.join(config.dir_weight, '{}_fold{}.bin'.format(config.save_model_name, fold_idx))
    for cur_epoch in range(config.num_epochs):

        start_time = int(time.time())
        model.train()
        print('epoch: ', cur_epoch + 1)
        cur_step = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            probs = model(batch_x)

            train_loss = criterion(probs, batch_y)
            train_loss.backward()
            optimizer.step()

            cur_step += 1
            if cur_step % config.step_train_print == 0:
                train_acc = accuracy(probs, batch_y)
                msg = 'the current step: {0}/{1}, train loss: {2:>5.2}, train acc: {3:>6.2%}'
                print(msg.format(cur_step, len(train_loader), train_loss.item(), train_acc[0].item()))
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        if val_acc > best_val_acc or best_val_loss > val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            improved_str = '*'
            last_improved_epoch = cur_epoch
        else:
            improved_str = ''
        # msg = 'the current epoch: {0}/{1}, train loss: {2:>5.2}, train acc: {3:>6.2%},  ' \
        #       'val loss: {4:>5.2}, val acc: {5:>6.2%}, {6}'
        msg = 'the current epoch: {0}/{1}, val loss: {2:>5.2}, val acc: {3:>6.2%}, cost: {4}s {5}'
        end_time = int(time.time())
        print(msg.format(cur_epoch + 1, config.num_epochs, val_loss, val_acc,
                         end_time - start_time, improved_str))
        if cur_epoch - last_improved_epoch > config.num_patience_epoch:
            print("No optimization for a long time, auto-stopping...")
            break
        scheduler_cosine.step()
    del model
    gc.collect()
    return best_val_acc, best_val_loss


def evaluate(model, val_loader, criterion):
    model.eval()
    data_len = 0
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_len = len(batch_y)
            # batch_len = len(batch_y.size(0))
            data_len += batch_len
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            probs = model(batch_x)
            loss = criterion(probs, batch_y)
            total_loss += loss.item()
            _acc = accuracy(probs, batch_y)
            total_acc += _acc[0].item() * batch_len

    return total_loss / data_len, total_acc / data_len


if __name__ == "__main__":
    torch.manual_seed(config.seed_random)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    sum_val_acc = 0
    sum_val_loss = 0

    train_df = pd.read_csv(config.dir_csv_train)
    print(train_df['label'].value_counts())

    train_df['filename'] = train_df['filename'].apply(
        lambda x: os.path.join(config.dir_raw_train, x))
    train_df['label'] = train_df['label'].apply(
        lambda x: label2int(x)
    )

    if config.num_KFold == 1:
        train_data, val_data = train_test_split(
            train_df, shuffle=True, test_size=config.size_valid)
        print('train:{}, val:{}'.format(
            train_data.shape[0], val_data.shape[0]))
        best_val_acc, best_val_loss = train(train_data, val_data)
        print("best_val_acc:" + str(best_val_acc))
        print("best_val_loss:" + str(best_val_loss))
    else:
        x = train_df['filename'].values
        y = train_df['label'].values
        skf = StratifiedKFold(
            n_splits=config.num_KFold,
            random_state=config.seed_random,
            shuffle=True
        )
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y)):
            best_val_acc, best_val_loss = train(train_df.iloc[train_idx], train_df.iloc[val_idx], fold_idx)
            sum_val_acc += best_val_acc
            sum_val_loss += best_val_loss
        print("avg_val_acc: {:.4f}".format(sum_val_acc / config.num_KFold))
        print("avg_val_loss: {:.4f}".format(sum_val_loss / config.num_KFold))
