import gc
import os
import time
import torch
import logging
from inicls import config, utils
import pandas as pd
import torch.nn as nn
from inicls.model import Net
from ranger import Ranger
from inicls.optimizer import get_optimizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from inicls.data import MyDataSet, transform_train, transform_valid, label2int
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_print(msg):
    logging.info(msg)
    print(msg)


def train(train_data, val_data, fold_idx=None):
    writer = SummaryWriter(comment="_{}".format(args.name))
    logging.basicConfig(filename=f'logs/{args.name}.log', level=logging.INFO, format='%(levelname)s: %(message)s')

    train_data = MyDataSet(train_data, transform_train)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_data = MyDataSet(val_data, transform_valid)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    if args.lr != 1:
        lr = args.lr
    else:
        lr = config.lr

    if args.network != False:
        model = Net(args.network).to(device)
    else:
        model = Net(config.network).to(device)
    if args.optimizer != False:
        optimizer = get_optimizer(name_optimizer=args.optimizer, model=model, lr=lr)
    else:
        optimizer = get_optimizer(name_optimizer=config.optimizer, model=model, lr=lr)

    if args.scheduler != False:
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    else:
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    if args.loss != False:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    logging.info(
        f'''Starting training:
                Epochs:          {config.num_epochs}
                Batch size:      {config.batch_size}
                Learning rate:   {lr}
                Training size:   {len(train_data)}
                Validation size: {len(val_data)}
                Device:          {device}
        '''
    )
    global_step = 0
    best_val_acc = 0
    best_val_loss = 0
    last_improved_epoch = 0

    if fold_idx is None:
        log_print('start')
        model_save_path = os.path.join(config.dir_weight, '{}.bin'.format(config.save_model_name))
    else:
        log_print('start fold: {}'.format(fold_idx + 1))
        model_save_path = os.path.join(config.dir_weight, '{}_fold{}.bin'.format(config.save_model_name, fold_idx))
    for cur_epoch in range(config.num_epochs):

        start_time = int(time.time())
        model.train()
        log_print(f'epoch: {cur_epoch + 1}', )
        cur_step = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            probs = model(batch_x)

            train_loss = criterion(probs, batch_y)
            writer.add_scalar('Loss/train', train_loss.item(), global_step=global_step)

            train_loss.backward()
            optimizer.step()

            cur_step += 1
            if cur_step % config.step_train_print == 0:
                train_acc = utils.accuracy(probs, batch_y)
                msg = 'the current step: {0}/{1}, train loss: {2:>5.2}, train acc: {3:>6.2%}'
                log_print(msg.format(cur_step, len(train_loader), train_loss.item(), train_acc[0].item()))
            global_step += 1
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        writer.add_scalar('Acc/valid', val_acc, global_step=global_step)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            improved_str = '*'
            last_improved_epoch = cur_epoch
        else:
            improved_str = ''
        msg = 'the current epoch: {0}/{1}, val loss: {2:>5.2}, val acc: {3:>6.2%}, cost: {4}s {5}'
        end_time = int(time.time())
        log_print(msg.format(cur_epoch + 1, config.num_epochs, val_loss, val_acc,
                             end_time - start_time, improved_str))
        if cur_epoch - last_improved_epoch > config.num_patience_epoch:
            log_print("No optimization for a long time, auto-stopping...")
            break
        scheduler_cosine.step()
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], global_step=global_step)
    del model
    gc.collect()
    log_print(f'Best valid_acc is {best_val_acc}, lowest valid_loss is {best_val_loss}')
    writer.add_scalar('Best_Acc/valid', best_val_acc, global_step=global_step)
    writer.close()
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
            _acc = utils.accuracy(probs, batch_y)
            total_acc += _acc[0].item() * batch_len

    return total_loss / data_len, total_acc / data_len


if __name__ == "__main__":

    args = config.get_args()
    utils.torch_config()

    sum_val_acc = 0
    sum_val_loss = 0

    # read csv_train
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
