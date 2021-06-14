import argparse
from mmcv import Config, DictAction
from torch_utils import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from inicls import build_model, build_optimizer, build_loss, build_scheduler, build_trainval_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--device', help='device used for training')
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    args = parser.parse_args()

    return args


def evaluate(model, valid_dataloader):
    model.eval()
    eval_timer.tik()

    data_len = 0
    total_acc = 0

    with torch.no_grad():
        for step, (images, labels) in enumerate(valid_dataloader):
            data_len += len(labels)
            images, labels = images.cuda(), labels.cuda()

            logits = model(images)
            _acc = accuracy(logits, labels)
            total_acc += _acc[0].item() * len(labels)

    model.train()
    return total_acc / data_len


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.config = args.config

    if args.options is not None:
        cfg.merge_from_dict(args.options)

    set_seed(cfg)
    set_cudnn(cfg)
    set_work_dir(cfg)
    make_log_dir(cfg)
    save_config(cfg)
    set_gpu(cfg)
    log_func = lambda string='': print_log(string, cfg)

    ###################################################################################
    # Dataset, DataLoader
    ###################################################################################
    train_dataset, valid_dataset = build_trainval_dataset(cfg)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True,
                                  drop_last=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, num_workers=1, pin_memory=True)

    ###################################################################################
    # Network
    ###################################################################################
    model = build_model(cfg)
    model = model.cuda()
    model.train()
    log_func('[i] Architecture is {}'.format(cfg.model))
    log_func('[i] Total Params: %.2fM' % (calculate_parameters(model)))
    log_func()

    if cfg.parallel:
        model = nn.DataParallel(model)
    load_model_func = lambda: load_model(model, cfg.model_path, parallel=cfg.parallel)
    save_model_func = lambda: save_model(model, cfg.model_path, parallel=cfg.parallel)

    ###################################################################################
    # Loss, Optimizer, LR_scheduler
    ###################################################################################
    criterion = build_loss(cfg)
    optimizer = build_optimizer(cfg=cfg, model=model)
    scheduler = build_scheduler(cfg=cfg, optimizer=optimizer)

    ###################################################################################
    # Train
    ###################################################################################
    train_timer = Timer()
    eval_timer = Timer()
    best_accuracy = -1
    val_iteration = len(train_dataloader)
    log_iteration = val_iteration
    max_iteration = cfg.max_epoch * val_iteration
    log_func(f'[i] val_iteration : {val_iteration}')
    log_func(f'[i] max_iteration : {max_iteration}')

    writer = SummaryWriter(cfg.tensorboard_dir)
    train_iterator = Iterator(train_dataloader)
    train_meter = Average_Meter(['loss'])

    for iteration in range(max_iteration):
        images, labels = train_iterator.get()
        images, labels = images.cuda(), labels.cuda()

        logits = model(images)
        loss = criterion(logits, labels).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.add({
            'loss': loss.item(),
        })

        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            time = train_timer.tok(clear=True)

            log_func(f'[i] iteration={iteration + 1}, \
                learning_rate={learning_rate}, \
                loss={loss}, \
                time={time} sec')

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)

        #################################################################################################
        # Evaluation
        #################################################################################################
        if (iteration + 1) % val_iteration == 0:
            acc = evaluate(model=model, valid_dataloader=valid_dataloader)
            time = eval_timer.tok(clear=True)
            if best_accuracy < acc:
                best_accuracy = acc
                save_model_func()
                log_func('[i] save model')

            log_func(f'[i] iteration={iteration + 1}, \
                train_ACC={acc}%, \
                best_train_ACC={best_accuracy}%, \
                time={time} sec')

            writer.add_scalar('Evaluation/train_ACC', acc, iteration)
            writer.add_scalar('Evaluation/best_train_ACC', best_accuracy, iteration)

        #################################################################################################
        # For Step()
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            scheduler.step()
    writer.close()
