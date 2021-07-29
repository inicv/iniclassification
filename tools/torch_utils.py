import cv2
import math
import torch
import random
import numpy as np
import os
import mmcv
import time
import pandas as pd

import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR

def set_seed(cfg):
    seed = cfg.random_seed
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_cudnn(cfg):
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def set_work_dir(cfg):
    if cfg.get('work_dir', False) is None:
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(cfg.config))[0]+f'_tag_{cfg.tag}')

def get_test_submission_path(cfg, tag):
    work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(cfg.config))[0]+f'_tag_{tag}')
    test_submission_path = os.path.join(work_dir, f'test_submission_{tag}.csv')
    return test_submission_path

def get_valid_submission_path(cfg, tag):
    work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(cfg.config))[0]+f'_tag_{tag}')
    valid_submission_path = os.path.join(work_dir, f'test_submission_{tag}.csv')
    return valid_submission_path

def calculate_acc_from_two_list(list1, list2):
    assert len(list1) == len(list2)
    T = 0
    for idx, (item1, item2) in enumerate(zip(list1, list2)):
        if item1 == item2:
            T += 1
    return T/len(list1)

def make_log_dir(cfg):
    log_dir = os.path.join(cfg.work_dir, 'logs')
    data_dir = os.path.join(cfg.work_dir, 'data')
    model_dir = os.path.join(cfg.work_dir, 'models')
    tensorboard_dir = os.path.join(cfg.work_dir, 'tensorboard')
    mmcv.mkdir_or_exist(log_dir)
    mmcv.mkdir_or_exist(data_dir)
    mmcv.mkdir_or_exist(model_dir)
    mmcv.mkdir_or_exist(tensorboard_dir)

    log_path = os.path.join(log_dir, 'run.txt')
    data_path = os.path.join(data_dir, 'data.json')
    model_path = os.path.join(model_dir, 'best.pth')
    valid_submission_path = os.path.join(cfg.work_dir, f'valid_submission_{cfg.tag}.csv')
    test_submission_path = os.path.join(cfg.work_dir, f'test_submission_{cfg.tag}.csv')

    cfg.model_dir = model_dir
    cfg.tensorboard_dir = tensorboard_dir
    cfg.log_path = log_path
    cfg.data_path = data_path
    cfg.model_path = model_path
    cfg.valid_submission_path = valid_submission_path
    cfg.test_submission_path = test_submission_path

def save_config(cfg):
    if isinstance(cfg, mmcv.Config):
        config_name = os.path.splitext(os.path.basename(cfg.config))[0]
        cfg.dump(os.path.join(cfg.work_dir, f'{config_name}.py'))

def print_log(message, cfg):
    print(message)
    if isinstance(message, str):
        with open(cfg.log_path, 'a+') as f:
            f.write(message + '\n')

def calculate_parameters(model):
    return sum(param.numel() for param in model.parameters())/1000000.0

def load_model(model, model_path, parallel=False):
    if parallel:
        model.module.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path))

def save_model(model, model_path, parallel=False):
    if parallel:
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)

def set_gpu(cfg):
    try:
        cfg.use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        cfg.use_gpu = '0'

    the_number_of_gpu = len(cfg.use_gpu.split(','))
    cfg.parallel = the_number_of_gpu > 1

    print_log(f'[i] using gpu : {cfg.use_gpu}', cfg)
    print_log(f'[i] the number of gpu : {the_number_of_gpu}', cfg)

def get_learning_rate_from_optimizer(optimizer):
    return optimizer.param_groups[0]['lr']

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

def average_precision(pred, target):
    """Calculate the average precision for a single class.
    AP summarizes a precision-recall curve as the weighted mean of maximum
    precisions obtained for any r'>r, where r is the recall:
    ..math::
        \\text{AP} = \\sum_n (R_n - R_{n-1}) P_n
    Note that no approximation is involved since the curve is piecewise
    constant.
    Args:
        pred (np.ndarray): The models prediction with shape (N, ).
        target (np.ndarray): The target of each prediction with shape (N, ).
    Returns:
        float: a single float as average precision value.
    """
    eps = np.finfo(np.float32).eps

    # sort examples
    sort_inds = np.argsort(-pred)
    sort_target = target[sort_inds]

    # count true positive examples
    pos_inds = sort_target == 1
    tp = np.cumsum(pos_inds)
    total_pos = tp[-1]

    # count not difficult examples
    pn_inds = sort_target != -1
    pn = np.cumsum(pn_inds)

    tp[np.logical_not(pos_inds)] = 0
    precision = tp / np.maximum(pn, eps)
    ap = np.sum(precision) / np.maximum(total_pos, eps)
    return ap

def multi_label_mAP(pred, target):
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')

    assert pred.shape == \
           target.shape, 'pred and target should be in the same shape.'
    num_classes = pred.shape[1]
    ap = np.zeros(num_classes)
    for k in range(num_classes):
        ap[k] = average_precision(pred[:, k], target[:, k])
    mean_ap = ap.mean() * 100.0
    return mean_ap

class Timer:
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0

        self.tik()

    def tik(self):
        self.start_time = time.time()

    def tok(self, ms=False, clear=False):
        self.end_time = time.time()

        if ms:
            duration = int((self.end_time - self.start_time) * 1000)
        else:
            duration = int(self.end_time - self.start_time)

        if clear:
            self.tik()

        return duration


class Iterator:
    def __init__(self, loader):
        self.loader = loader
        self.init()

    def init(self):
        self.iterator = iter(self.loader)

    def get(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.init()
            data = next(self.iterator)

        return data


class Average_Meter:
    def __init__(self, keys):
        self.keys = keys
        self.clear()

    def add(self, dic):
        for key, value in dic.items():
            self.data_dic[key].append(value)

    def get(self, keys=None, clear=False):
        if keys is None:
            keys = self.keys

        dataset = [float(np.mean(self.data_dic[key])) for key in keys]
        if clear:
            self.clear()

        if len(dataset) == 1:
            dataset = dataset[0]

        return dataset

    def clear(self):
        self.data_dic = {key: [] for key in self.keys}

if __name__ == '__main__':

    valid_gt = pd.read_csv('/home/muyun99/data/dataset/competition_data/kaggle_classify_leaves/dataset/valid_fold0.csv')
    valid_pred = pd.read_csv('/home/muyun99/MyGithub/iniclassification/valid_submission.csv')
    list1 = list(valid_gt['label'])
    list2 = list(valid_pred['label'])
    acc = calculate_acc_from_two_list(list1, list2)
    print(acc)
