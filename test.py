import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import ttach as tta
import argparse

from mmcv import Config, DictAction
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from inicls import build_model, build_optimizer, build_loss, build_scheduler, build_dataset, build_dataloader
from train import parse_args
from tools.torch_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(model):
    pred_list = []
    with torch.no_grad():
        for batch_x, _ in tqdm(test_dataloader):
            batch_x = batch_x.to(device)
            probs = model(batch_x)
            pred_list.extend(probs.cpu().numpy())
    return pred_list


def single_model_predict():
    pred_list = []
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            images = data['img']
            images = images.to(device)

            logits = model(images)
            logits = torch.max(torch.softmax(logits, dim=1), dim=1)
            logits = logits[1].cpu().numpy()
            pred_list += logits.tolist()
    return pred_list


def single_model_predict_tta():
    transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        # tta.Rotate90(angles=[0, 90, 180, 270]),
        # tta.Multiply(factors=[0.9, 1, 1.1]),
        tta.FiveCrops(224, 224)
    ])

    tta_model = tta.ClassificationTTAWrapper(model, transforms)

    data_len = 0
    total_acc = 0

    with torch.no_grad():
        for data in tqdm(valid_dataloader):
            images, labels = data['img'], data['gt_label']
            images, labels = images.cuda(), labels.cuda()
            data_len += len(labels)

            logits = tta_model(images)
            _acc = accuracy(logits, labels)
            total_acc += _acc[0].item() * len(labels)
    print(f'validation acc is {total_acc / data_len}')

    pred_list = []
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            images = data['img']
            images = images.to(device)

            probs = tta_model(images)
            probs = torch.max(torch.softmax(probs, dim=1), dim=1)
            probs = probs[1].cpu().numpy()
            pred_list += probs.tolist()

    return pred_list

# def multi_model_predict():
#     preds_dict = dict()
#     for model_name in model_name_list:
#         for fold_idx in range(5):
#             model = Net(model_name).to(device)
#             model_save_path = os.path.join(
#                 config.dir_weight, '{}_fold{}.bin'.format(model_name, fold_idx))
#             model.load_state_dict(torch.load(model_save_path))
#             pred_list = predict(model)
#             submission = pd.DataFrame(pred_list)
#             # submission = pd.DataFrame({"id": range(len(pred_list)), "label": pred_list})
#             submission.to_csv(
#                 '{}/{}_fold{}_submission.csv'.format(config.dir_submission, config.save_model_name, fold_idx),
#                 index=False,
#                 header=False)
#             preds_dict['{}_{}'.format(model_name, fold_idx)] = pred_list
#     pred_list = get_pred_list(preds_dict)
#     submission = pd.DataFrame(
#         {"id": range(len(pred_list)), "label": [int2label(x) for x in pred_list]})
#     submission.to_csv(config.dir_csv_test, index=False, header=False)
#
#
# def multi_model_predict_tta():
#     preds_dict = dict()
#     for model_name in model_name_list:
#         for fold_idx in range(5):
#             model = Net(model_name).to(device)
#             model_save_path = os.path.join(
#                 config.dir_weight, '{}_fold{}.bin'.format(model_name, fold_idx))
#             model.load_state_dict(torch.load(model_save_path))
#             '/home/muyun99/data/dataset/AIyanxishe/Image_Classification/weight/resnet18_train_size_256_fold0.bin'
#             transforms = tta.Compose([
#                 tta.Resize([int(config.size_test_image), int(config.size_test_image)]),
#                 tta.HorizontalFlip(),
#                 # tta.Rotate90(angles=[0, 180]),
#                 # tta.Scale(scales=[1, 2, 4]),
#                 # tta.Multiply(factors=[0.9, 1, 1.1]),
#                 tta.FiveCrops(config.size_test_image, config.size_test_image)
#             ])
#             tta_model = tta.ClassificationTTAWrapper(model, transforms)
#
#             pred_list = predict(tta_model)
#             submission = pd.DataFrame(pred_list)
#             submission.to_csv(
#                 '{}/{}_fold{}_submission.csv'.format(config.dir_submission, config.save_model_name, fold_idx),
#                 index=False,
#                 header=False
#             )
#             preds_dict['{}_{}'.format(model_name, fold_idx)] = pred_list
#
#     pred_list = get_pred_list(preds_dict)
#     submission = pd.DataFrame(
#         {"id": range(len(pred_list)), "label": [int2label(x) for x in pred_list]})
#     submission.to_csv(config.dir_csv_test, index=False, header=False)
#
#
# def file2submission():
#     preds_dict = dict()
#     for model_name in model_name_list:
#         for fold_idx in range(5):
#             df = pd.read_csv('{}/{}_fold{}_submission.csv'
#                              .format(config.dir_submission, model_name, fold_idx), header=None)
#             preds_dict['{}_{}'.format(model_name, fold_idx)] = df.values
#     pred_list = get_pred_list(preds_dict)
#     submission = pd.DataFrame(
#         {"id": range(len(pred_list)), "label": [int2label(x) for x in pred_list]})
#     submission.to_csv(config.dir_csv_test, index=False, header=False)
#
#
# def get_pred_list(preds_dict):
#     pred_list = []
#     if predict_mode == 1:
#         for i in range(data_len):
#             preds = []
#             for model_name in model_name_list:
#                 for fold_idx in range(5):
#                     prob = preds_dict['{}_{}'.format(model_name, fold_idx)][i]
#                     pred = np.argmax(prob)
#                     preds.append(pred)
#             pred_list.append(max(preds, key=preds.count))
#     else:
#         for i in range(data_len):
#             prob = None
#             for model_name in model_name_list:
#                 for fold_idx in range(5):
#                     if prob is None:
#                         prob = preds_dict['{}_{}'.format(
#                             model_name, fold_idx)][i] * ratio_dict[model_name]
#                     else:
#                         prob += preds_dict['{}_{}'.format(
#                             model_name, fold_idx)][i] * ratio_dict[model_name]
#             pred_list.append(np.argmax(prob))
#     return pred_list


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.config = args.config
    cfg.tag = args.tag

    if args.options is not None:
        cfg.merge_from_dict(args.options)

    set_seed(cfg)
    set_cudnn(cfg)
    set_work_dir(cfg)
    make_log_dir(cfg)
    set_gpu(cfg)
    log_func = lambda string='': print_log(string, cfg)

    valid_dataset = build_dataset(cfg.data.val)
    valid_dataloader = build_dataloader(dataset=valid_dataset, samples_per_gpu=cfg.batch_size,
                                        workers_per_gpu=cfg.num_workers, shuffle=False, pin_memory=False)

    test_dataset = build_dataset(cfg.data.test)
    test_dataloader = build_dataloader(dataset=test_dataset, samples_per_gpu=cfg.batch_size,
                                        workers_per_gpu=cfg.num_workers, shuffle=False, pin_memory=True)
    idx_cls_map = test_dataset.idx_to_class

    model = build_model(cfg)
    # Todo load_from
    # model.load_state_dict(torch.load(cfg.load_from))
    model.load_state_dict(torch.load(cfg.model_path))
    model = model.cuda()
    model.eval()
    log_func('[i] Architecture is {}'.format(cfg.model))
    log_func('[i] Total Params: %.2fM' % (calculate_parameters(model)))

    test_timer = Timer()
    # pred_list = single_model_predict()
    pred_list = single_model_predict_tta()
    # multi_model_predict_tta()
    # file2submission()


    # test_data = pd.read_csv('/home/muyun99/data/dataset/competition_data/kaggle_classify_leaves/classify-leaves/sample_submission.csv')
    # test_data['label'] = pd.Series([idx_cls_map[x] for x in pred_list])
    # submission = pd.concat([test_data['image'], test_data['label']], axis=1)
    submission = pd.DataFrame({
        "image": [f'images/{idx+18353}.jpg' for idx in range(len(pred_list))],
        "label": [idx_cls_map[x] for x in pred_list]
    })

    submission.to_csv(cfg.submission_path, index=False)
