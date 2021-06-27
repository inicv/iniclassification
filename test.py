import copy

import ttach as tta
from mmcv import Config
from tqdm import tqdm

from inicls import build_dataset, build_dataloader, build_model
from tools.torch_utils import *
from train import parse_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def single_model_predict(predict_model):
    data_len = 0
    total_acc = 0
    # 推理验证集性能
    pred_valid_cls_list = []
    with torch.no_grad():
        for data in tqdm(valid_dataloader):
            images, labels = data['img'], data['gt_label']
            images, labels = images.cuda(), labels.cuda()

            logits = predict_model(images)
            logits_bk = copy.deepcopy(logits)

            logits = torch.max(torch.softmax(logits, dim=1), dim=1)
            logits = logits[1].cpu().numpy()
            pred_valid_cls_list += logits.tolist()

            _acc = accuracy(logits_bk, labels)
            data_len += len(labels)
            total_acc += _acc[0].item() * len(labels)
    print(f'validation acc is {total_acc / data_len}')

    # 推理测试集并得到结果
    pred_test_cls_list = []
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            images = data['img']
            images = images.to(device)

            logits = predict_model(images)
            logits = torch.max(torch.softmax(logits, dim=1), dim=1)
            logits = logits[1].cpu().numpy()
            pred_test_cls_list += logits.tolist()
    return pred_valid_cls_list, pred_test_cls_list


def single_model_predict_tta(predict_model):
    transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        # tta.FiveCrops(224, 224)
    ])

    tta_model = tta.ClassificationTTAWrapper(predict_model, transforms)
    pred_cls_list = single_model_predict(tta_model)

    return pred_cls_list


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


def file2submission():
    valid_preds_dict = dict()
    test_preds_dict = dict()
    valid_data_len = len(valid_dataset)
    test_data_len = len(test_dataset)
    for model_name in model_name_list:
        valid_submission_path = get_valid_submission_path(cfg, model_name)
        test_submission_path = get_test_submission_path(cfg, model_name)
        valid_df = pd.read_csv(valid_submission_path)
        test_df = pd.read_csv(test_submission_path)
        valid_data_len = len(valid_df)
        test_data_len = len(test_df)
        valid_preds_dict[f'{model_name}'] = valid_df['label'].values
        test_preds_dict[f'{model_name}'] = test_df['label'].values

    valid_pred_list = get_pred_list(valid_preds_dict, valid_data_len)
    test_pred_list = get_pred_list(test_preds_dict, test_data_len)
    # calculate Metric
    # acc = calculate_acc_from_two_list(valid_pred_list, list(valid_data['label']))
    # print(f'validation acc is {acc}')
    return valid_pred_list, test_pred_list


def get_pred_list(preds_dict, data_len):
    pred_list = []
    for i in range(data_len):
        preds = []
        for model_name in model_name_list:
            pred = cls_idx_map[preds_dict[f'{model_name}'][i]]
            # pred = np.argmax(prob)
            preds.append(pred)
        pred_list.append(max(preds, key=preds.count))

    return pred_list


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
    valid_dataloader = build_dataloader(dataset=valid_dataset, samples_per_gpu=int(cfg.batch_size * 4),
                                        workers_per_gpu=cfg.num_workers, shuffle=False, pin_memory=False)

    test_dataset = build_dataset(cfg.data.test)
    test_dataloader = build_dataloader(dataset=test_dataset, samples_per_gpu=int(cfg.batch_size * 4),
                                       workers_per_gpu=cfg.num_workers, shuffle=False, pin_memory=True)
    idx_cls_map = test_dataset.idx_to_class
    cls_idx_map = test_dataset.class_to_idx

    valid_csv_path = os.path.join(cfg.data.val.data_prefix, cfg.data.val.ann_file)
    valid_data = pd.read_csv(valid_csv_path)

    # model = build_model(cfg)
    # model.load_state_dict(torch.load(cfg.model_path))
    # model = model.cuda()
    # model.eval()
    # log_func('[i] Architecture is {}'.format(cfg.model))
    # log_func('[i] Total Params: %.2fM' % (calculate_parameters(model)))
    #
    # test_timer = Timer()
    # # pred_valid_list, pred_test_list = single_model_predict(model)
    # pred_valid_list, pred_test_list = single_model_predict_tta(model)
    # valid_data['image'] = valid_data['filename']
    # valid_data['label'] = pd.Series(pred_valid_list)
    # valid_df = pd.concat([valid_data['image'], valid_data['label']], axis=1)
    #
    # test_df = pd.DataFrame({
    #     "image": [f'images/{idx + 18353}.jpg' for idx in range(len(pred_test_list))],
    #     "label": [idx_cls_map[x] for x in pred_test_list]
    # })
    #
    # valid_df.to_csv(cfg.valid_submission_path, index=False)
    # test_df.to_csv(cfg.test_submission_path, index=False)

    model_name_list = ['resnext101_32x8d_fold0', 'resnext101_32x8d_fold1', 'resnext101_32x8d_fold2',
                       'resnext101_32x8d_fold3', 'resnext101_32x8d_fold4', 'resnext101_32x8d', 'hrnet_w48',
                       'res2net101_26w_4s', 'legacy_seresnext101_32x4d', 'gluon_seresnext101_32x4d', 'res2next50',
                       'hrnet_w32', 'seresnext50_32x4d_fold0', 'seresnext50_32x4d_fold1', 'seresnext50_32x4d_fold2',
                       'seresnext50_32x4d_fold3', 'seresnext50_32x4d_fold4', 'resnest50d_fold0', 'resnest50d_fold1', 'resnest50d_fold2',
                       'resnest50d_fold3', 'resnest50d_fold4', 'res2net50_26w_6s_fold0', 'res2net50_26w_6s_fold1', 'res2net50_26w_6s_fold2',
                       'res2net50_26w_6s_fold3', 'res2net50_26w_6s_fold4']
    # model_name_list = ['seresnext50_32x4d_fold0', 'seresnext50_32x4d_fold1', 'seresnext50_32x4d_fold2',
    #                    'seresnext50_32x4d_fold3', 'seresnext50_32x4d_fold4']
    # model_name_list = ['resnest50d_fold0', 'resnest50d_fold1', 'resnest50d_fold2',
    #                    'resnest50d_fold3', 'resnest50d_fold4']
    # model_name_list = ['res2net50_26w_6s_fold0', 'res2net50_26w_6s_fold1', 'res2net50_26w_6s_fold2',
    #                    'res2net50_26w_6s_fold3', 'res2net50_26w_6s_fold4']
    pred_ensemble_valid_list, pred_ensemble_test_list = file2submission()

    valid_data['image'] = valid_data['filename']
    valid_data['label'] = pd.Series(pred_ensemble_valid_list)
    valid_df = pd.concat([valid_data['image'], valid_data['label']], axis=1)

    test_df = pd.DataFrame({
        "image": [f'images/{idx + 18353}.jpg' for idx in range(len(pred_ensemble_test_list))],
        "label": [idx_cls_map[x] for x in pred_ensemble_test_list]
    })
    valid_df.to_csv('valid_submission.csv', index=False)
    test_df.to_csv('test_submission.csv', index=False)
    print(f'{model_name_list}_TTA_ensemble saved!')
