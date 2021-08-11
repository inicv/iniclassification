from mmcv import Config

from inicls import build_dataset, build_dataloader
from tools.torch_utils import *
from train import parse_args


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

    valid_pred_list = get_pred_list(valid_preds_dict, valid_data_len, mode='valid')
    test_pred_list = get_pred_list(test_preds_dict, test_data_len, mode='test')
    # calculate Metric
    acc = calculate_acc_from_two_list(valid_pred_list, list(valid_data['label']))
    print(f'validation acc is {acc}')
    return valid_pred_list, test_pred_list


def get_pred_list(preds_dict, data_len, mode='valid'):
    pred_list = []
    for i in range(data_len):
        preds = []
        for model_name in model_name_list:
            if mode == 'test':
                pred = cls_idx_map[preds_dict[f'{model_name}'][i]]
            elif mode == 'valid':
                pred = preds_dict[f'{model_name}'][i]
            else:
                raise Exception('mode is illeagal')
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
    valid_dataloader = build_dataloader(dataset=valid_dataset, samples_per_gpu=cfg.batch_size,
                                        workers_per_gpu=cfg.num_workers, shuffle=False, pin_memory=False)

    test_dataset = build_dataset(cfg.data.test)
    test_dataloader = build_dataloader(dataset=test_dataset, samples_per_gpu=cfg.batch_size,
                                       workers_per_gpu=cfg.num_workers, shuffle=False, pin_memory=False)

    idx_cls_map = test_dataset.idx_to_class
    cls_idx_map = test_dataset.class_to_idx

    valid_csv_path = os.path.join(cfg.data.val.data_prefix, cfg.data.val.ann_file)
    valid_data = pd.read_csv(valid_csv_path)

    test_csv_path = os.path.join(cfg.data.test.data_prefix, cfg.data.test.ann_file)
    test_data = pd.read_csv(test_csv_path)

    model_name_list_resnet = ['resnet18_b16x8_xunfei_face_fp16_fold0_pseudo_labelsmooth',
                              'resnet18_b16x8_xunfei_face_fp16_fold1_pseudo_labelsmooth',
                              'resnet18_b16x8_xunfei_face_fp16_fold2_pseudo_labelsmooth',
                              'resnet18_b16x8_xunfei_face_fp16_fold3_pseudo_labelsmooth',
                              'resnet18_b16x8_xunfei_face_fp16_fold4_pseudo_labelsmooth']
    model_name_list_resnest50 = ['resnest50d_b16x8_xunfei_face_fp16_fold0', 'resnest50d_b16x8_xunfei_face_fp16_fold1',
                       'resnest50d_b16x8_xunfei_face_fp16_fold2',
                       'resnest50d_b16x8_xunfei_face_fp16_fold3', 'resnest50d_b16x8_xunfei_face_fp16_fold4', ]

    model_name_list = model_name_list_resnest50+model_name_list_resnet

    pred_ensemble_valid_list, pred_ensemble_test_list = file2submission()

    valid_data['image'] = valid_data['filename']
    valid_data['label'] = pd.Series(pred_ensemble_valid_list)
    valid_df = pd.concat([valid_data['image'], valid_data['label']], axis=1)

    test_data['name'] = test_data['filename'].apply(lambda x: x.split('/')[-1])
    test_data['label'] = pd.Series([idx_cls_map[x] for x in pred_ensemble_test_list])
    test_df = pd.concat([test_data['name'], test_data['label']], axis=1)

    test_data['image_raw'] = test_data['filename']
    test_data['label_raw'] = pd.Series(pred_ensemble_test_list)
    test_pseudo_df = pd.concat([test_data['image_raw'], test_data['label_raw']], axis=1)

    cfg.valid_ensemble_submission_path = cfg.valid_submission_path[:-4] + '_ensemble.csv'
    cfg.test_ensemble_submission_path = cfg.test_submission_path[:-4] + '_ensemble.csv'
    cfg.test_pseudo_path = cfg.test_submission_path[:-4] + '_pseudo.csv'

    valid_df.to_csv(cfg.valid_ensemble_submission_path, index=False)
    test_df.to_csv(cfg.test_ensemble_submission_path, index=False)
    test_pseudo_df.to_csv(cfg.test_pseudo_path, index=False)

    print(f'saving valid result in {cfg.valid_ensemble_submission_path}')
    print(f'saving test result in {cfg.test_ensemble_submission_path}')
    print(f'{model_name_list}_TTA_ensemble saved!')
