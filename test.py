import copy

import ttach as tta
from mmcv import Config
from tqdm import tqdm

from inicls import build_dataset, build_dataloader, build_model
from tools.torch_utils import *
from train import parse_args
from torch.cuda.amp import autocast
from torchsummary import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def single_model_predict(predict_model):
    data_len = 0
    total_acc = 0

    # summary(predict_model, input_size=(3, 224, 224))
    # for i in range(10):
    #     model.eval()
    #     acc = valid_dataset.evaluate(cfg=cfg, model=predict_model, valid_dataloader=valid_dataloader)
    #     print(f'{[i]} validation acc is {acc}')

    # 推理验证集性能
    model.eval()
    pred_valid_cls_list = []
    with torch.no_grad():
        for data in tqdm(valid_dataloader):
            images, labels = data['img'], data['gt_label']
            images, labels = images.cuda(), labels.cuda()
            data_len += len(labels)
            if cfg.fp16:
                with autocast():
                    logits = predict_model(images)
            else:
                logits = predict_model(images)

            # logits_acc 用于计算 valid dataset 上的 acc
            logits_acc = copy.deepcopy(logits)
            _acc = accuracy(logits_acc, labels)
            total_acc += _acc[0].item() * len(labels)

            # 经过 softmax 可以得到最终的类别
            logits = torch.max(torch.softmax(logits, dim=1), dim=1)
            logits = logits[1].cpu().numpy()
            pred_valid_cls_list += logits.tolist()

    print(f'validation acc is {total_acc / data_len}')

    # 推理测试集并得到结果
    model.eval()
    pred_test_cls_list = []
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            images = data['img']
            images = images.to(device)

            if cfg.fp16:
                with autocast():
                    logits = predict_model(images)
            else:
                logits = predict_model(images)
            logits = torch.max(torch.softmax(logits, dim=1), dim=1)
            logits = logits[1].cpu().numpy()
            pred_test_cls_list += logits.tolist()
    return pred_valid_cls_list, pred_test_cls_list


def single_model_predict_tta(predict_model):
    transforms = tta.Compose([
        # tta.HorizontalFlip(),
        # tta.VerticalFlip(),
        # tta.FiveCrops(200, 200)
    ])

    tta_model = tta.ClassificationTTAWrapper(predict_model, transforms)
    pred_cls_list = single_model_predict(tta_model)
    return pred_cls_list

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

    model = build_model(cfg)
    model = model.cuda()
    log_func(f'[i] Architecture is {cfg.model}')
    log_func(f'[i] Total Params: %.2fM' % (calculate_parameters(model)))
    model.load_state_dict(torch.load(cfg.model_path))
    log_func(f'[i] Loading weight from: {cfg.model_path}')

    test_timer = Timer()
    pred_valid_list, pred_test_list = single_model_predict_tta(model)

    valid_data['image'] = valid_data['filename']
    valid_data['label'] = pd.Series(pred_valid_list)
    valid_df = pd.concat([valid_data['image'], valid_data['label']], axis=1)

    test_data['name'] = test_data['filename'].apply(lambda x: x.split('/')[-1])
    test_data['label'] = pd.Series([idx_cls_map[x] for x in pred_test_list])
    test_df = pd.concat([test_data['name'], test_data['label']], axis=1)

    valid_df.to_csv(cfg.valid_submission_path, index=False)
    test_df.to_csv(cfg.test_submission_path, index=False)