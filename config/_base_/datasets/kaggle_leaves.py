# additional settings
num_classes = 176
train_ratio = 0.8
batch_size = 64
num_workers = 8
root = '/home/muyun99/data/dataset/competition_data/kaggle_classify_leaves/dataset'

# dataset settings
dataset_type = 'competition_base_dataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
train_pipeline = [
    dict(type='RandomCrop', size=224, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
valid_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix=root,
        pipeline=train_pipeline,
        ann_file='train_fold0.csv',
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_prefix=root,
        pipeline=valid_pipeline,
        ann_file='valid_fold0.csv',
        test_mode=False),
    test=dict(
        type=dataset_type,
        data_prefix=root,
        pipeline=test_pipeline,
        ann_file='test.csv',
        test_mode=True))
