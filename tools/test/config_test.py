# build_model
model = 'resnet18'
pretrained = True

# build_optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)

# build_loss
loss = dict(type='CrossEntropyLoss')

# build_scheduler
lr_scheduler = dict(type='MultiStepLR', milestones=[100, 150], gamma=0.1)

# build_dataset
# additional settings
num_classes = 7
train_ratio = 0.85
batch_size = 256
num_workers = 8
root = '/home/muyun99/data/dataset/competition_data/kaggle_classify_leaves/dataset'

# dataset settings
dataset_type = 'competition_base_dataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
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
    )

