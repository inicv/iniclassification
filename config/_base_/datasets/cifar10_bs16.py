# additional settings
num_classes = 10
train_ratio = 0.8
batch_size = 128
num_workers = 8
root = '/home/muyun99/data/dataset/Public-Dataset/cifar-10'

# dataset settings
dataset_type = 'CIFAR10'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=True)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
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
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=root,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix=root,
        pipeline=test_pipeline,
        test_mode=True))
