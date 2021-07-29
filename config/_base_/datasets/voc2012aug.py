num_classes = 20
train_ratio = 1
batch_size = 64
num_workers = 8

# dataset settings
dataset_type = 'VOC2012Aug'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
valid_pipeline = [
    dict(type='RandomResizedCrop', size=224),
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
        data_prefix='/home/muyun99/data/dataset/Public-Dataset/VOC2012Aug',
        ann_file='ImageSets/Segmentation/trainaug.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/home/muyun99/data/dataset/Public-Dataset/VOC2012Aug',
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=valid_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='/home/muyun99/data/dataset/Public-Dataset/VOC2012Aug',
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline)
)