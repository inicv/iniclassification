# build_model
model = 'resnet18'
pretrained = True
num_classes = 10

# build_optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)

# build_loss
loss = dict(type='CrossEntropyLoss')

# build_scheduler
lr_scheduler = dict(type='MultiStepLR', milestones=[100, 150], gamma=0.1)

# build_dataset
dataset_type = 'CIFAR10'
num_classes = 10
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=True)
root = '/home/muyun99/data/dataset/Public-Dataset/cifar-10'
train_ratio = 0.8
batch_size = 32
num_workers = 8
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
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, data_prefix='/home/muyun99/data/dataset/Public-Dataset/cifar-10',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/home/muyun99/data/dataset/Public-Dataset/cifar-10',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='/home/muyun99/data/dataset/Public-Dataset/cifar-10',
        pipeline=test_pipeline,
        test_mode=True))
