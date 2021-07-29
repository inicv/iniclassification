_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/voc2012aug.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]
loss = dict(type='multilabel_soft_margin_loss')