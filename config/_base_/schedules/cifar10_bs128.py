# get_optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0)

# TODO: grad_clip
# optimizer_config = dict(grad_clip=None)

# get_loss
loss = dict(type='CrossEntropyLoss')

# get_scheduler
lr_scheduler = dict(type='MultiStepLR', milestones=[40, 45], gamma=0.1)

max_epoch = 50