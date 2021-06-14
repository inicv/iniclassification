from torch import optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, \
    ExponentialLR, CosineAnnealingWarmRestarts


def build_scheduler(cfg, optimizer):
    name_scheduler = cfg.lr_scheduler.type
    scheduler = None

    if name_scheduler == 'StepLR':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step()
        scheduler = StepLR(optimizer=optimizer, step_size=cfg.lr_scheduler.step_size, gamma=cfg.lr_scheduler.gamma)
    elif name_scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=cfg.lr_scheduler.T_max)
    elif name_scheduler == 'ReduceLROnPlateau':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step(val_loss)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode=cfg.lr_scheduler.mode)
    elif name_scheduler == 'LambdaLR':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step()
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=cfg.lr_scheduler.lr_lambda)
    elif name_scheduler == 'MultiStepLR':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step()
        scheduler = MultiStepLR(optimizer=optimizer, milestones=cfg.lr_scheduler.milestones, gamma=cfg.lr_scheduler.gamma)
    elif name_scheduler == 'CyclicLR':
        # >>> for epoch in range(10):
        # >>>   for batch in data_loader:
        # >>>       train_batch(...)
        # >>>       scheduler.step()
        scheduler = CyclicLR(optimizer=optimizer, base_lr=cfg.lr_scheduler.base_lr, max_lr=cfg.lr_scheduler.max_lr)
    elif name_scheduler == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer=optimizer, gamma=cfg.lr_scheduler.gamma)
    elif name_scheduler == 'CosineAnnealingWarmRestarts':
        # >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
        # >>> for epoch in range(20):
        #     >>> scheduler.step()
        # >>> scheduler.step(26)
        # >>> scheduler.step()  # scheduler.step(27), instead of scheduler(20)
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=cfg.lr_scheduler.T_0,
                                                T_mult=cfg.lr_scheduler.T_mult)

    if scheduler is None:
        raise Exception('scheduler is wrong')
    return scheduler
