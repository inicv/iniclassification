from torch import optim


def get_scheduler(name_scheduler):
    super()
    scheduler = None
    if name_scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR()
    if name_scheduler == 'CosineLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR
    if name_scheduler == 'ReduceLR':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau

    return scheduler
