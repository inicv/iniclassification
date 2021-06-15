import torch_optimizer as optim
import torch.optim
def build_optimizer(cfg, model):
    name_optimizer = cfg.optimizer.type
    optimizer = None

    if name_optimizer == 'A2GradExp':
        optimizer = optim.A2GradExp(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'A2GradInc':
        optimizer = optim.A2GradInc(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'A2GradUni':
        optimizer = optim.A2GradUni(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'AccSGD':
        optimizer = optim.AccSGD(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'AdaBelief':
        optimizer = optim.AdaBelief(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'AdaBound':
        optimizer = optim.AdaBound(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'AdaMod':
        optimizer = optim.AdaMod(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'Adafactor':
        optimizer = optim.Adafactor(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'AdamP':
        optimizer = optim.AdamP(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'AggMo':
        optimizer = optim.AggMo(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'Apollo':
        optimizer = optim.Apollo(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'DiffGrad':
        optimizer = optim.DiffGrad(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'Lamb':
        optimizer = optim.Lamb(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'Lookahead':
        yogi = optim.Yogi(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
        optimizer = optim.Lookahead(yogi, k=5, alpha=0.5)
    if name_optimizer == 'NovoGrad':
        optimizer = optim.NovoGrad(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'PID':
        optimizer = optim.PID(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'QHAdam':
        optimizer = optim.QHAdam(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'QHM':
        optimizer = optim.QHM(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'RAdam':
        optimizer = optim.RAdam(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'Ranger':
        optimizer = optim.Ranger(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'RangerQH':
        optimizer = optim.RangerQH(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'RangerVA':
        optimizer = optim.RangerVA(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'SGDP':
        optimizer = optim.SGDP(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'SGDW':
        optimizer = optim.SGDW(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'SWATS':
        optimizer = optim.SWATS(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'Shampoo':
        optimizer = optim.Shampoo(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'Yogi':
        optimizer = optim.Yogi(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if name_optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr
        )
    if optimizer is None:
        raise Exception('optimizer is wrong')
    return optimizer
