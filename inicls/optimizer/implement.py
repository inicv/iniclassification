import torch_optimizer as optim
import torch.optim
def get_optimizer(name_optimizer, model, lr):
    optimizer = None
    if name_optimizer == 'A2GradExp':
        optimizer = optim.A2GradExp(
            model.parameters(),
            # kappa=1000.0,
            beta=10.0,
            lips=10.0,
            rho=0.5,
        )
    if name_optimizer == 'A2GradInc':
        optimizer = optim.A2GradInc(
            model.parameters(),
            # kappa=1000.0,
            beta=10.0,
            lips=10.0,
        )
    if name_optimizer == 'A2GradUni':
        optimizer = optim.A2GradUni(
            model.parameters(),
            # kappa=1000.0,
            beta=10.0,
            lips=10.0,
        )
    if name_optimizer == 'AccSGD':
        optimizer = optim.AccSGD(
            model.parameters(),
            lr=lr,
            kappa=1000.0,
            xi=10.0,
            small_const=0.7,
            weight_decay=0
        )

    if name_optimizer == 'AdaBelief':
        optimizer = optim.AdaBelief(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-3,
            weight_decay=0,
            amsgrad=False,
            weight_decouple=False,
            fixed_decay=False,
            rectify=False,
        )

    if name_optimizer == 'AdaBound':
        optimizer = optim.AdaBound(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            final_lr=0.1,
            gamma=1e-3,
            eps=1e-8,
            weight_decay=0,
            amsbound=False,
        )
    if name_optimizer == 'AdaMod':
        optimizer = optim.AdaMod(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            beta3=0.999,
            eps=1e-8,
            weight_decay=0,
        )
    if name_optimizer == 'Adafactor':
        optimizer = optim.Adafactor(
            model.parameters(),
            lr=lr,
            eps2=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            scale_parameter=True,
            relative_step=True,
            warmup_init=False,
        )
    if name_optimizer == 'AdamP':
        optimizer = optim.AdamP(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            delta=0.1,
            wd_ratio=0.1
        )
    if name_optimizer == 'AggMo':
        optimizer = optim.AggMo(
            model.parameters(),
            lr=lr,
            betas=(0.0, 0.9, 0.99),
            weight_decay=0,
        )
    if name_optimizer == 'Apollo':
        optimizer = optim.Apollo(
            model.parameters(),
            lr=1e-2,
            beta=0.9,
            eps=1e-4,
            warmup=0,
            init_lr=0.01,
            weight_decay=0,
        )
    if name_optimizer == 'DiffGrad':
        optimizer = optim.DiffGrad(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
        )
    if name_optimizer == 'Lamb':
        optimizer = optim.Lamb(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
        )
    if name_optimizer == 'Lookahead':
        yogi = optim.Yogi(
            model.parameters(),
            lr=1e-2,
            betas=(0.9, 0.999),
            eps=1e-3,
            initial_accumulator=1e-6,
            weight_decay=0,
        )
        optimizer = optim.Lookahead(yogi, k=5, alpha=0.5)
    if name_optimizer == 'NovoGrad':
        optimizer = optim.NovoGrad(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            grad_averaging=False,
            amsgrad=False,
        )
    if name_optimizer == 'PID':
        optimizer = optim.PID(
            model.parameters(),
            lr=lr,
            momentum=0,
            dampening=0,
            weight_decay=1e-2,
            integral=5.0,
            derivative=10.0,
        )
    if name_optimizer == 'QHAdam':
        optimizer = optim.QHAdam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            nus=(1.0, 1.0),
            weight_decay=0,
            decouple_weight_decay=False,
            eps=1e-8,
        )
    if name_optimizer == 'QHM':
        optimizer = optim.QHM(
            model.parameters(),
            lr=lr,
            momentum=0,
            nu=0.7,
            weight_decay=1e-2,
            weight_decay_type='grad',
        )
    if name_optimizer == 'RAdam':
        optimizer = optim.RAdam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
        )
    if name_optimizer == 'Ranger':
        optimizer = optim.Ranger(
            model.parameters(),
            lr=lr,
            alpha=0.5,
            k=6,
            N_sma_threshhold=5,
            betas=(.95, 0.999),
            eps=1e-5,
            weight_decay=0
        )
    if name_optimizer == 'RangerQH':
        optimizer = optim.RangerQH(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            nus=(.7, 1.0),
            weight_decay=0.0,
            k=6,
            alpha=.5,
            decouple_weight_decay=False,
            eps=1e-8,
        )
    if name_optimizer == 'RangerVA':
        optimizer = optim.RangerVA(
            model.parameters(),
            lr=lr,
            alpha=0.5,
            k=6,
            n_sma_threshhold=5,
            betas=(.95, 0.999),
            eps=1e-5,
            weight_decay=0,
            amsgrad=True,
            transformer='softplus',
            smooth=50,
            grad_transformer='square'
        )
    if name_optimizer == 'SGDP':
        optimizer = optim.SGDP(
            model.parameters(),
            lr=lr,
            momentum=0,
            dampening=0,
            weight_decay=1e-2,
            nesterov=False,
            delta=0.1,
            wd_ratio=0.1
        )
    if name_optimizer == 'SGDW':
        optimizer = optim.SGDW(
            model.parameters(),
            lr=lr,
            momentum=0,
            dampening=0,
            weight_decay=1e-2,
            nesterov=False,
        )
    if name_optimizer == 'SWATS':
        optimizer = optim.SWATS(
            model.parameters(),
            lr=1e-1,
            betas=(0.9, 0.999),
            eps=1e-3,
            weight_decay=0.0,
            amsgrad=False,
            nesterov=False,
        )
    if name_optimizer == 'Shampoo':
        optimizer = optim.Shampoo(
            model.parameters(),
            lr=1e-1,
            momentum=0.0,
            weight_decay=0.0,
            epsilon=1e-4,
            update_freq=1,
        )
    if name_optimizer == 'Yogi':
        optimizer = optim.Yogi(
            model.parameters(),
            lr=1e-2,
            betas=(0.9, 0.999),
            eps=1e-3,
            initial_accumulator=1e-6,
            weight_decay=0,
        )
    if name_optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999)
        )
    if name_optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9
        )
    return optimizer
