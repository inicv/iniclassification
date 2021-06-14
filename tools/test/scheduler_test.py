from inicls import get_optimizer, get_model, get_scheduler
from mmcv import Config
from tools.torch_utils import *

cfg = Config.fromfile('./config_test.py')

model = get_model(cfg=cfg)
optimizer = get_optimizer(cfg=cfg, model=model)
scheduler = get_scheduler(cfg=cfg, optimizer=optimizer)

print(f'[i] lr_scheduler is {cfg.lr_scheduler.type}')
print(f'[i] lr_scheduler milestones is {cfg.lr_scheduler.milestones}')
print(f'[i] lr_scheduler gamma is {cfg.lr_scheduler.gamma}')

for i in range(200):
    optimizer.step()
    scheduler.step()
    print(f'{i}: {scheduler.get_last_lr()}')