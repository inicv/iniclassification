from inicls import build_optimizer, build_model, build_scheduler
from mmcv import Config
from tools.torch_utils import *

cfg = Config.fromfile('./config_test.py')

model = build_model(cfg=cfg)
optimizer = build_optimizer(cfg=cfg, model=model)
scheduler = build_scheduler(cfg=cfg, optimizer=optimizer)

print(f'[i] lr_scheduler is {cfg.lr_scheduler.type}')
print(f'[i] lr_scheduler milestones is {cfg.lr_scheduler.milestones}')
print(f'[i] lr_scheduler gamma is {cfg.lr_scheduler.gamma}')

for i in range(200):
    optimizer.step()
    scheduler.step()
    print(f'{i}: {scheduler.build_last_lr()}')