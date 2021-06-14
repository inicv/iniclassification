from inicls import get_optimizer, get_model
from mmcv import Config
from tools.torch_utils import *

cfg = Config.fromfile('./config_test.py')
model = get_model(cfg=cfg)
optimizer = get_optimizer(cfg=cfg, model=model)

print(f'[i] Optimizer is {cfg.optimizer.type}')
print(f'[i] Learning rate is {cfg.optimizer.lr}')