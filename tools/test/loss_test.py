from inicls import get_optimizer, get_model, get_loss
from mmcv import Config


cfg = Config.fromfile('./config_test.py')
criterion = get_loss(cfg=cfg)

print(f'[i] Criterion is {cfg.loss.type}')