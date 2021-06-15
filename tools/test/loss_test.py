from inicls import build_optimizer, build_model, build_loss
from mmcv import Config


cfg = Config.fromfile('./config_test.py')
criterion = build_loss(cfg=cfg)

print(f'[i] Criterion is {cfg.loss.type}')