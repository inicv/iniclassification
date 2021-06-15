from inicls import build_model
from mmcv import Config
from tools.torch_utils import *

cfg = Config.fromfile('./config_test.py')
model = build_model(cfg=cfg)
model = model.cuda()
model.train()

print('[i] Architecture is {}'.format(cfg.model))
print('[i] Total Params: %.2fM' % (calculate_parameters(model)))