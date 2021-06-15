from mmcv import Config
from inicls.data import build_dataset
cfg = Config.fromfile('./config_test.py')

dataset = build_dataset(cfg.data.train)
print(dataset)