from mmcv import Config

cfg = Config.fromfile('./config_test.py')

pipeline = cfg.train_pipeline
for transform in pipeline:
    type = transform['type']
print(pipeline)