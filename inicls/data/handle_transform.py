from mmcv.utils import build_from_cfg
from .builder import PIPELINES
from mmcv import Config
import torchvision.transforms as transforms


transform_map = dict()

transform_map['RandomCrop'] = transforms.RandomCrop
transform_map['RandomHorizontalFlip'] = transforms.RandomHorizontalFlip
transform_map['ImageToTensor'] = transforms.ToTensor
transform_map['Normalize'] = transforms.Normalize



def build_transforms(cfg):
    pipeline = cfg.train_pipeline
    augmentations = build_from_cfg(cfg, PIPELINES)
    print(augmentations)

if __name__ == '__main__':
    cfg = Config.fromfile('/home/muyun99/MyGithub/iniclassification/tools/test/config_test.py')
    build_transforms(cfg)