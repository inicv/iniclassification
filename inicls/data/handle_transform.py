from mmcv.utils import build_from_cfg
import torchvision.transforms as transforms

transform_map = dict()

transform_map['RandomCrop'] = transforms.RandomCrop
transform_map['RandomHorizontalFlip'] = transforms.RandomHorizontalFlip
transform_map['ImageToTensor'] = transforms.ToTensor
transform_map['Normalize'] = transforms.Normalize



def build_transforms(cfg):
    pipeline = cfg.train_pipeline

