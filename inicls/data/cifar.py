import torch
import torchvision
import torchvision.transforms as transforms
from .handle_transform import build_transforms

def build_CIFAR10_trainval_dataset(cfg):
    # transforms = build_transforms(cfg)
    augmentations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    trainval_dataset = torchvision.datasets.CIFAR10(root=cfg.root,
                                                    train=True,
                                                    download=True,
                                                    transform=augmentations)

    train_size = int(len(trainval_dataset) * cfg.train_ratio)
    valid_size = len(trainval_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, valid_size])

    return train_dataset, valid_dataset