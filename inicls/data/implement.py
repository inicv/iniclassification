from .cifar import build_CIFAR10_trainval_dataset


def build_trainval_dataset(cfg):
    if cfg.dataset_type == 'CIFAR10':
        return build_CIFAR10_trainval_dataset(cfg=cfg)

def build_test_dataset(cfg):
    if cfg.dataset_type == 'CIFAR10':
        pass