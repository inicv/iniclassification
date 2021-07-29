import timm

def build_model(cfg):
    if cfg.model == 'resnet38':
        return resnet
    else:
        return timm.create_model(model_name=cfg.model, pretrained=cfg.pretrained, num_classes=cfg.num_classes)

