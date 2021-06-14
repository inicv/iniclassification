import timm

def build_model(cfg):
    return timm.create_model(model_name=cfg.model, pretrained=cfg.pretrained, num_classes=cfg.num_classes)