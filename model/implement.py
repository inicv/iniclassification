import torch.nn as nn
from torchvision.models import resnet18, resnet34, wide_resnet50_2, wide_resnet101_2, resnet152
from torchvision.models import resnext50_32x4d, resnext101_32x8d
from torchvision.models import densenet121, densenet161, densenet169, densenet201
from efficientnet_pytorch import EfficientNet
import config
from .inceptionv4 import inceptionv4
from .xception import xception
from .senet import se_resnext50, se_resnext101
from resnest.torch import resnest50, resnest101, resnest200, resnest269


class Net(nn.Module):
    def __init__(self, model_name):
        super(Net, self).__init__()
        model = None
        if model_name == 'resnet18':
            model = resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, config.num_classes)
        elif model_name == 'resnet34':
            model = resnet34(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, config.num_classes)
        elif model_name == 'resnet50':
            model = wide_resnet50_2(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, config.num_classes)
        elif model_name == 'resnet101':
            model = wide_resnet101_2(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, config.num_classes)
        elif model_name == 'resnet152':
            model = resnet152(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, config.num_classes)
        elif model_name == 'resnext50':
            model = resnext50_32x4d(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, config.num_classes)
        elif model_name == 'resnext101':
            model = resnext101_32x8d(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, config.num_classes)
        elif model_name == 'densenet121':
            model = densenet121(pretrained=True)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, config.num_classes)
        elif model_name == 'densenet161':
            model = densenet161(pretrained=True)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, config.num_classes)
        elif model_name == 'densenet169':
            model = densenet169(pretrained=True)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, config.num_classes)
        elif model_name == 'densenet201':
            model = densenet201(pretrained=True)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, config.num_classes)
        elif model_name == 'inceptionv4':
            model = inceptionv4(pretrained='imagenet')
            model.last_linear = nn.Linear(model.last_linear.in_features, config.num_classes)
        elif model_name == 'xception':
            model = xception(pretrained='imagenet')
            model.last_linear = nn.Linear(model.last_linear.in_features, config.num_classes)
        elif model_name == 'se_resnext50':
            model = se_resnext50(num_classes=config.num_classes)
        elif model_name == 'se_resnext101':
            model = se_resnext101(num_classes=config.num_classes)
        elif model_name == 'efficientnet-b4':
            model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=config.num_classes)
        elif model_name == 'efficientnet-b7':
            model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=config.num_classes)
        elif model_name == 'resnest50':
            model = resnest50(pretrained=True)
        elif model_name == 'resnest101':
            model = resnest101(pretrained=True)
        elif model_name == 'resnest200':
            model = resnest200(pretrained=True)
        elif model_name == 'resnest269':
            model = resnest269(pretrained=True)

        self.model = model

    def forward(self, img):
        out = self.model(img)
        return out