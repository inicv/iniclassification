from pytorch_loss import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, CTCLoss, NLLLoss, PoissonNLLLoss, KLDivLoss, BCELoss, \
    BCEWithLogitsLoss, MarginRankingLoss, HingeEmbeddingLoss, MultiLabelMarginLoss, SmoothL1Loss, SoftMarginLoss, \
    MultiLabelSoftMarginLoss, CosineEmbeddingLoss, MultiMarginLoss, TripletMarginLoss, TripletMarginWithDistanceLoss

class LabelSmoothCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label, smoothing=0.1):
        pred = F.softmax(pred, dim=1)
        one_hot_label = F.one_hot(label, pred.size(1)).float()
        smoothed_one_hot_label = (
            1.0 - smoothing) * one_hot_label + smoothing / pred.size(1)
        loss = (-torch.log(pred)) * smoothed_one_hot_label
        loss = loss.sum(axis=1, keepdim=False)
        loss = loss.mean()

        return loss

def build_loss(cfg):
    name_loss = cfg.loss.type
    criterion = None
    if name_loss == 'CrossEntropyLoss':
        criterion = CrossEntropyLoss()
    elif name_loss == 'LabelSmoothCELoss':
        criterion = LabelSmoothCELoss()

    if criterion is None:
        raise Exception('criterion is wrong')
    return criterion