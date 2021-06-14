from pytorch_loss import *
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, CTCLoss, NLLLoss, PoissonNLLLoss, KLDivLoss, BCELoss, \
    BCEWithLogitsLoss, MarginRankingLoss, HingeEmbeddingLoss, MultiLabelMarginLoss, SmoothL1Loss, SoftMarginLoss, \
    MultiLabelSoftMarginLoss, CosineEmbeddingLoss, MultiMarginLoss, TripletMarginLoss, TripletMarginWithDistanceLoss


def build_loss(cfg):
    name_loss = cfg.loss.type
    criterion = None
    if name_loss == 'CrossEntropyLoss':
        criterion = CrossEntropyLoss()

    if criterion is None:
        raise Exception('criterion is wrong')
    return criterion