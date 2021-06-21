import numpy as np
import torch


def calculate_confusion_matrix(pred, target):
    """Calculate confusion matrix according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).

    Returns:
        torch.Tensor: Confusion matrix with shape (C, C), where C is the number
             of classes.
    """

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert (
        isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor)), \
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    num_classes = pred.size(1)
    _, pred_label = pred.topk(1, dim=1)
    pred_label = pred_label.view(-1)
    target_label = target.view(-1)
    assert len(pred_label) == len(target_label)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for t, p in zip(target_label, pred_label):
            confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix


def precision_recall_f1(pred, target, average_mode='macro', thrs=None):
    """Calculate precision, recall and f1 score according to the prediction and
    target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (float | tuple[float], optional): Predictions with scores under
            the thresholds are considered negative. Default to None.

    Returns:
        float | np.array | list[float | np.array]: Precision, recall, f1 score.
            If the ``average_mode`` is set to macro, np.array is used in favor
            of float to give class-wise results. If the ``average_mode`` is set
             to none, float is used to return a single value.
            If ``thrs`` is a single float or None, the function will return
            float or np.array. If ``thrs`` is a tuple, the function will return
             a list containing metrics for each ``thrs`` condition.
    """

    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')

    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    assert (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)),\
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    if thrs is None:
        thrs = 0.0
    if isinstance(thrs, float):
        thrs = (thrs, )
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError(
            f'thrs should be float or tuple, but got {type(thrs)}.')

    label = np.indices(pred.shape)[1]
    pred_label = np.argsort(pred, axis=1)[:, -1]
    pred_score = np.sort(pred, axis=1)[:, -1]

    precisions = []
    recalls = []
    f1_scores = []
    for thr in thrs:
        # Only prediction values larger than thr are counted as positive
        _pred_label = pred_label.copy()
        if thr is not None:
            _pred_label[pred_score <= thr] = -1
        pred_positive = label == _pred_label.reshape(-1, 1)
        gt_positive = label == target.reshape(-1, 1)
        precision = (pred_positive & gt_positive).sum(0) / np.maximum(
            pred_positive.sum(0), 1) * 100
        recall = (pred_positive & gt_positive).sum(0) / np.maximum(
            gt_positive.sum(0), 1) * 100
        f1_score = 2 * precision * recall / np.maximum(precision + recall,
                                                       1e-20)
        if average_mode == 'macro':
            precision = float(precision.mean())
            recall = float(recall.mean())
            f1_score = float(f1_score.mean())
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    if return_single:
        return precisions[0], recalls[0], f1_scores[0]
    else:
        return precisions, recalls, f1_scores


def precision(pred, target, average_mode='macro', thrs=None):
    """Calculate precision according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (float | tuple[float], optional): Predictions with scores under
            the thresholds are considered negative. Default to None.

    Returns:
         float | np.array | list[float | np.array]: Precision.
            If the ``average_mode`` is set to macro, np.array is used in favor
            of float to give class-wise results. If the ``average_mode`` is set
             to none, float is used to return a single value.
            If ``thrs`` is a single float or None, the function will return
            float or np.array. If ``thrs`` is a tuple, the function will return
             a list containing metrics for each ``thrs`` condition.
    """
    precisions, _, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return precisions


def recall(pred, target, average_mode='macro', thrs=None):
    """Calculate recall according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (float | tuple[float], optional): Predictions with scores under
            the thresholds are considered negative. Default to None.

    Returns:
         float | np.array | list[float | np.array]: Recall.
            If the ``average_mode`` is set to macro, np.array is used in favor
            of float to give class-wise results. If the ``average_mode`` is set
             to none, float is used to return a single value.
            If ``thrs`` is a single float or None, the function will return
            float or np.array. If ``thrs`` is a tuple, the function will return
             a list containing metrics for each ``thrs`` condition.
    """
    _, recalls, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return recalls


def f1_score(pred, target, average_mode='macro', thrs=None):
    """Calculate F1 score according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (float | tuple[float], optional): Predictions with scores under
            the thresholds are considered negative. Default to None.

    Returns:
         float | np.array | list[float | np.array]: F1 score.
            If the ``average_mode`` is set to macro, np.array is used in favor
            of float to give class-wise results. If the ``average_mode`` is set
             to none, float is used to return a single value.
            If ``thrs`` is a single float or None, the function will return
            float or np.array. If ``thrs`` is a tuple, the function will return
             a list containing metrics for each ``thrs`` condition.
    """
    _, _, f1_scores = precision_recall_f1(pred, target, average_mode, thrs)
    return f1_scores


def support(pred, target, average_mode='macro'):
    """Calculate the total number of occurrences of each label according to the
    prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted sum.
            Defaults to 'macro'.

    Returns:
        float | np.array: Precision, recall, f1 score.
            The function returns a single float if the average_mode is set to
            macro, or a np.array with shape C if the average_mode is set to
             none.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        res = confusion_matrix.sum(1)
        if average_mode == 'macro':
            res = float(res.sum().numpy())
        elif average_mode == 'none':
            res = res.numpy()
        else:
            raise ValueError(f'Unsupport type of averaging {average_mode}.')
    return res


def average_precision(pred, target):
    """Calculate the average precision for a single class.
    AP summarizes a precision-recall curve as the weighted mean of maximum
    precisions obtained for any r'>r, where r is the recall:
    ..math::
        \\text{AP} = \\sum_n (R_n - R_{n-1}) P_n
    Note that no approximation is involved since the curve is piecewise
    constant.
    Args:
        pred (np.ndarray): The model prediction with shape (N, ).
        target (np.ndarray): The target of each prediction with shape (N, ).
    Returns:
        float: a single float as average precision value.
    """
    eps = np.finfo(np.float32).eps

    # sort examples
    sort_inds = np.argsort(-pred)
    sort_target = target[sort_inds]

    # count true positive examples
    pos_inds = sort_target == 1
    tp = np.cumsum(pos_inds)
    total_pos = tp[-1]

    # count not difficult examples
    pn_inds = sort_target != -1
    pn = np.cumsum(pn_inds)

    tp[np.logical_not(pos_inds)] = 0
    precision = tp / np.maximum(pn, eps)
    ap = np.sum(precision) / np.maximum(total_pos, eps)
    return ap


def mAP(pred, target):
    """Calculate the mean average precision with respect of classes.
    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 1 stands for
            positive examples, 0 stands for negative examples and -1 stands for
            difficult examples.
    Returns:
        float: A single float as mAP value.
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')

    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'
    num_classes = pred.shape[1]
    ap = np.zeros(num_classes)
    for k in range(num_classes):
        ap[k] = average_precision(pred[:, k], target[:, k])
    mean_ap = ap.mean() * 100.0
    return mean_ap