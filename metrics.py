import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score
)

import torch


def softmax(x, axis=1):
    if isinstance(x, np.ndarray):
        m = x.max(axis=axis, keepdims=True)
        e = np.exp(x - m)
        result = e / e.sum(axis=axis, keepdims=True)
    elif isinstance(x, torch.Tensor):
        m, _ = x.max(dim=axis, keepdim=True)
        e = torch.exp(x - m)
        result = e / e.sum(dim=axis, keepdim=True)
    else:
        raise ValueError('Wrong "x" type!')
    return result


def accuracy(gt, input, mode='batch', threshold=0.5):
    if mode == 'batch':
        pred = softmax(input)
        binary = (pred > threshold)[:, 1]
    else:
        if np.issubdtype(input.dtype, np.floating):
            binary = (input > threshold).astype(np.uint8)
        else:
            binary = input
    return accuracy_score(gt.flatten(), binary.flatten())


def precision(mask, prediction, logits=False, mode='batch', eps=1e-7):
    if mode == 'batch':
        axis = 1
    else:
        axis = len(mask.shape) - 1
    if logits:
        prediction = softmax(prediction, axis=axis)

    correct = mask == prediction

    intersection = mask * prediction
    union = mask + prediction - intersection
    axes = tuple([i for i in range(len(mask.shape)) if i != axis])
    masked_channels = mask.sum(axes) > 0
    result = intersection.sum(axes) / (union + eps).sum(axes)
    if isinstance(result, torch.Tensor):
        result = result.cpu().data.numpy()
        masked_channels = masked_channels.cpu().data.numpy()
    return np.where(masked_channels, result, np.nan)


# def precision(gt, input, mode='batch', threshold=0.5):
#     if mode== 'batch':
#         pred = softmax(input)
#         binary = (pred > threshold)[:, 1]
#     else:
#         if np.issubdtype(input.dtype, np.floating):
#             binary = (input > threshold).astype(np.uint8)
#         else:
#             binary = input
#     return precision_score(gt.flatten(), binary.flatten())


def recall(gt, input, mode='batch', threshold=0.5):
    if mode == 'batch':
        pred = softmax(input)
        binary = (pred > threshold)[:, 1]
    else:
        if np.issubdtype(input.dtype, np.floating):
            binary = (input > threshold).astype(np.uint8)
        else:
            binary = input
    return recall_score(gt.flatten(), binary.flatten())


def f1(gt, input, mode='batch', threshold=0.5):
    if mode == 'batch':
        pred = softmax(input)
        binary = (pred > threshold)[:, 1]
    else:
        if np.issubdtype(input.dtype, np.floating):
            binary = (input > threshold).astype(np.uint8)
        else:
            binary = input
    return f1_score(gt.flatten(), binary.flatten())


def pr_auc(gt, input, mode='batch'):
    if mode == 'batch':
        pred = softmax(input)[:, 1]
    else:
        pred = input
    return average_precision_score(gt.flatten(), pred.flatten())


def iou(mask, prediction, logits=False, mode='batch', eps=1e-7):
    if mode == 'batch':
        axis = 1
    else:
        axis = len(mask.shape) - 1
    if logits:
        prediction = softmax(prediction, axis=axis)
    intersection = mask * prediction
    union = mask + prediction - intersection
    axes = tuple([i for i in range(len(mask.shape)) if i != axis])
    masked_channels = mask.sum(axes) > 0
    result = intersection.sum(axes) / (union + eps).sum(axes)
    if isinstance(result, torch.Tensor):
        result = result.cpu().data.numpy()
        masked_channels = masked_channels.cpu().data.numpy()
    return np.where(masked_channels, result, np.nan)
