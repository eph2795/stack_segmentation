import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score


def softmax(x):
    m = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=1, keepdims=True)


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


def precision(gt, input, mode='batch', threshold=0.5):
    if mode== 'batch':
        pred = softmax(input)
        binary = (pred > threshold)[:, 1]
    else:
        if np.issubdtype(input.dtype, np.floating):
            binary = (input > threshold).astype(np.uint8)
        else:
            binary = input
    return precision_score(gt.flatten(), binary.flatten())


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


def iou(gt, input, mode='batch', threshold=0.5):
    if mode == 'batch':
        pred = softmax(input)
        binary = (pred > threshold)[:, 1]
    else:
        if np.issubdtype(input.dtype, np.floating):
            binary = (input > threshold).astype(np.uint8)
        else:
            binary = input
    intersection = ((binary == 1) & (gt == 1)).sum()
    union = (binary == 1).sum() + (gt == 1).sum() - intersection
    return intersection / union