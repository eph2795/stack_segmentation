import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score

threshold = 0.5


def softmax(x):
    m = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=1, keepdims=True)


def accuracy(gt, logits):
    pred = softmax(logits)
    binary = (pred > threshold)
    return accuracy_score(gt.flatten(), binary[:, 1].flatten())


def precision(gt, logits):
    pred = softmax(logits)
    binary = (pred > threshold)
    return precision_score(gt.flatten(), binary[:, 1].flatten())


def recall(gt, logits):
    pred = softmax(logits)
    binary = (pred > threshold)
    return recall_score(gt.flatten(), binary[:, 1].flatten())


def f1(gt, logits):
    pred = softmax(logits)
    binary = (pred > threshold)
    return f1_score(gt.flatten(), binary[:, 1].flatten())


def pr_auc(gt, logits):
    pred = softmax(logits)
    return average_precision_score(gt.flatten(), pred[:, 1].flatten())


def iou(gt, logits):
    pred = softmax(logits)
    binary = (pred > threshold)
    
    intersection = ((binary[:, 1] == 1) & (gt == 1)).sum()
    union = (binary[:, 1] == 1).sum() + (gt == 1).sum() - intersection
    return intersection / union