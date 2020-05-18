from typing import Union, Tuple

import numpy as np

import torch

from .metric import Metric


class Precision(Metric):

    def __init__(self, from_logits=False, axis=1, eps=1e-7):
        super().__init__(from_logits=from_logits, axis=axis)
        self.eps = eps

    def compute(
            self,
            mask: Union[np.ndarray, torch.Tensor],
            prediction: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        tp = mask * prediction
        fp = (1 - mask) * prediction
        axes = tuple([i for i in range(len(mask.shape)) if i != self.axis])
        masked_channels = mask.sum(axes) > 0
        result = tp.sum(axes) / (fp + tp).sum(axes)
        return result, masked_channels
