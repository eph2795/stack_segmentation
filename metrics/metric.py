from typing import Union, Tuple

import numpy as np

import torch

from ..utils.functionals import softmax


class Metric:

    def __init__(self, from_logits: bool =False, axis: int = 1) -> None:
        self.from_logits = from_logits
        self.axis = axis

    @staticmethod
    def _torch_postprocess(
            result: torch.Tensor,
            masked_channels: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        result = result.cpu().data.numpy()
        masked_channels = masked_channels.cpu().data.numpy()
        return result, masked_channels

    def compute(
            self,
            mask: Union[np.ndarray, torch.Tensor],
            prediction: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        raise NotImplementedError

    def __call__(
            self,
            mask: Union[np.ndarray, torch.Tensor],
            prediction: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        if self.from_logits:
            prediction = softmax(prediction, axis=self.axis)

        result, masked_channels = self.compute(mask, prediction)

        if isinstance(result, torch.Tensor):
            result, masked_channels = self._torch_postprocess(result, masked_channels)

        return np.where(masked_channels, result, np.nan)
