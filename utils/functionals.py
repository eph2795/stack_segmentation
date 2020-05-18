from typing import Union

import numpy as np

import torch


def softmax(x: Union[np.ndarray, torch.Tensor], axis: int = 1) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute numerical stable softmax along given axis

    Args:
        x: input multidimensional array
        axis: axis to compute softmax along

    Returns:
        result: multidimensional array with soft probabilities
    """

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
