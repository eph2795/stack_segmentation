from typing import Union

from torch import Tensor
from torch.nn.modules.loss import _Loss


__all__ = ["SoftLabelCE"]


class SoftLabelCE(_Loss):
    """
    Implementation of Cross Entropy loss with soft true labels
    """

    def __init__(
            self,
            weight: Union[None, Tensor],
            reduction: str = 'mean',
            from_logits=True,
            eps=1e-7
    ):
        """

        :param reduction: one of "none", "sum", "mean"
        :param from_logits: If True assumes input is raw logits
        :param eps: Small epsilon for numerical stability
        """
        assert reduction in {'none', 'sum', 'mean'}
        super(SoftLabelCE, self).__init__()
        self.weight = weight
        if weight is not None:
            self.num_classes = weight.size(0)
        self.reduction = reduction
        self.from_logits = from_logits
        self.eps = eps

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """

        :param y_pred: NxCxHxW
        :param y_true: NxCxHxW
        :return:
        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            y_pred = y_pred.log_softmax(dim=1)
        else:
            y_pred = y_pred.clamp_min(self.eps).log()

        scores = -(y_true * y_pred)
        if self.weight is not None:
            scores = scores * self.weight.view(1, self.num_classes, 1, 1) # / self.weight.sum()

        if self.reduction == 'sum':
            return scores.sum()
        else:
            denom = (y_true * self.weight.view(1, self.num_classes, 1, 1)).sum()
            scores = scores.sum(dim=1)
            if self.reduction == 'mean':
                return (scores / denom).sum()
            return scores
