from torch import Tensor
from torch.nn.modules.loss import _Loss


__all__ = ["SoftLabelCE"]


class SoftLabelCE(_Loss):
    """
    Implementation of Cross Entropy loss with soft true labels
    """

    def __init__(
            self,
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
        self.reduction = reduction
        self.from_logits = from_logits
        self.eps = eps

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """

        :param y_pred: NxKxHxW
        :param y_true: NxCxHxW
        :return:
        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            y_pred = y_pred.log_softmax(dim=1)
        else:
            y_pred = y_pred.clamp_min(self.eps).log()

        scores = -(y_true * y_pred)

        if self.reduction == 'sum':
            return scores.sum()
        else:
            scores = scores.sum(dim=1)
            if self.reduction == 'mean':
                return scores.mean()
            else:
                return scores.sum(dim=1)
