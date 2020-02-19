import torch
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss

from pytorch_toolbelt.losses import DiceLoss, FocalLoss, LovaszLoss, WeightedLoss


class JointLoss(_Loss):
    def __init__(self, *losses):
        super().__init__()
        self.losses = losses
        
    def forward(self, y_pred, y_gt):
        return sum(loss(y_pred, y_gt) for loss in self.losses)

    
def _loss_factory(loss, weight, params, device):
    if loss == 'BCE':
        if 'weight' in params:
            params['weight'] = torch.FloatTensor(params['weight']).to(device)
        criterion = CrossEntropyLoss(**params)
    elif loss == 'Dice':
        criterion = DiceLoss(**params)
    elif loss == 'Focal':
        criterion = FocalLoss(**params)
    elif loss == 'Lovasz':
        criterion = LovaszLoss(**params)
    else:
        raise ValueError('Wrong loss type!')
    return WeightedLoss(criterion, weight=weight)


def make_joint_loss(loss_config, device):    
    return JointLoss(*[_loss_factory(device=device, **loss_params) for loss_params in loss_config])