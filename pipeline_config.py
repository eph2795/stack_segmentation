
aug_config = {
    'aug_type': 'medium',
    'original_height': 128,
    'original_width': 128,
    'k': 2,
}

dataloaders_conf = {
    'train': {
        'batch_size': 32,
        'num_workers': 16,
        'shuffle': True,
    },
   'val': {
        'batch_size': 32,
        'num_workers': 16,
        'shuffle': False,
    },
    'test': {
        'batch_size': 32,
        'num_workers': 16,
        'shuffle': True,
    },
}

loss_config = [
    {
        'loss': 'BCE',
        'weight': 0.5,
        'params': {
#             'weight': [1, 10]
        }
    }, 
    {
        'loss': 'Dice',
        'weight': 0.5, 
        'params': {
            'mode': 'multiclass',
            'log_loss': True,
            'from_logits': True,
            'smooth': 1,
            'eps': 1e-7
        }
    }
]

model_config = {
#     'source': 'basic',
    'source': 'qubvel',
    'model_type': 'Unet',
    'encoder_name': 'resnet50',
    'encoder_weights': 'imagenet',
}

optimizer_config = {
    'opt_type': 'AdamW',
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'amsgrad': False,
    'nesterov': False,
    'momentum': 0.9,
    'centered': False,
}

scheduler_config = {
    'min_lr': 1e-6,
    'factor': 0.5,
    'patience': 5,
}

train_conf = {
    'num_epochs': 200,
    'device': 'cuda:0',
#     'device': 'cpu',
}