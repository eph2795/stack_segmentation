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
        'shuffle': True,
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
            'weight': [1, 10]
        }
    }, 
#     {
#         'loss': 'Dice',
#         'weight': 0.5, 
#         'params': {
#             'mode': 'multiclass',
#             'log_loss': True,
#             'from_logits': True,
#             'smooth': 1,
#             'eps': 1e-7
#         }
#     }
]

model_config = {
    # 'source': 'basic',
    # 'model_type': None,
    # 'params': {
    #     'in_channels': 1,
    #     'num_classes': 2,
    #     'padding': True
    # },
    'source': 'qubvel',
    'model_type': 'Unet',
    'params': {
        'encoder_name': 'resnet101',
        'encoder_weights': 'imagenet',
        'classes': 1,
        'in_channels': 2,
    }
}

optimizer_config = {
    # 'opt_type': 'SGD',
    # 'params': {
    #     'lr': 1e-4,
    #     'weight_decay': 1e-4,
    #     'nesterov': False,
    #     'momentum': 0.9
    # }
    # 'opt_type': 'RMSprop',
    # 'params': {
    #     'lr': 1e-4,
    #     'weight_decay': 1e-4,
    #     'momentum': 0.9,
    #     'centered': False
    # }
    'opt_type': 'AdamW',
    'params': {
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'amsgrad': False
    }
}

scheduler_config = {
    'min_lr': 1e-6,
    'factor': 0.5,
    'patience': 3,
}

train_conf = {
    'num_epochs': 200,
    'device': 'cuda:1',
    'es_patience': 10
}
