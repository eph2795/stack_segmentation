from .aug_pipelines import medium_aug


dataloaders_conf = {
    'train': {
        'batch_size': 32,
        'num_workers': 8,
        'shuffle': True,
        'augmentation_pipeline': medium_aug(original_height=128, original_width=128),
    },
   'val': {
        'batch_size': 32,
        'num_workers': 8,
        'shuffle': False,
        'augmentation_pipeline': None,
    },
    'test': {
        'batch_size': 32,
        'num_workers': 8,
        'shuffle': False,
        'augmentation_pipeline': None,
    },
}

model_conf = {
    'device': 'cuda:0',
    'lr': 1e-4,
    'factor': 0.75,
    'patience': 2,
}

train_conf = {
    'num_epochs': 50,
    'device': 'cuda:0',
}