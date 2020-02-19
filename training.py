from collections import defaultdict

from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

import segmentation_models_pytorch as smp

from .stack import Stack
from .unet import UNet
from .early_stopping import EarlyStopping
from .loss import make_joint_loss


def handle_stacks_data(stacks, patches, **kwargs):
    data_train, data_val = [], []
    data_test = dict()
    for stack_conf in stacks:
        stack = Stack.read_from_source(stack_conf['path'])
        
        if 'slice_train' in stack_conf:
            stack_train = stack[stack_conf['slice_train']]
            data_train.extend(stack_train.slice_up(patches['train']))
        if 'slice_val' in stack_conf:
            stack_val = stack[stack_conf['slice_val']]
            data_val.extend(stack_val.slice_up(patches['val']))
        if 'slice_test' in stack_conf:
            stack_test = stack[stack_conf['slice_test']]
            data_test[stack_conf['path']] = stack_test.slice_up(patches['test'])
    return data_train, data_val, data_test


def make_optimizer(
        opt_type, 
        parameters, 
        lr=1e-3, 
        weight_decay=0, 
        amsgrad=False, 
        nesterov=False, 
        momentum=0.9, 
        centered=False,
        **kwargs
    ):
    
    if opt_type == 'SGD':
        opt = SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    elif opt_type == 'Adam':
        opt = Adam(parameters, lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
    elif opt_type == 'AdamW':
        opt = AdamW(parameters, lr=lr, weight_decay=weight_decay, amsgrad=amsgrad) 
    elif opt_type == 'RMSprop':
        opt = RMSprop(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum, centered=centered)
    return opt



def make_model(
        source, 
        model_type=None, 
        encoder_name=None, 
        encoder_weights=None):
    if source == 'basic':
        model = UNet(in_channels=1, n_classes=2, padding=True)
    elif source == 'qubvel':
        if model_type == 'Unet':
            model = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=2, activation=None)
    else:
        raise ValueError('Wrong model source!')
    return model
    
    
def make_optimization_task(
        device, 
        model_config,
        loss_config,
        optimizer_config,
        scheduler_config):
    model = make_model(**model_config).to(device)
    criterion = make_joint_loss(loss_config, device)
    optimizer = make_optimizer(parameters=model.parameters(), **optimizer_config)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=True, **scheduler_config)
    return model, criterion, optimizer, scheduler


def train_loop(
        model, 
        dataloader_train, 
        dataloader_val, 
        dataloaders_test, 
        criterion, 
        optimizer, 
        scheduler,
        metrics,
        device, 
        num_epochs,
        exp_name,
        es_patience=15):
    
    train_losses = []
    val_losses = []
    es = EarlyStopping(patience=es_patience, verbose=False, delta=1e-6, checkpoint_path='{}.pt'.format(exp_name))
        
    for i in range(num_epochs):
        print('Epoch {}...'.format(i))
        
        model.train() 
        losses = []
        for x, y in tqdm(dataloader_train):
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)

            out = model(x)

            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.cpu().data.numpy())
        train_losses.append(np.array(losses))
        print('Mean train loss: {:.5}'.format(np.mean(losses)))
        
        scheduler.step(np.mean(losses))
        
        model.eval()
        losses = []
        for x, y in tqdm(dataloader_val):
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)

            out = model(x)

            loss = criterion(out, y)

            losses.append(loss.cpu().data.numpy())
        val_losses.append(np.array(losses))
        print('Mean val loss: {:.5}'.format(np.mean(losses)))
        
        es(np.mean(losses), model)
        if es.early_stop:
            break
    
    model.load_state_dict(torch.load('{}.pt'.format(exp_name)))
    model.eval()
    metrics_dict = dict()
    for stack_name, dataloader_test in dataloaders_test.items():
        
        stack_dict = defaultdict(list)
        for x, y in tqdm(dataloader_test):
            x = torch.from_numpy(x).to(device)
            out = model(x).cpu().data.numpy()

            for metric_name, fn in metrics.items():
                stack_dict[metric_name].append(fn(y, out))
        
#         for metric_name in metrics:
#             metrics_dict[metric_name] = np.array(stack_dict[metric_name])
        metrics_dict[stack_name] = stack_dict
    
    results = {
        'model_checkpoint': '{}.pt'.format(exp_name),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_metrics': metrics_dict,
    }
    
    return results