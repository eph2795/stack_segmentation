from collections import defaultdict

from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .stack import Stack
from .unet import UNet
from .early_stopping import EarlyStopping


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


def make_model(device, lr, factor, patience):
    model = UNet(in_channels=1, n_classes=2, padding=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
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
        exp_name):
    
    train_losses = []
    val_losses = []
    es = EarlyStopping(patience=5, verbose=False, delta=2.5e-5, checkpoint_path='{}.pt'.format(exp_name))
        
    for i in range(num_epochs):
        print('Epoch {}...'.format(i))
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
    metrics_dict = dict()
    for stack_name, dataloader_test in dataloaders_test.items():
        
        stack_dict = defaultdict(list)
        for x, y in tqdm(dataloader_test):
            x = torch.from_numpy(x).to(device)
            out = model(x).cpu().data.numpy()

            for metric_name, fn in metrics.items():
                stack_dict[metric_name].append(fn(y, out))
        
        for metric_name in metrics:
            metrics_dict[metric_name] = np.array(stack_dict[metric_name])
        metrics_dict[stack_name] = stack_dict
    
    results = {
        'model_checkpoint': '{}.pt'.format(exp_name),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_metrics': metrics_dict,
    }
    
    return results