import os
from itertools import product
import collections

import numpy as np
import imageio
from tqdm import tqdm

from catalyst.dl.callbacks import InferCallback    

import torch
import torchvision.transforms as transforms
from catalyst.data.augmentor import Augmentor
from catalyst.dl.utils import UtilsFactory

# TODO: пофиксить ненужную зависимость, тк предсказываю для стека внутри себя
from .io import collate_fn_basic, make_dataloader


class Stack:
    
    image_subfolder = 'NLM'
    groundtruth_subfolder = 'CAC'
    
    data_types = ['features', 'targets', 'logits']
    
    def __init__(self, features, targets=None, logits=None):
        self.H, self.W, self.D = features.shape
        self.features = features
        self.targets = targets
        self.logits = logits
    
    @property
    def shape(self):
        return self.features.shape 
    
    @classmethod
    def assembly(cls, H, W, D, patches):
        kwargs = dict()
        for data_type in cls.data_types:
            if data_type in patches[0]:
                kwargs[data_type] = np.zeros((H, W, D), dtype=np.float32)
        
        for patch in tqdm(patches):
            selector = tuple(slice(x, x + dx, 1) for x, dx in zip(patch['coordinates'], 
                                                                  patch['features'].shape))
            for data_type, data in patch.items():
                if data_type in cls.data_types:
                    kwargs[data_type][selector] = data
        
        return cls(**kwargs)
        
        
    @classmethod
    def read_from_source(cls, folder_name, has_targets=True):
        folder_paths = [os.path.join(folder_name, cls.image_subfolder), 
                        os.path.join(folder_name, cls.groundtruth_subfolder)]
        if not has_targets:
            folder_paths = folder_paths[:1]
        files_iterator = enumerate(zip(*(sorted(os.listdir(fpath)) for fpath in folder_paths)))            
       
        samples = []
        for i, paths in tqdm(files_iterator):
            sample = dict()
            for folder_path, file_name, im_type in zip(folder_paths, paths, ['features', 'targets']):
                image = imageio.imread(os.path.join(folder_path, file_name))
                sample[im_type] = image[:, :, np.newaxis] 
            sample['coordinates'] = [0, 0, i] 
            samples.append(sample)
            
        if len(samples) == 0:
            raise Exception('Empty stack!')
        D = len(samples)
        H, W = image.shape
            
        return cls.assembly(H, W, D, samples) 
    
    
    def __getitem__(self, selector):
        if (not isinstance(selector, slice) 
                and not ((isinstance(selector, tuple) 
                          and all([isinstance(subarg, slice) or isinstance(subarg, int) for subarg in selector])))):
            raise ValueError('Wrong slicing type! Must be slice or tuple, got {}'.format(type(selector)))
        
        kwargs = dict()
        for data_type in self.data_types:
            arg = getattr(self, data_type, None)
            if arg is not None:
                kwargs[data_type] = arg[selector].copy()
        return self.__class__(**kwargs)
        
    def _get_one_dimensional_grid(self, size, patch_size):
        patch_num = size // patch_size + (size % patch_size != 0)
        total_overlap_size = patch_size * patch_num - size
        max_overlap_size = total_overlap_size // (patch_num - 1) + (total_overlap_size % (patch_num - 1) != 0)
        k = total_overlap_size - (patch_num - 1) * (max_overlap_size - 1)
        grid = np.cumsum([0] 
                         + [patch_size - max_overlap_size] * k 
                         + [patch_size - max_overlap_size + 1] * (patch_num - k - 1))
        return grid
    
    def slice_up(self, patch_sizes):
        grids = []
        for dim, patch_size in zip([self.H, self.W, self.D], patch_sizes):
            grids.append(self._get_one_dimensional_grid(dim, patch_size))
                         
        patches = []
        for x, y, z in tqdm(product(*grids)):
            patch = {
                'coordinates': [x, y, z],
            }
            selector = tuple(slice(x, x + dx, 1) for x, dx in zip([x, y, z], patch_sizes))
            for data_type in self.data_types:             
                if getattr(self, data_type) is not None:
                    patch[data_type] = getattr(self, data_type)[selector]
            patches.append(patch)
        return patches
    
    def apply(self, model, patch_sizes, bs=1, num_workers=16, device='cpu'):
        data = self.slice_up(patch_sizes=patch_sizes)
        
        dataloader = make_dataloader(samples=data, 
                                     collate_fn=collate_fn_basic,
                                     augmentation_pipeline=None,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=num_workers)
        
        

        for i, (x, _) in enumerate(dataloader):
            logit = model(torch.from_numpy(x).to(device)).cpu().data.numpy()
            logit = logit - logit.max(axis=1, keepdims=True)
            exp = np.exp(logit)
            probs = exp / exp.sum(axis=1, keepdims=True)
            pred = np.argmax(probs, axis=1)
            data[i]['logits'] = pred.reshape(patch_sizes)

        return self.assembly(self.H, self.W, self.D, data)
    
    def measure(self, metric, threshold=None):
        
        if self.logits is None:
            raise ValueError('There is no prediction for this stack!')
        
        gt = (self.targets / 255).astype(np.int32).flatten()
        if threshold is not None:
            pred = (self.logits > threshold).flatten()
        else:
            pred = self.logits.flatten()
        return metric(gt, pred)