import os
from itertools import product
import collections

import numpy as np
import imageio
from tqdm import tqdm

import torch
import torchvision.transforms as transforms

# TODO: пофиксить ненужную зависимость, тк предсказываю для стека внутри себя
from .io import collate_fn_basic, make_dataloader
from .metrics import softmax


class Stack:
    
    image_subfolder = 'NLM'
    groundtruth_subfolder = 'CAC'
    
    data_types = ['features', 'targets', 'preds']
    
    def __init__(self, features, targets=None, preds=None):
        self.H, self.W, self.D = features.shape
        self.features = features
        self.targets = targets
        self.preds = preds
    
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
    
    def apply(self, model, model_config, patch_sizes, bs=1, num_workers=16, device='cpu', threshold=0.5):
        data = self.slice_up(patch_sizes=patch_sizes)
        
        dataloader = make_dataloader(samples=data, 
                                     collate_fn=collate_fn_basic,
                                     model_config=model_config,
                                     aug_config=None,
                                     batch_size=bs,
                                     shuffle=False,
                                     num_workers=num_workers)
        
        

        offset = 0
        for (x, _) in tqdm(dataloader):
            logit = model(torch.from_numpy(x).to(device)).cpu().data.numpy()
            probs = softmax(logit)
            if threshold is None:
                preds = probs[:, 1]
            else:
                preds = (probs[:, 1] > threshold).astype(np.uint8)
            
            for i, pred in enumerate(preds):
                data[i + offset]['preds'] = pred.reshape(patch_sizes)
            offset += preds.shape[0]
            
        return self.assembly(self.H, self.W, self.D, data)
    
    def dump(self, dump_directory, features=False, targets=False, preds=True, threshold=0.5):
        if not os.path.exists(dump_directory):
            os.mkdir(dump_directory)

        features_path = os.path.join(dump_directory, 'features')
        if features:
            if not os.path.exists(features_path):
                os.mkdir(features_path)

        targets_path = os.path.join(dump_directory, 'targets')
        if targets:
            if not os.path.exists(targets_path):
                os.mkdir(targets_path)

        preds_path = os.path.join(dump_directory, 'preds')
        if preds:
            if not os.path.exists(preds_path):
                os.mkdir(preds_path)

        for i in tqdm(range(self.features.shape[2])):
            if features:
                imageio.imwrite(os.path.join(features_path, 'feats{:04}.bmp'.format(i)), 
                                self.features[:, :, i].astype(np.uint8))
            if targets:
                imageio.imwrite(os.path.join(targets_path, 'targets{:04}.bmp'.format(i)), 
                                self.targets[:, :, i].astype(np.uint8))
            if preds:
                pred = self.preds[:, :, i]
                if np.issubdtype(pred.dtype, np.floating):
                    pred = (pred > threshold)
                pred = np.where(pred, 0, 255).astype(np.uint8)
                imageio.imwrite(os.path.join(preds_path, 'preds{:04}.bmp'.format(i)), 
                                pred)
#     def measure(self, metric, threshold=None):
        
#         if self.preds is None:
#             raise ValueError('There is no prediction for this stack!')
        
#         gt = (self.targets / 255).astype(np.uint8).flatten()
#         if threshold is not None:
#             pred = (self.preds > threshold).flatten()
#         else:
#             pred = self.preds.flatten()
#         return metric(gt, pred)