from itertools import product

import numpy as np
from tqdm import tqdm
import imageio

import torch
from torch.utils.data import DataLoader

from segmentation_models_pytorch.encoders import get_preprocessing_fn

from .stack import Stack
from .metrics import softmax
from .aug_pipelines import make_aug


class TomoDataset:
    
    def __init__(
            self,
            samples,
            augmentation_fn=None,
            preprocessing_image_fn=None,
            preprocessing_mask_fn=None
    ):
        self._images = np.concatenate(
            [np.squeeze(sample['features'])[np.newaxis, :, :, np.newaxis]
             for sample in samples],
            axis=0)
        if 'targets' in samples[0]:
            self._masks = np.concatenate(
                [np.squeeze(sample['targets'])[np.newaxis, :, :]
                 for sample in samples],
                axis=0)
        else:
            self._masks = None
        self._len = len(samples)
        self._augmentation_fn = augmentation_fn
        self._preprocessing_image_fn = preprocessing_image_fn
        self._preprocessing_mask_fn = preprocessing_mask_fn
        
    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        image = self._images[idx]
        mask = self._masks[idx] if self._masks is not None else None
        
        if self._augmentation_fn is not None:
            augmented = self._augmentation_fn(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        if self._preprocessing_image_fn is not None:
            image = self._preprocessing_image_fn(image)
        if (mask is not None) and (self._preprocessing_mask_fn is not None):
            mask = self._preprocessing_mask_fn(mask)
            
        result = {
            'features': image,
        }
        if mask is not None:
            result['targets'] = mask
        return result


def get_grids(size, patch_sizes):
    grids = []
    for dim, patch_size in zip(size, patch_sizes):
        grids.append(Stack.get_one_dimensional_grid(dim, patch_size))

    selectors = []
    for x, y, z in product(*grids):
        selector = tuple(slice(x, x + dx, 1) for x, dx in zip([x, y, z], patch_sizes))
        selectors.append(selector)
    return selectors


class TomoLazyDataset:

    def __init__(
            self,
            samples,
            patch_sizes,
            augmentation_fn=None,
            preprocessing_image_fn=None,
            preprocessing_mask_fn=None
    ):
        lens = []
        self.samples = samples
        self.patch_sizes = patch_sizes
        for image, mask in samples:
            img = imageio.imread(image)[:, :, np.newaxis]
            # gt = imageio.imread(mask)[:, :, np.newaxis]
            grids = get_grids(img.shape, self.patch_sizes)
            lens.append(len(grids))
        self.lens = np.cumsum(lens)
        self._len = self.lens[-1]
        self._augmentation_fn = augmentation_fn
        self._preprocessing_image_fn = preprocessing_image_fn
        self._preprocessing_mask_fn = preprocessing_mask_fn

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        img_idx = np.searchsorted(self.lens, idx, side='right')
        image = imageio.imread(self.samples[img_idx][0])[:, :, np.newaxis]
        mask = imageio.imread(self.samples[img_idx][1])[:, :, np.newaxis]
        grids = get_grids(image.shape, self.patch_sizes)
        patch_idx = idx - self.lens[img_idx]
        image = image[grids[patch_idx]]
        mask = np.squeeze(mask[grids[patch_idx]])

        if self._augmentation_fn is not None:
            augmented = self._augmentation_fn(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        if self._preprocessing_image_fn is not None:
            image = self._preprocessing_image_fn(image)
        if (mask is not None) and (self._preprocessing_mask_fn is not None):
            mask = self._preprocessing_mask_fn(mask)

        result = {
            'features': image,
        }
        if mask is not None:
            result['targets'] = mask
        return result


def image_process_basic(image, mean=0.3057127, std=0.13275838):
    scaled = image / 255
    normalized = (scaled - mean) / std
    return normalized


def mask_process_basic(mask):
    binary = np.where(mask == 255, 0, 1)
    return binary


def collate_fn_basic(samples):
    image_samples, gt_samples = [], []
    
    for sample in samples:
        image = sample['features']
        if 'targets' in sample:
            mask = sample['targets']
            gt_samples.append(mask[np.newaxis, :, :])
        image_samples.append(image[np.newaxis].transpose(0, 3, 1, 2))
    if len(gt_samples) == len(image_samples):
        gt_samples = np.concatenate(gt_samples, axis=0).astype(np.int64)
        image_samples = np.concatenate(image_samples, axis=0).astype(np.float32)
        return image_samples, gt_samples
    else:
        image_samples = np.concatenate(image_samples, axis=0).astype(np.float32)
        return image_samples


def make_dataloader(
        samples, 
        collate_fn, 
        model_config,
        patch_sizes,
        aug_config=None, 
        batch_size=32, 
        shuffle=True, 
        num_workers=8,
        dataloader_type='inmemory'
    ):
    if aug_config is not None:
        augmentation_pipeline = make_aug(**aug_config)
    else:
        augmentation_pipeline = None
    
    if model_config['source'] == 'basic':
        preprocessing_image_fn = image_process_basic
    else:
        preprocessing_image_fn = get_preprocessing_fn(encoder_name=model_config['encoder_name'],
                                                      pretrained=model_config['encoder_weights'])

    if dataloader_type == 'inmemory':
        dataset = TomoDataset(samples,
                              augmentation_fn=augmentation_pipeline,
                              preprocessing_image_fn=preprocessing_image_fn,
                              preprocessing_mask_fn=mask_process_basic)
    elif dataloader_type == 'lazy':
        dataset = TomoLazyDataset(
            samples,
            patch_sizes=patch_sizes,
            augmentation_fn=augmentation_pipeline,
            preprocessing_image_fn=preprocessing_image_fn,
            preprocessing_mask_fn=mask_process_basic
        )
    else:
        raise ValueError('Wrong dataset type!')

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader


def apply(
        stack,
        model,
        model_config,
        patch_sizes,
        bs=1,
        num_workers=16,
        device='cpu',
        threshold=0.5
):
    data = stack.slice_up(patch_sizes=patch_sizes)
    dataloader = make_dataloader(samples=data,
                                 collate_fn=collate_fn_basic,
                                 model_config=model_config,
                                 aug_config=None,
                                 batch_size=bs,
                                 shuffle=False,
                                 patch_sizes=None,
                                 num_workers=num_workers)
    model.eval()
    with torch.no_grad():
        offset = 0
        for item in tqdm(dataloader, mininterval=10, maxinterval=20):

            def handle_batch():
                if isinstance(item, tuple):
                    x, _ = item
                else:
                    x = item
                logit = model(torch.from_numpy(x).to(device)).cpu().data.numpy()
                probs = softmax(logit)
                if threshold is None:
                    preds = probs[:, 1]
                else:
                    preds = (probs[:, 1] > threshold).astype(np.uint8)
                return preds

            preds = handle_batch()
            for i, pred in enumerate(preds):
                data[offset + i]['preds'] = pred.reshape(patch_sizes)
            offset += preds.shape[0]

    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    return stack.assembly(stack.H, stack.W, stack.D, data)
