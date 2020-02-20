import numpy as np

from torch.utils.data import DataLoader

from segmentation_models_pytorch.encoders import get_preprocessing_fn

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
        if 'mask' in samples[0]:
            self._masks = np.concatenate(
                [np.squeeze(sample['mask'])[np.newaxis, :, :]
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
        aug_config=None, 
        batch_size=32, 
        shuffle=True, 
        num_workers=8
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
        
    dataset = TomoDataset(samples, 
                          augmentation_fn=augmentation_pipeline,
                          preprocessing_image_fn=preprocessing_image_fn,
                          preprocessing_mask_fn=mask_process_basic)
    
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader
