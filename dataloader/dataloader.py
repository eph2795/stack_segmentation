import numpy as np

from torch.utils.data import DataLoader

from segmentation_models_pytorch.encoders import get_preprocessing_fn

from ..aug_pipelines import make_aug
from .image_dataset import ImageDataset
from .patch_dataset import PatchDataset


def image_process_basic(image, mean=0.3057127, std=0.13275838):
    normalized = ((image / 255 - mean) / std).astype(np.float32)
    return normalized


def mask_process_basic(mask):
    binary = np.where(mask == 255, 0, 1).astype(np.int64)
    return binary


def collate_fn_basic(samples):
    image_samples, gt_samples = [], []

    for sample in samples:
        image = sample['features']
        if 'targets' in sample:
            mask = sample['targets']
            if len(mask.shape) == 3:
                mask = mask.transpose(2, 0, 1)
            gt_samples.append(mask[np.newaxis])
        image_samples.append(image[np.newaxis].transpose(0, 3, 1, 2))
    image_samples = np.concatenate(image_samples, axis=0)
    if len(gt_samples) == len(image_samples):
        gt_samples = np.concatenate(gt_samples, axis=0)
        return image_samples, gt_samples
    else:
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
        dataloader_type='inmemory',
        preprocessing_mask_fn=None,
        image_reading_fn=None,
        gt_reading_fn=None
):
    if aug_config is not None:
        augmentation_pipeline = make_aug(**aug_config)
    else:
        augmentation_pipeline = None

    if model_config['source'] == 'basic':
        preprocessing_image_fn = image_process_basic
    else:
        preprocessing_image_fn = get_preprocessing_fn(
            encoder_name=model_config['params']['encoder_name'],
            pretrained=model_config['params']['encoder_weights']
        )

    if dataloader_type == 'inmemory':
        dataset = PatchDataset(
            samples,
            augmentation_fn=augmentation_pipeline,
            preprocessing_image_fn=preprocessing_image_fn,
            preprocessing_mask_fn=preprocessing_mask_fn
        )
    elif dataloader_type == 'lazy':
        dataset = ImageDataset(
            samples,
            patch_sizes=patch_sizes,
            image_reading_fn=image_reading_fn,
            gt_reading_fn=gt_reading_fn,
            augmentation_fn=augmentation_pipeline,
            preprocessing_image_fn=preprocessing_image_fn,
            preprocessing_mask_fn=preprocessing_mask_fn
        )
    else:
        raise ValueError('Wrong dataset type!')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader
