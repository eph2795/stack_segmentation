from itertools import product

import numpy as np
import imageio

from ..stack import Stack


def get_grids(size, patch_sizes):
    grids = []
    for dim, patch_size in zip(size, patch_sizes):
        grids.append(Stack.get_one_dimensional_grid(dim, patch_size))

    selectors = []
    for x, y in product(*grids):
        selector = tuple(slice(x, x + dx) for x, dx in zip([x, y], patch_sizes))
        selectors.append(selector)
    return selectors


class ImageDataset:

    def __init__(
            self,
            samples,
            patch_sizes,
            image_reading_fn=None,
            gt_reading_fn=None,
            augmentation_fn=None,
            preprocessing_image_fn=None,
            preprocessing_mask_fn=None
    ):
        if image_reading_fn is None:
            image_reading_fn = imageio.imread
        self._image_reading_fn = image_reading_fn
        if gt_reading_fn is None:
            gt_reading_fn = imageio.imread
        self._gt_reading_fn = gt_reading_fn
        self.samples = samples
        self.patch_sizes = patch_sizes
        img_path, _ = samples[0]
        img = self._image_reading_fn(img_path)
        self._grids = get_grids(img.shape, self.patch_sizes)
        lens = [len(self._grids) for _ in samples]
        self.lens = np.cumsum(lens)
        self._len = self.lens[-1]
        self._augmentation_fn = augmentation_fn
        self._preprocessing_image_fn = preprocessing_image_fn
        self._preprocessing_mask_fn = preprocessing_mask_fn

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        img_idx = np.searchsorted(self.lens, idx, side='right')
        image = self._image_reading_fn(self.samples[img_idx][0])
        mask = self._gt_reading_fn(self.samples[img_idx][1])
        patch_idx = idx - self.lens[img_idx]
        image = image[self._grids[patch_idx]]
        mask = mask[self._grids[patch_idx]]

        if self._augmentation_fn is not None:
            augmented = self._augmentation_fn(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        if self._preprocessing_image_fn is not None:
            image = self._preprocessing_image_fn(image)
        if (mask is not None) and (self._preprocessing_mask_fn is not None):
            mask = self._preprocessing_mask_fn(mask)

        result = {
            'features': image.astype(np.float32),
        }
        if mask is not None:
            result['targets'] = mask
        return result
