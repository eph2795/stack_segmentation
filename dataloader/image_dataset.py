from itertools import product

import numpy as np
import imageio

from .dataset import CustomDataset


def get_one_dimensional_grid(size, patch_size):
    patch_num = size // patch_size + (size % patch_size != 0)
    total_overlap_size = patch_size * patch_num - size
    max_overlap_size = (total_overlap_size // (patch_num - 1)
                        + (total_overlap_size % (patch_num - 1) != 0)) if patch_num > 1 else 0
    k = total_overlap_size - (patch_num - 1) * (max_overlap_size - 1)
    grid = np.cumsum([0]
                     + [patch_size - max_overlap_size] * k
                     + [patch_size - max_overlap_size + 1] * (patch_num - k - 1))
    return grid


def get_grids(image_sizes, patch_sizes):
    grids = []
    for dim, patch_size in zip(image_sizes, patch_sizes):
        grids.append(get_one_dimensional_grid(dim, patch_size))

    selectors = []
    for x, y in product(*grids):
        selector = tuple(slice(x, x + dx) for x, dx in zip([x, y], patch_sizes))
        selectors.append(selector)
    return selectors


class ImageDataset(CustomDataset):

    def __init__(
            self,
            samples,
            lens,
            grids,
            image_loader=None,
            mask_loader=None,
            augmentation_fn=None,
            preprocessing_image_fn=None,
            preprocessing_mask_fn=None
    ):
        super(ImageDataset, self).__init__(
            image_loader=image_loader,
            mask_loader=mask_loader,
            augmentation_fn=augmentation_fn,
            preprocessing_image_fn=preprocessing_image_fn,
            preprocessing_mask_fn=preprocessing_mask_fn)
        self._samples = samples
        self._lens = lens
        self._len = lens[-1]
        self._grids = grids

    @classmethod
    def from_samples(
            cls,
            samples,
            patch_sizes,
            image_loader=None,
            mask_loader=None,
            augmentation_fn=None,
            preprocessing_image_fn=None,
            preprocessing_mask_fn=None
    ):
        if image_loader is None:
            image_loader = imageio.imread
        if mask_loader is None:
            mask_loader = imageio.imread

        img_path, _ = samples[0]
        img = image_loader(img_path)
        grids = get_grids(img.shape, patch_sizes)
        lens = [len(grids) for _ in samples]
        lens = np.cumsum(lens)

        return cls(samples=samples,
                   lens=lens,
                   grids=grids,
                   image_loader=image_loader,
                   mask_loader=mask_loader,
                   augmentation_fn=augmentation_fn,
                   preprocessing_image_fn=preprocessing_image_fn,
                   preprocessing_mask_fn=preprocessing_mask_fn)

    def _get_image(self, idx):
        img_idx = np.searchsorted(self._lens, idx, side='right')
        image = self._image_loader(self._samples[img_idx][0])
        patch_idx = idx - self._lens[img_idx]
        image = image[self._grids[patch_idx]]
        return image

    def _get_mask(self, idx):
        img_idx = np.searchsorted(self._lens, idx, side='right')
        mask = self._mask_loader(self._samples[img_idx][1])
        patch_idx = idx - self._lens[img_idx]
        mask = mask[self._grids[patch_idx]]
        return mask

    def __len__(self):
        return self._len

    # def __getitem__(self, idx, use_augmentation=True, use_preprocessing=True):
    #     results = super().__getitem__(idx, use_augmentation, use_preprocessing)
    #     return results
