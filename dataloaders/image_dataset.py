from typing import Tuple, List, Callable, Union

from itertools import product

import numpy as np

from .dataset import CustomDataset
from ..utils.utils import basic_loader


def get_one_dimensional_grid(size: int, patch_size: int) -> np.ndarray:
    """
    Get grid of patches with fixed size that cover given size
    Args:
        size: size to cover with patches
        patch_size:

    Returns:
        grid: array of left borders for patches

    Examples:
        >>> print(get_one_dimensional_grid(10, 3)
        np.array([0, 2, 4, 7])

        >>> get_one_dimensional_grid(11, 3)
        np.array([0, 2, 5, 8])
    """

    patch_num = size // patch_size + (size % patch_size != 0)
    overlaps_number = patch_num - 1
    total_overlap_size = patch_size * patch_num - size
    max_overlap_size = (total_overlap_size // overlaps_number
                        + (total_overlap_size % overlaps_number != 0)) if patch_num > 1 else 0
    # Total number of patches which has "maximum - 1" overlap with its neighbors
    k = total_overlap_size - overlaps_number * (max_overlap_size - 1)
    grid = np.cumsum([0]
                     + [patch_size - max_overlap_size] * k
                     + [patch_size - max_overlap_size + 1] * (overlaps_number - k))
    return grid


def get_grids(image_sizes: Tuple[int, int, int], patch_sizes: Tuple[int, int, int]) -> List[Tuple]:
    """
    Create patches grid that cover each pixel of image.
    Number of image dimensions must be equal with number of patch dimensions.

    Args:
        image_sizes: size of the image, could be any number
            of dimensions  ([H x W], [H x W x D], etc.)
        patch_sizes: size of patch, for example [h x w]

    Returns:
        selectors: list of selectors for required patches
    """
    grids = []
    for dim, patch_size in zip(image_sizes, patch_sizes):
        grids.append(get_one_dimensional_grid(dim, patch_size))

    selectors = []
    for patch_coordinates in product(*grids):
        selector = tuple(slice(x, x + dx) for x, dx in zip(patch_coordinates, patch_sizes))
        selectors.append(selector)
    return selectors


class ImageDataset(CustomDataset):
    """
    Dataset class that could take patch of fixes size from any image, also
    supports strict indexing.
    """

    def __init__(
            self,
            samples: List[Tuple[str, str]],
            lens: List[int],
            grids: List[Tuple],
            image_loader: Union[Callable, None] = None,
            mask_loader: Union[Callable, None] = None,
            augmentation_fn: Union[Callable, None] = None,
            preprocessing_image_fn: Union[Callable, None] = None,
            preprocessing_mask_fn: Union[Callable, None] = None
    ) -> None:
        """

        Args:
            samples: list of image-mask paths pairs
            lens: list of number of patches for each image-mask
            grids: list of grids for each image-mask
            image_loader: callable that load image from given path
            mask_loader: callable that load mask from given path
            augmentation_fn:
            preprocessing_image_fn:
            preprocessing_mask_fn:
        """
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
            samples: List[Tuple[str, str]],
            patch_sizes: Tuple[int, int, int],
            image_loader: Union[Callable, None] = None,
            mask_loader: Union[Callable, None] = None,
            augmentation_fn: Union[Callable, None] = None,
            preprocessing_image_fn: Union[Callable, None] = None,
            preprocessing_mask_fn: Union[Callable, None] = None
    ):
        """
        Construct ImageDataset object from number of samples, grid size and utility functions
        Args:
            samples: list of image-mask paths pairs
            patch_sizes: size of patch to split samples by
            image_loader: callable that load image from given path
            mask_loader: callable that load mask from given path
            augmentation_fn:
            preprocessing_image_fn:
            preprocessing_mask_fn:

        Returns:
            ImageDataset instance
        """

        if image_loader is None:
            image_loader = basic_loader
        if mask_loader is None:
            mask_loader = basic_loader

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

    def _get_image(self, idx: int) -> np.ndarray:
        """
        Get image sample from index

        Args:
            idx: integer in range [0, len(self)]

        Returns:
            image: numpy array with sizes [H, W, C]
        """

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
