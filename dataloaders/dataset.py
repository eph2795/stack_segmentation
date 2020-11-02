from abc import abstractmethod

import numpy as np

from ..utils.utils import basic_loader, get_patch_coordinates


class CustomDataset:

    def __init__(
            self,
            image_loader=None,
            mask_loader=None,
            augmentation_fn=None,
            preprocessing_image_fn=None,
            preprocessing_mask_fn=None
    ):
        if image_loader is None:
            image_loader = basic_loader
        if mask_loader is None:
            mask_loader = basic_loader
        self._image_loader = image_loader
        self._mask_loader = mask_loader
        self._augmentation_fn = augmentation_fn
        self._preprocessing_image_fn = preprocessing_image_fn
        self._preprocessing_mask_fn = preprocessing_mask_fn

    @abstractmethod
    def _get_image(self, idx):
        pass

    @abstractmethod
    def _get_mask(self, idx):
        pass

    def _augment_sample(self, image, mask):
        if self._augmentation_fn is not None:
            augmented = self._augmentation_fn(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        return image, mask

    def _preprocess_image(self, image):
        if self._preprocessing_image_fn is not None:
            image = self._preprocessing_image_fn(image)
        return image

    def _preprocess_mask(self, mask):
        if self._preprocessing_mask_fn is not None:
            mask = self._preprocessing_mask_fn(mask)
        return mask

    def __getitem__(self, idx, use_augmentation=True, use_preprocessing=True):
        image = self._get_image(idx)
        mask = self._get_mask(idx)

        if use_augmentation:
            image, mask = self._augment_sample(image, mask)

        if use_preprocessing:
            image = self._preprocess_image(image)
            mask = self._preprocess_mask(mask)

        result = {
            'image': image
        }
        if mask is not None:
            result['mask'] = mask
        return result


class PatchDataset:

    def __init__(
            self,
            data,
            ground_truth,
            patch_size,
            augmentation_fn=None,
            preprocessing_image_fn=None,
            preprocessing_mask_fn=None,
            postprocessing_image_fn=None,
            postprocessing_mask_fn=None
    ):
        self.data = data
        self.ground_truth = ground_truth

        self.patch_size = patch_size
        self.augmentation_fn = augmentation_fn
        self.preprocessing_image_fn = preprocessing_image_fn
        self.preprocessing_mask_fn = preprocessing_mask_fn
        self.postprocessing_image_fn = postprocessing_image_fn
        self.postprocessing_mask_fn = postprocessing_mask_fn

        self.h, self.w, self.d = data.shape
        self._len = (
                self.get_dim_size(self.h, self.w, self.d, patch_size, dim=0)
                + self.get_dim_size(self.h, self.w, self.d, patch_size, dim=1)
                + self.get_dim_size(self.h, self.w, self.d, patch_size, dim=2)
        )

    @staticmethod
    def get_dim_size(h, w, d, patch_size, dim):
        if h < patch_size:
            raise ValueError('h < patch_size')
        if w < patch_size:
            raise ValueError('w < patch_size')
        if d < patch_size:
            raise ValueError('d < patch_size')

        if dim == 0:
            w = w - patch_size
            d = d - patch_size
        elif dim == 1:
            h = h - patch_size
            d = d - patch_size
        elif dim == 2:
            h = h - patch_size
            w = w - patch_size
        else:
            raise ValueError('dim value must be 0, 1 or 2')
        return h * w * d

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        x, y, z, dim = get_patch_coordinates(self.h, self.w, self.d, self.patch_size, idx)
        selector_x = slice(x, x + self.patch_size)
        selector_y = slice(y, y + self.patch_size)
        selector_z = slice(z, z + self.patch_size)
        if dim == 0:
            selector_x = slice(x, x + 1)
        elif dim == 1:
            selector_y = slice(y, y + 1)
        elif dim == 2:
            selector_z = slice(z, z + 1)

        data_patch = np.squeeze(self.data[selector_x, selector_y, selector_z])
        gt_patch = np.squeeze(self.ground_truth[selector_x, selector_y, selector_z])
        if self.preprocessing_image_fn is not None:
            data_patch = self.preprocessing_image_fn(data_patch)
        if self.preprocessing_mask_fn is not None:
            gt_patch = self.preprocessing_mask_fn(gt_patch)
        if self.augmentation_fn is not None:
            augmented = self.augmentation_fn(image=data_patch, mask=gt_patch)
            data_patch = augmented['image']
            gt_patch = augmented['mask']
        if self.postprocessing_image_fn is not None:
            data_patch = self.postprocessing_image_fn(data_patch)
        if self.postprocessing_mask_fn is not None:
            gt_patch = self.postprocessing_mask_fn(gt_patch)

        return data_patch, gt_patch
