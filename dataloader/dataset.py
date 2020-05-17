from abc import abstractmethod

import imageio


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
            image_loader = imageio.imread
        if mask_loader is None:
            mask_loader = imageio.imread
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
