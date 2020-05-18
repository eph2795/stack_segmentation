import numpy as np


class PatchDataset:

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
            'features': image.astype(np.float32),
        }
        if mask is not None:
            result['targets'] = mask
        return result
