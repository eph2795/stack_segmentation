import numpy as np


def image_process_basic(image, mean=0.3057127, std=0.13275838):
    normalized = ((image / 255 - mean) / std).astype(np.float32)
    return normalized


def mask_process_basic(mask):
    binary = np.where(mask == 255, 0, 1).astype(np.int64)
    return binary


def expand_grayscale_image(image):
    image = np.copy(image)[:, :, np.newaxis]
    return np.repeat(image, repeats=3, axis=-1)


def get_ohe_mask_processing(num_classes, dtype):
    def ohe_mask_processing(mask):
        output = np.zeros(shape=(*mask.shape, num_classes), dtype=dtype)
        i = np.arange(mask.shape[0]).reshape(-1, 1)
        j = np.arange(mask.shape[1]).reshape(1, -1)
        output[i, j, mask] = 1
        return output
    return ohe_mask_processing
