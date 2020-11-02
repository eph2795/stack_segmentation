import os

from tqdm import tqdm
import numpy as np

import imageio
from skimage.transform import downscale_local_mean


def downscale(stack, scale=6):
    downscaled_2d = np.zeros(
        shape=(stack.shape[0] // scale, stack.shape[1] // scale, stack.shape[2]),
        dtype=np.uint8
    )
    for i in tqdm(range(stack.shape[2])):
        image = stack[:, :, i]
        image_downscaled = downscale_local_mean(image, (scale, scale))
        downscaled_2d[:, :, i] = image_downscaled
    downscaled = np.zeros(
        shape=(stack.shape[0] // scale, stack.shape[1] // scale, stack.shape[2] // scale),
        dtype=np.uint8
    )
    for i in tqdm(range(downscaled_2d.shape[0])):
        image = downscaled_2d[i]
        image_downscaled = downscale_local_mean(image, (1, scale))
        downscaled[i] = image_downscaled
    return downscaled


def basic_loader(img_path):
    image = imageio.imread(img_path)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    return image


def compress_3d(image: np.array, factor: int):
    indices = np.arange(0, image.shape[0], factor)
    first = np.add.reduceat(image, indices=indices, axis=0)
    second = np.add.reduceat(first, indices=indices, axis=1)
    third = np.add.reduceat(second, indices=indices, axis=2)
    output = (third / (factor ** 3)).astype(np.float32)
    return output


def process_gt(image: np.array, factor: int, num_classes: int = 5):
    class_images = []
    for i in range(num_classes):
        mask = (image == i).astype(np.float32)
        if factor != 1:
            processed = compress_3d(mask, factor=factor)
        else:
            processed = mask
        class_images.append(processed[:, :, :, np.newaxis])
    return np.concatenate(class_images, axis=3)


def synthetic_to_common_stack(
        dataset_path: str,
        synthetic_stack_path: str,
        recon_name: str,
        recon_size: int,
        original_path: str,
        original_size: int,
        pad_size: int,
        num_classes: int = 5
) -> str:
    _, synthetic_stack_name = os.path.split(synthetic_stack_path)
    true_data = (
        np
        .fromfile(original_path,dtype=np.uint8)
        .reshape((original_size, original_size, original_size))
    )
    recon_data_padded = np.fromfile(
        os.path.join(synthetic_stack_path, recon_name).format(recon_size=recon_size),
        dtype=np.uint8
    ).reshape((recon_size, recon_size, recon_size))

    recon_data = recon_data_padded[pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size]

    factor = int(true_data.shape[0] / recon_data.shape[0])
    recon_unpadded_size = recon_size - 2 * pad_size
    true_labels = process_gt(true_data, factor=factor, num_classes=num_classes)

    stack_path = os.path.join(dataset_path, '{}_{}_{}'
                              .format(synthetic_stack_name, recon_unpadded_size, num_classes))
    gt_path = os.path.join(stack_path, 'GT')
    os.makedirs(gt_path, exist_ok=True)
    for i, slice_ in tqdm(enumerate(true_labels)):
        slice_.tofile(os.path.join(gt_path, '{:05}.bin'.format(i)))
    data_path = os.path.join(stack_path, 'DATA')
    os.makedirs(data_path, exist_ok=True)
    for i, slice_ in tqdm(enumerate(recon_data)):
        slice_.tofile(os.path.join(data_path, '{:05}.bin'.format(i)))
    return stack_path


def get_patch_index(h, w, d, batch_size, x, y, z, dim):
    if dim == 0:
        w = w - batch_size
        d = d - batch_size
        offset = 0
    elif dim == 1:
        offset = h * (w - batch_size) * (d - batch_size)
        h = h - batch_size
        d = d - batch_size
    elif dim == 2:
        offset = h * (w - batch_size) * (d - batch_size) + (h - batch_size) * w * (d - batch_size)
        h = h - batch_size
        w = w - batch_size
    else:
        raise ValueError("dim param must be 0, 1 or 2")
    if x > h:
        raise ValueError("x > h, {} > {}".format(x, h))
    if y > w:
        raise ValueError("y > w, {} > {}".format(y, w))
    if z > d:
        raise ValueError("z > d, {} > {}".format(z, d))
    return offset + x * w * d + y * d + z


def get_patch_coordinates(h, w, d, batch_size, idx):
    first_dim_size = h * (w - batch_size) * (d - batch_size)
    second_dim_size = (h - batch_size) * w * (d - batch_size)
    third_dim_size = (h - batch_size) * (w - batch_size) * d

    if idx < first_dim_size:
        w = w - batch_size
        d = d - batch_size
        dim = 0
    elif idx < first_dim_size + second_dim_size:
        h = h - batch_size
        d = d - batch_size
        dim = 1
        idx = idx - first_dim_size
    elif idx < first_dim_size + second_dim_size + third_dim_size:
        h = h - batch_size
        w = w - batch_size
        dim = 2
        idx = idx - first_dim_size - second_dim_size
    else:
        raise ValueError('idx > total number of patches')
    x = idx // (w * d)
    residual = idx % (w * d)
    y = residual // d
    z = idx % d
    return x, y, z, dim
