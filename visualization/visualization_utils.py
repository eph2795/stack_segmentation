from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def make_df(data: dict, model_name: str, classes_of_interest: list = None) -> pd.DataFrame:
    """
    Construct data frame with metrics from experiment data
    Args:
        data: dict-like data from from train() method
        model_name:
        classes_of_interest: classes for which metrics will be stored in the resulting data frame

    Returns:
        data frame
    """
    if classes_of_interest is None:
        classes_of_interest = [1]
    records = []
    for stack_name, metrics_values in data['test_metrics'].items():
        record = dict()
        for metric_name, metric_batch_values in metrics_values.items():
            metric_class_values = np.nanmean(metric_batch_values, axis=0)
            for cls_number, metric_value in enumerate(metric_class_values):
                if cls_number not in classes_of_interest:
                    continue
                key = '{metric_name}: class {cls_number}'.format(metric_name=metric_name,
                                                                 cls_number=cls_number)
                record[key] = metric_value
        record['stack'] = stack_name
        record['model'] = model_name
        records.append(record)
    return pd.DataFrame.from_records(records)


def to_single_dim_uint8(image):
    return np.argmax(image, axis=2).astype(np.uint8)


def make_colored_diff(
        mask: np.ndarray,
        prediction: np.ndarray,
        cls_number: int = 1,
        path: Union[str, None] = None
) -> np.ndarray:
    """
    Plotting difference between mask and prediction images for one selected class.
    Red color means
    Args:
        mask: image of type uint8 with segmentation mask of [H x W x K] shape
        prediction: predicted mask of type uint8 of [H x W x K] shape
        cls_number: class number to plot difference for
        path: path to save difference figure

    Returns:
        colored_image: image [H x W x 3]
    """
    mask_single_dim = to_single_dim_uint8(mask)
    predicted_single_dim = to_single_dim_uint8(prediction)

    red_mask = np.where((mask_single_dim == cls_number)
                        & (predicted_single_dim != cls_number), 255, 0)
    blue_mask = np.where((mask_single_dim != cls_number)
                         & (predicted_single_dim == cls_number), 255, 0)
    valid_mask = np.where((mask_single_dim == cls_number)
                          & (predicted_single_dim == cls_number), 255, 0)[:, :, np.newaxis]

    colored_image = np.concatenate([valid_mask, valid_mask, valid_mask], axis=2)
    colored_image[:, :, 0] += red_mask
    colored_image[:, :, 2] += blue_mask

    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(colored_image)

    if path is not None:
        plt.savefig(path)
    return colored_image


def fill_image_with_colors(image: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """
    Add one pixel with value in range [0, n_classes)
    Args:
        image: given image
        n_classes: number of classes to add

    Returns:
        image: image with at least one pixel for each values in range [0, n_classes)
    """
    for i in range(n_classes):
        image[i, 0] = i
    return image


def plot_sample(
        image: np.ndarray,
        mask: np.ndarray,
        predicted: np.ndarray,
        metrics: dict,
        n_classes: int = 5,
        fig_path: str = None
) -> None:
    """
    Plot image and mask/model prediction in discrete color map fashion;
    also compute metrics contained in metrics dict

    Args:
        image: source image [H x W x C]
        mask: desired segmentation mask [H х W х K]
        predicted: predicted segmentation mask [H x W x K]
        metrics: dict of metrics to calculate
        n_classes: number of segmentation classes
        fig_path: path to save figure

    Returns:

    """
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
    mask_single_dim = to_single_dim_uint8(mask)
    predicted_single_dim = to_single_dim_uint8(predicted)

    colors = ['k', 'b', 'w', 'g', 'r']
    cmap = ListedColormap(colors[:n_classes])

    plt.figure(figsize=(30, 10))

    plt.subplot(1, 3, 1)
    plt.title('image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('mask')
    plt.imshow(fill_image_with_colors(mask_single_dim, n_classes=n_classes), cmap=cmap)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('prediction')
    plt.imshow(fill_image_with_colors(predicted_single_dim, n_classes=n_classes), cmap=cmap)
    plt.axis('off')

    if fig_path is not None:
        plt.savefig(fig_path)

    plt.show()
    for metric_name, metric_func in metrics.items():
        print('{metric_name:9}: {metric_value}'
              .format(metric_name=metric_name,
                      metric_value=metric_func(mask, predicted)))
