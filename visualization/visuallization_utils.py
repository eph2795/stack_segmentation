import string
from itertools import chain

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def make_df(data, model_name):
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'pr_auc', 'iou']
    data = data['test_metrics']
    records = []
    for s, v in data.items():
        if s in metrics_list:
            continue
        stack_name = s.split('/')[-1]
        record = {k: v[k][-1] for k in metrics_list}
        record['stack'] = stack_name
        record['model'] = model_name
        records.append(record)
    return pd.DataFrame.from_records(records)


def to_single_dim_uint8(image):
    return np.argmax(image, axis=2).astype(np.uint8)


def make_colored_diff(mask, prediction, cls_number=1, path=None):
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


def fill_image_with_colors(image, n_classes=5):
    for i in range(n_classes):
        image[i, 0] = i
    return image


def plot_sample(
        image,
        mask,
        predicted,
        metrics,
        n_classes=5,
        fig_path=None
):
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
    plt.title('pred')
    plt.imshow(fill_image_with_colors(mask_single_dim, n_classes=n_classes), cmap=cmap)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('gt')
    plt.imshow(fill_image_with_colors(predicted_single_dim, n_classes=n_classes), cmap=cmap)
    plt.axis('off')

    if fig_path is not None:
        plt.savefig(fig_path)

    plt.show()
    for metric_name, metric_func in metrics.items():
        print('{metric_name:9}: {metric_value}'
              .format(metric_name=metric_name,
                      metric_value=metric_func(mask, predicted, logits=True, mode=None)))
