import sys
sys.path.append('../..')

from copy import deepcopy
from itertools import chain
import pickle

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from stack_segmentation.stack import Stack
from stack_segmentation.io import make_dataloader, collate_fn_basic
from stack_segmentation.training import (
#     handle_stacks_data,
    handle_image_data,
    make_optimization_task,
    train_loop
)

from pipeline_config import (
    dataloaders_conf,
    aug_config,
    train_conf,
    model_config,
    optimizer_config,
    loss_config,
    scheduler_config,
)

from exp_config import data_conf
from stack_segmentation.io import apply
from stack_segmentation.metrics import iou


def moving_average(a, n=5) :
    ret = np.cumsum([a[0]] * (n - 1) + a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def main():
    for i, stack_conf in enumerate(data_conf['stacks']):
        current_data_conf = deepcopy(data_conf)
        stack_name = stack_conf['path'].split('/')[-1]
        current_data_conf['conf_name'] = stack_name
        current_data_conf['stacks'] = data_conf['stacks'][:i] + data_conf['stacks'][i + 1:]
        data_train, data_val, data_test = handle_image_data(**current_data_conf)
        print('Images for training: {}\nImages for validation: {}'.format(len(data_train),
                                                                          len(data_val)))

        dataloader_train = make_dataloader(
            samples=data_train,
            collate_fn=collate_fn_basic,
            model_config=model_config,
            **dataloaders_conf['train'],
            patch_sizes=data_conf['patches']['train'],
            dataloader_type='lazy'
        )

        dataloader_val = make_dataloader(
            samples=data_val,
            collate_fn=collate_fn_basic,
            model_config=model_config,
            **dataloaders_conf['val'],
            patch_sizes=data_conf['patches']['val'],
            dataloader_type='lazy'
        )

        device = train_conf['device']
        model, criterion, optimizer, scheduler = make_optimization_task(
            device,
            model_config=model_config,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config
        )

        results = train_loop(
            model=model,
            dataloader_train=dataloader_train,
            dataloader_val=dataloader_val,
            dataloaders_test=None,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=None,
            exp_name=current_data_conf['conf_name'],
            **train_conf
        )
        p = './{}_exp_results.pkl'.format(current_data_conf['conf_name'])
        with open(p, 'wb') as f:
            pickle.dump(results, f)

        train_losses = list(chain(*[item for item in results['train_losses']]))
        val_losses = list(chain(*[item for item in results['val_losses']]))
        mean_train_loss = [np.mean(item) for item in results['train_losses']]
        mean_val_loss = [np.mean(item) for item in results['val_losses']]
        plt.figure(figsize=(10, 10))
        plt.title('Moving-averaged batch losses {}'.format(stack_name))
        plt.plot(np.arange(len(train_losses)), moving_average(train_losses), label='train')
        plt.plot(np.arange(len(val_losses)), moving_average(val_losses), label='validation')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.show()
        plt.figure(figsize=(10, 10))
        plt.title('Epoch losses {}'.format(stack_name))
        plt.plot(np.arange(len(mean_train_loss)) + 1, mean_train_loss, label='train')
        plt.plot(np.arange(len(mean_val_loss)) + 1, mean_val_loss, label='val')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.xlim([1, len(mean_train_loss) + 1])
        plt.show()

        stack = Stack.read_from_source('../../soil_data/{}'.format(stack_name))

        predicted_stack = apply(
            stack,
            model,
            model_config,
            patch_sizes=(128, 128, 1),
            bs=32, num_workers=8, device=device,
            threshold=None)

        current_iou = iou(
            np.where(predicted_stack_1.targets == 255, 0, 1).astype(np.uint8),
            predicted_stack_1.preds,
            mode='stack',
            threshold=0.5
        )
        print('IOU for {}: {}'.format(stack_name, current_iou))
        predicted_stack.dump('../segmented_stacks/{}'.format(stack_name),
                             features=False,
                             targets=True,
                             preds=True)


if __name__ == '__main__':
    main()
