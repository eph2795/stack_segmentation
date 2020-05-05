from tqdm import tqdm
import numpy as np

import torch

from .metrics import softmax
from .dataloader.dataloader import make_dataloader, collate_fn_basic


def apply(
        stack,
        model,
        model_config,
        patch_sizes,
        bs=1,
        num_workers=16,
        device='cpu',
        threshold=0.5
):
    data = stack.slice_up(patch_sizes=patch_sizes)
    dataloader = make_dataloader(samples=data,
                                 collate_fn=collate_fn_basic,
                                 model_config=model_config,
                                 aug_config=None,
                                 batch_size=bs,
                                 shuffle=False,
                                 patch_sizes=None,
                                 num_workers=num_workers)
    model.eval()
    with torch.no_grad():
        offset = 0
        for item in tqdm(dataloader, mininterval=10, maxinterval=20):

            def handle_batch():
                if isinstance(item, tuple):
                    x, _ = item
                else:
                    x = item
                logit = model(torch.from_numpy(x).to(device)).cpu().data.numpy()
                probs = softmax(logit)
                if threshold is None:
                    preds = probs[:, 1]
                else:
                    preds = (probs[:, 1] > threshold).astype(np.uint8)
                return preds

            preds = handle_batch()
            for i, pred in enumerate(preds):
                data[offset + i]['preds'] = pred.reshape(patch_sizes)
            offset += preds.shape[0]

    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    return stack.assembly(stack.H, stack.W, stack.D, data)
