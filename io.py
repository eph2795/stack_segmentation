import numpy as np

from torch.utils.data import DataLoader


class TomoDataset:
    
    def __init__(self, samples):
        self.samples = samples
        self.len = len(samples)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        samples = self.samples[idx]
        return {k: samples[k] for k in ('features', 'targets')}
    
    
def image_process_basic(image, mean=0.3057127, std=0.13275838):
    scaled = image / 255
    normalized = (scaled - mean) / std
    return normalized


def mask_process_basic(mask):
    binary = np.where(mask == 255, 1, 0)
    return binary
    

def collate_fn_basic(samples, augmentation_pipeline):
    image_samples, gt_samples = [], []
    
    for sample in samples:
        image = np.squeeze(sample['features'])
        mask = np.squeeze(sample['targets'])
        
        if augmentation_pipeline is not None:
            augmented = augmentation_pipeline(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        image = image_process_basic(image)
        mask = mask_process_basic(mask)
        image_samples.append(image[np.newaxis, np.newaxis, :, :])
        gt_samples.append(mask[np.newaxis, :, :])
    
    image_samples = np.concatenate(image_samples, axis=0).astype(np.float32)
    gt_samples = np.concatenate(gt_samples, axis=0).astype(np.int64)
    return image_samples, gt_samples


def make_dataloader(samples, collate_fn, augmentation_pipeline=None, batch_size=32, shuffle=True, num_workers=8):
    dataset = TomoDataset(samples)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            collate_fn=lambda batch: collate_fn(batch, augmentation_pipeline),
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader
