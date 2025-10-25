# Correct File Location: ChangeStar/data/custom/custom_cd_loader.py

import os
import glob
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset, ConcatDataset, SequentialSampler
from torch.utils.data.dataloader import default_collate

import ever as er


def safe_collate(batch):
    """
    A custom collate function that filters out None values.
    This is important for handling cases where a file might be missing or corrupt.
    """
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return None
    return default_collate(batch)


class CustomChangeDetectionDataset(Dataset):
    """
    A robust dataset class for your custom data.
    It finds all images in the 'A' directory and constructs paths for 'B' and 'label'
    in a more reliable way than simple string replacement.
    """

    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        path_a = os.path.join(root_dir, 'A')
        path_b = os.path.join(root_dir, 'B')
        path_label = os.path.join(root_dir, 'label')

        self.A_image_fps = sorted(glob.glob(os.path.join(path_a, '*.png')))

        self.B_image_fps = [os.path.join(path_b, os.path.basename(fp)) for fp in self.A_image_fps]
        self.gt_fps = [os.path.join(path_label, os.path.basename(fp)) for fp in self.A_image_fps]

    def __getitem__(self, idx):
        try:
            img1 = imread(self.A_image_fps[idx])
            img2 = imread(self.B_image_fps[idx])
            gt = imread(self.gt_fps[idx])
        except FileNotFoundError:
            # If a file is missing (e.g., moved to val set), return None
            return None, None

        imgs = np.concatenate([img1, img2], axis=2)
        if self.transforms:
            blob = self.transforms(**dict(image=imgs, mask=gt))
            imgs = blob['image']
            gt = blob['mask']

        return imgs, dict(change=gt, image_filename=os.path.basename(self.A_image_fps[idx]))

    def __len__(self):
        return len(self.A_image_fps)


@er.registry.DATALOADER.register()
class CustomCDLoader(er.ERDataLoader):
    def __init__(self, config):
        super(CustomCDLoader, self).__init__(config)

    @property
    def dataloader_params(self):
        if any([isinstance(self.config.root_dir, tuple),
                isinstance(self.config.root_dir, list)]):
            dataset_list = [CustomChangeDetectionDataset(im_dir, self.config.transforms) for im_dir in
                            self.config.root_dir]
            dataset = ConcatDataset(dataset_list)
        else:
            dataset = CustomChangeDetectionDataset(self.config.root_dir, self.config.transforms)

        sampler = er.data.StepDistributedSampler(dataset) if self.config.training else SequentialSampler(dataset)

        return dict(dataset=dataset,
                    batch_size=self.config.batch_size,
                    sampler=sampler,
                    num_workers=self.config.num_workers,
                    pin_memory=True,
                    drop_last=False,
                    # FIX: Use the safe_collate function to handle missing files
                    collate_fn=safe_collate)

    def set_default_config(self):
        self.config.update(dict(
            root_dir='',
            transforms=None,
            batch_size=1,
            num_workers=0,
            training=False
        ))
