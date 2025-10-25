import glob
import os

import ever as er
import numpy as np
from albumentations import Compose, Normalize
from skimage.io import imread
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torch.utils.data import SequentialSampler, SubsetRandomSampler


class LEVIRCD(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.A_image_fps = glob.glob(os.path.join(root_dir, 'A', '*.png'))
        self.B_image_fps = [fp.replace('A', 'B') for fp in self.A_image_fps]
        self.gt_fps = [fp.replace('A', 'label') for fp in self.A_image_fps]
        self.transforms = transforms

    def __getitem__(self, idx):
        img1 = imread(self.A_image_fps[idx])
        img2 = imread(self.B_image_fps[idx])
        gt = imread(self.gt_fps[idx])

        imgs = np.concatenate([img1, img2], axis=2)
        if self.transforms:
            blob = self.transforms(**dict(image=imgs, mask=gt))
            imgs = blob['image']
            gt = blob['mask']

        return imgs, dict(change=gt, image_filename=os.path.basename(self.A_image_fps[idx]))

    def __len__(self):
        return len(self.A_image_fps)


@er.registry.DATALOADER.register()
class LEVIRCDLoader(er.ERDataLoader):
    def __init__(self, config):
        super(LEVIRCDLoader, self).__init__(config)

    @property
    def dataloader_params(self):
        if any([isinstance(self.config.root_dir, tuple),
                isinstance(self.config.root_dir, list)]):
            dataset_list = []
            for im_dir in self.config.root_dir:
                dataset_list.append(LEVIRCD(im_dir,
                                            self.config.transforms))

            dataset = ConcatDataset(dataset_list)

        else:
            dataset = LEVIRCD(self.config.root_dir,
                              self.config.transforms)

        if self.config.subsample_ratio < 1.0:
            num_train = len(dataset)
            indices = list(range(num_train))
            split = int(np.floor(self.config.subsample_ratio * num_train))
            sampler = SubsetRandomSampler(indices[:split])
        else:
            sampler = er.data.StepDistributedSampler(dataset) if self.config.training else SequentialSampler(
                dataset)

        return dict(dataset=dataset,
                    batch_size=self.config.batch_size,
                    sampler=sampler,
                    num_workers=self.config.num_workers,
                    pin_memory=True,
                    drop_last=False,
                    timeout=0,
                    worker_init_fn=None)

    def set_default_config(self):
        self.config.update(dict(
            root_dir='',
            transforms=Compose([
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225), max_pixel_value=255),
                er.preprocess.albu.ToTensor(),
            ]),
            subsample_ratio=1.0,
            batch_size=1,
            num_workers=0,
            training=False
        ))

class ChangeDetectionDataset(Dataset):
    """
    A generic change detection dataset class that can be configured
    for different directory names (e.g., 'A'/'B' or 'time1'/'time2').
    """
    def __init__(self, root_dir, transforms=None, t1_dir='A', t2_dir='B', lbl_dir='label'):
        self.root_dir = root_dir
        # Make directory names configurable
        self.t1_dir_name = t1_dir
        self.t2_dir_name = t2_dir
        self.lbl_dir_name = lbl_dir

        # Find all images in the time 1 directory
        self.t1_image_fps = glob.glob(os.path.join(root_dir, self.t1_dir_name, '*.png'))
        # Create paths for time 2 and label images by replacing the directory name
        self.t2_image_fps = [fp.replace(self.t1_dir_name, self.t2_dir_name) for fp in self.t1_image_fps]
        self.gt_fps = [fp.replace(self.t1_dir_name, self.lbl_dir_name) for fp in self.t1_image_fps]
        self.transforms = transforms

    def __getitem__(self, idx):
        img1 = imread(self.t1_image_fps[idx])
        img2 = imread(self.t2_image_fps[idx])
        gt = imread(self.gt_fps[idx])

        imgs = np.concatenate([img1, img2], axis=2)
        if self.transforms:
            blob = self.transforms(**dict(image=imgs, mask=gt))
            imgs = blob['image']
            gt = blob['mask']

        # Match the output format of the original LEVIRCD class
        return imgs, dict(change=gt, image_filename=os.path.basename(self.t1_image_fps[idx]))

    def __len__(self):
        return len(self.t1_image_fps)


# === ADD THIS NEW LOADER FOR SYSU-CD TO dataset.py ===
@er.registry.DATALOADER.register()
class SYSUCDLoader(er.ERDataLoader):
    def __init__(self, config):
        super(SYSUCDLoader, self).__init__(config)

    @property
    def dataloader_params(self):
        # Use our new generic ChangeDetectionDataset
        dataset = ChangeDetectionDataset(self.config.root_dir,
                                         self.config.transforms,
                                         t1_dir=self.config.t1_dir,
                                         t2_dir=self.config.t2_dir,
                                         lbl_dir=self.config.lbl_dir)

        sampler = er.data.StepDistributedSampler(dataset) if self.config.training else SequentialSampler(dataset)

        return dict(dataset=dataset,
                    batch_size=self.config.batch_size,
                    sampler=sampler,
                    num_workers=self.config.num_workers,
                    pin_memory=True,
                    drop_last=False)

    def set_default_config(self):
        # Set default configs, including the directory names for SYSU-CD
        self.config.update(dict(
            root_dir='',
            transforms=Compose([
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225), max_pixel_value=255),
                er.preprocess.albu.ToTensor(),
            ]),
            t1_dir='time1', # Default t1 dir for SYSU-CD
            t2_dir='time2', # Default t2 dir for SYSU-CD
            lbl_dir='label',  # Default label dir
            batch_size=1,
            num_workers=0,
            training=False
        ))