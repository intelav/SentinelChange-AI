from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop
import ever as er

data = dict(
    train=dict(
        type='SYSUCDLoader' ,
        params=dict(
            root_dir='/media/avaish/aiwork/satellite-work/datasets/sysu-cd/train',
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True)
                ], p=0.5),
                er.preprocess.albu.RandomDiscreteScale([0.75, 1.25, 1.5], p=0.5),
                RandomCrop(256, 256, True),
                Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225), max_pixel_value=255),
                er.preprocess.albu.ToTensor(),
            ]),
            batch_size=32,
            num_workers=8,
            training=True
        ),
    ),
    test=dict(
        type='SYSUCDLoader',
        params=dict(
            root_dir='/media/avaish/aiwork/satellite-work/datasets/sysu-cd/test',
            transforms=Compose([
                Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225), max_pixel_value=255),
                er.preprocess.albu.ToTensor(),
            ]),
            batch_size=8,
            num_workers=0,
            training=False
        ),
    ),
)
optimizer = dict(
    type='sgd',
    params=dict(
        momentum=0.9,
        weight_decay=0.0001
    ),
    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    )
)
learning_rate = dict(
    type='poly',
    params=dict(
        base_lr=0.03, # NOTE: This LR may be high, consider starting with 0.01
        power=0.9,
        # CRITICAL CHANGE 4: Update max_iters for SYSU-CD (e.g., 40 epochs)
        # 12000 images / 8 batch_size * 40 epochs = 60000
        max_iters=60000,
    )
)
train = dict(
    forward_times=1,
    # CRITICAL CHANGE 4 (cont.): Match max_iters here
    num_iters=12000,
    eval_per_epoch=False,
    summary_grads=False,
    summary_weights=False,
    distributed=True,
    apex_sync_bn=True,
    sync_bn=False,
    eval_after_train=True,
    log_interval_step=50,
    save_ckpt_interval_epoch=10, # Saving every 10 epochs seems more reasonable
    eval_interval_epoch=10,
)
test = dict(
)