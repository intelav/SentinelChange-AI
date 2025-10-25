# Correct File Location: ChangeStar/configs/custom/finetune_custom_final.py

from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop
import ever as er
from ever.module import fpn

# 1. Data Configuration for Your Custom Dataset
data = dict(
    train=dict(
        type='CustomCDLoader', # Use our new custom loader
        params=dict(
            root_dir=('../data/MyCustomCD_Dataset/train',), # Path from ChangeStar root
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True)
                ], p=0.75),
                RandomCrop(512, 512, True),
                Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225), max_pixel_value=255),
                er.preprocess.albu.ToTensor(),
            ]),
            batch_size=4,
            num_workers=4,
            training=True
        ),
    ),
    test=dict(
        type='CustomCDLoader', # Use our new custom loader
        params=dict(
            root_dir='../data/MyCustomCD_Dataset/val', # Path to your validation split
            transforms=Compose([
                Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225), max_pixel_value=255),
                er.preprocess.albu.ToTensor(),
            ]),
            batch_size=2,
            num_workers=0,
            training=False
        ),
    ),
)

# 2. Optimizer Configuration
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

# 3. Learning Rate Schedule
learning_rate = dict(
    type='poly',
    params=dict(
        base_lr=1e-5, # Use a smaller learning rate for fine-tuning
        power=0.9,
        max_iters=20000,
    )
)

# 4. Training Loop Configuration
train = dict(
    forward_times=1,
    num_iters=20000,
    eval_per_epoch=False,
    summary_grads=False,
    summary_weights=False,
    distributed=True,
    apex_sync_bn=True,
    sync_bn=False,
    eval_after_train=True,
    log_interval_step=50,
    save_ckpt_interval_epoch=1000,
    eval_interval_epoch=1000,
)

# 5. Final Configuration Dictionary
config = dict(
    model=dict(
        type='ChangeStarBiSup',
        params=dict(
            # This section is copied from the r50_farseg_changestar_bisup.py config
            segmenation=dict(
                model_type='farseg',
                backbone=dict(
                    resnet_type='resnet50',
                    pretrained=True,
                    freeze_at=0,
                    output_stride=32,
                ),
                head=dict(
                    fpn=dict(
                        in_channels_list=(256, 512, 1024, 2048),
                        out_channels=256,
                        conv_block=fpn.conv_bn_relu_block
                    ),
                    fs_relation=dict(
                        scene_embedding_channels=2048,
                        in_channels_list=(256, 256, 256, 256),
                        out_channels=256,
                        scale_aware_proj=True
                    ),
                    fpn_decoder=dict(
                        in_channels=256,
                        out_channels=256,
                        in_feat_output_strides=(4, 8, 16, 32),
                        out_feat_output_stride=4,
                        classifier_config=None
                    )
                ),
            ),
            detector=dict(
                name='convs',
                in_channels=256 * 2,
                inner_channels=16,
                out_channels=1,
                scale=4.0,
                num_convs=4,
            ),
            loss_config=dict(
                bce=True,
                dice=True,
                ignore_index=-1
            )
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    # FIX: Add the required 'test' dictionary to the main config
    test=dict(),
    # Fine-tuning specific settings
    finetune=True,
    finetune_checkpoint_path='log/finetune-SYSUCD/r50_farseg_changestar/model-12000.pth',
    model_dir='./log/finetune-CUSTOM-FINAL/r50_farseg_changestar'
)
