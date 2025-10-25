from ever.module import fpn
from configs.trainxView2 import standard

config = dict(
    model=dict(
        type='ChangeStar',
        params=dict(
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
            classifier=dict(
                in_channels=256,
                out_channels=1,
                scale=4.0
            ),
            detector=dict(
                name='convs',
                in_channels=256 * 2,
                inner_channels=16,
                out_channels=1,
                scale=4.0,
                num_convs=4,
                symmetry_loss=True,
                drop_rate=0.2,
                t1t2=True,
                t2t1=True
            ),
            loss_config=dict(
                change=dict(
                    on=False,
                    bce=True,
                    weight=0.5,
                    ignore_index=-1
                ),
                semantic=dict(
                    on=True,
                    bce=True,
                    dice=True,
                    ignore_index=-1,
                ),
            )
        ),
    ),
    data=standard.data,
    optimizer=standard.optimizer,
    learning_rate=standard.learning_rate,
    train=standard.train,
    test=standard.test
)
# You can set this value directly in r50_farseg_changemixin_symmetry.py
# This will override any save_interval_step that might be in standard.py
config['train']['save_interval_step'] = 500  # Example: Save every 500 global steps (iterations)
# Given 40,000 total iterations:
# 500 steps: Saves 80 checkpoints (40000 / 500)
# 1000 steps: Saves 40 checkpoints
# Adjust based on how much progress you're willing to lose and available disk space.