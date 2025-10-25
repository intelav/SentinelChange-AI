from ever.module import fpn
from configs.levircd import standard

config = dict(
    model=dict(
        type='ChangeStarBiSup',
        params=dict(
            segmenation=dict(
                model_type='farseg',
                backbone=dict(
                    resnet_type='resnet18',
                    pretrained=True,
                    freeze_at=0,
                    output_stride=32,
                ),
                head=dict(
                    fpn=dict(
                        in_channels_list=(64, 128, 256, 512),
                        out_channels=256,
                        conv_block=fpn.conv_bn_relu_block
                    ),
                    fs_relation=dict(
                        scene_embedding_channels=512,
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
                loss_config=dict(
                    bce=True,
                    dice=True,
                    ignore_index=-1
                )
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
    data=standard.data,
    optimizer=standard.optimizer,
    learning_rate=standard.learning_rate,
    train=standard.train,
    test=standard.test
)
# --- ADD THIS SECTION to the 'train' configuration ---
config['train']['save_interval_step'] = 500  # Save every 500 global steps (iterations)
# You can adjust this value. For example:
# 200 steps: Saves approximately every ~3-4 minutes based on your log (0.9-1.0 sec/step).
# 500 steps: Saves approximately every ~8-9 minutes.
# 1000 steps: Saves approximately every ~16-18 minutes.
# Choose a value that balances recovery time with disk space usage.
# -----------------------------------------------------