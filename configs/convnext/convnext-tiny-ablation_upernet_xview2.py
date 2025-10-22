custom_imports = dict(
    imports=['mmseg.datasets.xview2'],
    allow_failed_imports=False
)
_base_ = [
    '../_base_/models/upernet_convnext.py',  # UPerNet + ConvNeXt backbone
    '../_base_/default_runtime.py',  # Logging, checkpoints
    '../_base_/schedules/schedule_40k.py',  # Optimizer, LR schedule
    '../_base_/datasets/xview2.py' 
]

# Dataset settings
work_dir = '/workspace/mmsegmentation_xview2/work_dirs/ablation1'

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict(
             project='xview2-segmentation',
             name='convnext-tiny-experiment',
         ))
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend',
             init_kwargs=dict(
                 project='xview2-segmentation',
                 name='convnext-tiny-experiment',
                 tags=['binary', 'convnext', 'xview2']
             ))
    ],
    name='visualizer'
)

# Model settings
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ConvNeXt_Ablation',
        arch='tiny',
        # drop_path_rate randomly drops residual connections during training to regularize
        drop_path_rate=0.1, # we want this to be lower because we are training from scratch
        init_cfg=None),  # From scratch
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=2,
        loss_decode=dict(
            type='DiceLoss',
            use_sigmoid=False  # Use softmax for 2 classes
        )
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=2,
        loss_decode=dict(
            type='DiceLoss',
            use_sigmoid=False  # Use softmax for 2 classes
        )
    ),
    test_cfg=dict(mode='whole')  # Simpler for 512x512, no sliding window
)

# Optimizer and schedule
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0012, betas=(0.9, 0.999), weight_decay=0.05),
    accumulative_counts=4, # we can't use a batch_size=16 like we did with the 4x4 stem. We will compensate by accumulating gradients. 4x4 = 16x1 = 16
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 6
    },
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')


param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', power=1.0, begin=1500, end=40000, eta_min=0.0, by_epoch=False)
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1000,
        max_keep_ckpts=3,
        by_epoch=False,
        save_best='mDice',   # automatically saves best model based on validation mDice
        rule='greater'      # larger mDice = better
    )
)
cfg_dict = {
    "model": "UPerNet + ConvNeXt tiny",
    "num_classes": 2,
    "batch_size": 16,
    "crop_size": crop_size,
    "optimizer": "AdamW",
    "lr": 0.0012,
    "max_iters": 40000,
    "dataset": "xview2",
    "work_dir": work_dir
}

log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
      #  dict(type='TensorboardLoggerHook', by_epoch=False),
        dict(type='MMSegWandbHook', 
             by_epoch=False, # The Wandb logger is also supported, It requires `wandb` to be installed.
             interval=200,
             init_kwargs={'entity': "alex-meislich-k-benhavns-universitet", # The entity used to log on Wandb
                          'project': "xview2-segmentation", # Project name in WandB
                          'config': cfg_dict}), # Check https://docs.wandb.ai/ref/python/init for more init arguments.
        # MMSegWandbHook is mmseg implementation of WandbLoggerHook. ClearMLLoggerHook, DvcliveLoggerHook, MlflowLoggerHook, NeptuneLoggerHook, PaviLoggerHook, SegmindLoggerHook are also supported based on MMCV implementation.
    ])


custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mDice',
        min_delta=0.001,
        patience=5,  # Stop after 5 evaluations
       # verbose=True
    )
]

# Training settings
train_cfg = dict(val_interval=2000)

# make our environment deterministic for research purposes
env_cfg = dict(seed=42, deterministic=True)

train_dataloader = dict(batch_size=4) # prevent an OOM


# test with:
# export PYTHONPATH=$(pwd):$PYTHONPATH
# python3 tools/train.py configs/convnext/convnext-tiny-ablation_upernet_xview2.py
# python3 tools/test.py configs/convnext/convnext-tiny_upernet_xview2.py work_dirs/baseline2/best_mDice_iter_40000.pth  --show-dir work_dirs/check_preds_baseline2
# python3 tools/analysis_tools/confusion_matrix.py configs/convnext/convnext-tiny_upernet_xview2.py  work_dirs/baseline2/pred_results.pkl work_dirs/baseline2/confusion_matrix --show
