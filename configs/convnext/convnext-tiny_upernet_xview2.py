custom_imports = dict(imports=['mmseg.datasets.xview2'], allow_failed_imports=False)
_base_ = [
    '../_base_/models/upernet_convnext.py',  # UPerNet + ConvNeXt backbone
    '../_base_/default_runtime.py',  # Logging, checkpoints
    '../_base_/schedules/schedule_160k.py'  # Optimizer, LR schedule
]

# Dataset settings
dataset_type = 'CustomDataset'
data_root = 'C:\\Users\\alexm\\2025_fall\\ATIA\\geotiffs\\hold\\dataset'  # e.g., '/home/user/dataset' or '/content/drive/MyDrive/xview2/dataset'
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=4,  # Small for CPU testing; increase to 4–8 on Colab GPU
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images', seg_map_path='masks'),
        pipeline=train_pipeline)
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images_val', seg_map_path='masks_val'),
        pipeline=test_pipeline)
)
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# Model settings
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmpretrain.ConvNeXt',
        arch='tiny',
        drop_path_rate=0.1,
        init_cfg=None),  # From scratch
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=2),  # Binary
    auxiliary_head=dict(in_channels=384, num_classes=2),
    test_cfg=dict(mode='whole')  # Simpler for 512x512, no sliding window
)

# Optimizer and schedule
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
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

# Training settings
runner = dict(type='IterBasedRunner', max_iters=4000)  # ~5 epochs for 720 images, CPU test
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU')
env_cfg = dict(seed=42, deterministic=True)