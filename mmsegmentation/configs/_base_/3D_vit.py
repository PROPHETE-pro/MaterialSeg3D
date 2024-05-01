_base_ = [
'../_base_/default_runtime.py'
]


# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (1024, 1024)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
num_classes = 28


model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='/home/ruitong_gan/3D/segment-anything/SAM_b.pth',
    backbone=dict(
        type='SAMvit',
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(2, 5, 8, 11),
        out_chans=256,
        qkv_bias=True,
        ),
    neck=dict(
        type='MultiLevelNeck',
        in_channels=[768, 768, 768, 768],
        out_channels=768,
        scales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=28,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            # class_weight=[1.827806273526598, 1.1505384474463052, 1.9961427782346053, 1.1404721864645975,
            #           0.8524145255163509, 0.9481104576488283, 4.991269232675962, 1.2744479618980034,
            #           1.8320375869939072, 0.4855403861302117, 1.0897772174877416, 1.5451343268288333,
            #           0.7451360761325668, 0.3950385032689794, 1.5429022161650416, 1.1538405632784352,
            #           1.4030927941448603, 1.0214320268797614, 1.8378927623264383, 0.7292644647545895,
            #           1.6218392848611516, 1.0710388318132456, 1.0404968281096338, 1.4011807437285182,
            #           1.557389246790005, 1.3319639494982687, 1.1204273153626783, 1.4144948644449729, 0.1]
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=28,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable











# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'backbone.patch_embed.norm': dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.norm': dict(lr_mult=0.1, decay_mult=0.0),
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]




# By default, models are trained on 8 GPUs with 2 images per GPU

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomChoiceResize',
        scales=[int(x * 0.1 * 1024) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=2048),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]


custom3D_train = dict(
    type = 'custom3DDataset_28',
    data_root = '/home/ruitong_gan/3D/data/',
data_prefix=dict(
            img_path='28_labels/crop_Image', seg_map_path='28_labels/crop_GT'),
    pipeline = train_pipeline
)

custom3D_val = dict(
    type = 'custom3DDataset_28',
    data_root = '/home/ruitong_gan/3D/data/',
data_prefix=dict(
            img_path='28_labels/val/Image', seg_map_path='28_labels/val/GT'),
    pipeline = test_pipeline
)

custom3D_class = dict(
    type='ClassBalancedDataset',
    dataset=custom3D_train,
    oversample_thr=0.04
)



train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=custom3D_train)


val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=custom3D_val)
test_dataloader = val_dataloader


val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# training schedule for 160k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=5000,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

find_unused_parameters = True