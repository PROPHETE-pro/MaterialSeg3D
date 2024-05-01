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
num_classes = 20


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
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
            # class_weight=[1.140546280033195, 0.8065108095645472, 1.583729717956947,
            #               0.8166341102832059, 1.1953530153756073, 1.5266202606950854, 2.440800949814674,
            #               1.1602980455946896, 1.4961748575807308, 0.47990875164434516, 1.511730895524171,
            #               1.2422262668055604, 0.5419533109891433, 1.6837089484750791, 1.7308032423054336,
            #               1.5071253988892497, 1.1386206103345216, 0.5129235488956965, 1.6016182962674348,
            #               1.9377099368561053, 1.7221766370773655, 1.333192912743074, 1.1766787280191293,
            #               1.9469286341708252, 1.2605452203462426, 0.5356510946269596, 1.154698630726214,
            #               2.0852789515201544, 2.6584862447045707]
        ),

    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
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
    dict(type='LoadAnnotations'),
    dict(
        type='RandomChoiceResize',
        scales=[int(x * 0.1 * 1024) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=2048),
    # dict(
    #     type='RandomResize',
    #     scale=(1024, 1024),
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=False),
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
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]


custom3D_train = dict(
    type = 'texture3DDataset',
    data_root = '/home/ruitong_gan/3D/data/texture',
data_prefix=dict(
            img_path='train/furniture/Image', seg_map_path='train/furniture/GT'),
    pipeline = train_pipeline
)

custom3D_val = dict(
    type = 'texture3DDataset',
    data_root = '/home/ruitong_gan/3D/data/texture',
data_prefix=dict(
            img_path='val/furniture/Image', seg_map_path='val/furniture/GT'),
    pipeline = test_pipeline
)

custom3D_class = dict(
    type='ClassBalancedDataset',
    dataset=custom3D_train,
    oversample_thr=0.03
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
    type='IterBasedTrainLoop', max_iters=80000, val_interval=5000)
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