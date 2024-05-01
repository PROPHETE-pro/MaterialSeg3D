_base_ = [
    '../../../_base_/models/fcn_hr18.py',
    '../../../_base_/default_runtime.py',
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)


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
            img_path='train/car/Image', seg_map_path='train/car/GT'),
    pipeline = train_pipeline
)

custom3D_val = dict(
    type = 'texture3DDataset',
    data_root = '/home/ruitong_gan/3D/data/texture',
data_prefix=dict(
            img_path='val/car/Image', seg_map_path='val/car/GT'),
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

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False)
]
# training schedule for 40k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=20000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))