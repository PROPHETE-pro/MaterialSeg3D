_base_ = [
    '../../../_base_/models/upernet_vit-b16_ln_mln.py',
     '../../../_base_/default_runtime.py',
    '../../../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/vit_base_patch16_224.pth',
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    )
]

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
            img_path='train/buildings/Image', seg_map_path='train/buildings/GT'),
    pipeline = train_pipeline
)

custom3D_val = dict(
    type = 'texture3DDataset',
    data_root = '/home/ruitong_gan/3D/data/texture',
data_prefix=dict(
            img_path='val/buildings/Image', seg_map_path='val/buildings/GT'),
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


# resume_from = "/home/ruitong_gan/3D/mmsegmentation/work_dir/car/iter_30000.pth"

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator



