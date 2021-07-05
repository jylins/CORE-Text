# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='exp',
    warmup_iters=500,
    warmup_ratio=1 / 16.,
    step=[20, 32])
runner = dict(type='EpochBasedRunner', max_epochs=40)
