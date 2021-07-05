_base_ = './base.py'
# model settings
model = dict(
    roi_head=dict(
        type='VRMPolyRoIHead',
        bbox_head=dict(
            type='VRM2FCBBoxHead',
            sampler_num=512,
            sampler_pos_fraction=0.25,
            num_vrms=[1, 1],
            num_vrm_group=16,
            position_embedding_channels=64,
            wave_length=1000)),
    # model training and testing settings
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(
                type='RandomSampler',
                num=1e+8,  # all samples
                pos_fraction=1.0,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.475,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5,
            mask_nms_thr=0.2)))