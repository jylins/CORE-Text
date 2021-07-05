_base_ = './vrm.py'
load_from = './work_dirs/mask_rcnn_r50_fpn_train_core_pretrain/epoch_40.pth'
optimizer = dict(lr=0.04)
# model settings
model = dict(
    roi_head=dict(
        type='COREPolyRoIHead',
        bbox_head=dict(
            type='CORE2FCBBoxHead',
            tau=0.2,
            min_iof=0.7,
            loss_sim=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.1))),
    # model training and testing settings
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.57,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5,
            mask_nms_thr=0.2)))