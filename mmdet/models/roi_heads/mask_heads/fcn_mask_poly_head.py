import numpy as np
import torch
import cv2

from mmdet.models.builder import HEADS

from .fcn_mask_head import FCNMaskHead, _do_paste_mask

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


@HEADS.register_module()
class FCNMaskPolyHead(FCNMaskHead):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=80,
                 class_agnostic=False,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 conv_cfg=None,
                 norm_cfg=None,
                 predictor_cfg=dict(type='Conv'),
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 segm_type='rect'):
        assert segm_type in ['poly', 'rect']
        self.segm_type = segm_type

        super(FCNMaskPolyHead, self).__init__(
            num_convs=num_convs,
            roi_feat_size=roi_feat_size,
            in_channels=in_channels,
            conv_kernel_size=conv_kernel_size,
            conv_out_channels=conv_out_channels,
            num_classes=num_classes,
            class_agnostic=class_agnostic,
            upsample_cfg=upsample_cfg,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            predictor_cfg=predictor_cfg,
            loss_mask=loss_mask)

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, det_scores, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            mask_pred = det_bboxes.new_tensor(mask_pred)

        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes
        bboxes = det_bboxes[:, :4]
        scores = det_scores
        labels = det_labels

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes / scale_factor

        threshold = rcnn_test_cfg.mask_thr_binary

        if not self.class_agnostic:
            mask_pred = mask_pred[range(len(mask_pred)), labels][:, None]  # (n, 1, h, w)
            # To select the connect component with the max area
            mask_pred = self._select_max_cc(mask_pred >= threshold)

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        if rescale:
            im_mask = torch.zeros(
                N,
                np.round(img_h * scale_factor[0].cpu().numpy()).astype(np.int32),
                np.round(img_w * scale_factor[0].cpu().numpy()).astype(np.int32),
                device=device,
                dtype=torch.bool if threshold >= 0 else torch.uint8)
        else:
            im_mask = torch.zeros(
                N,
                img_h,
                img_w,
                device=device,
                dtype=torch.bool if threshold >= 0 else torch.uint8)

        for inds in chunks:
            if rescale:
                masks_chunk, spatial_inds = _do_paste_mask(
                    mask_pred[inds],
                    det_bboxes[inds, :4],
                    np.round(img_h * scale_factor[0].cpu().numpy()).astype(np.int32),
                    np.round(img_w * scale_factor[0].cpu().numpy()).astype(np.int32),
                    skip_empty=device.type == 'cpu')
            else:
                masks_chunk, spatial_inds = _do_paste_mask(
                    mask_pred[inds],
                    det_bboxes[inds, :4],
                    img_h,
                    img_w,
                    skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        keep = self.ins_mask_nms(im_mask, scores, labels, rcnn_test_cfg.mask_nms_thr)

        # mask -> poly or rect
        if self.segm_type == 'poly':
            segms, valid = self._mask2poly(mask_pred[keep, ...], bboxes[keep, :4])
        elif self.segm_type == 'rect':
            if rescale:
                segms, valid = self._mask2rect(im_mask[keep, ...], scale_factor[0].cpu().numpy())
            else:
                segms, valid = self._mask2rect(im_mask[keep, ...], 1.0)
        assert len(keep) == len(valid)
        keep = keep[valid].cpu().numpy()

        for i, keep_ind in enumerate(keep):
            cls_segms[labels[keep_ind]].append(segms[i])
        return cls_segms, [keep]  # TODO: for multi-classes

    def _select_max_cc(self, masks):
        """Select connected component with max area

        Args:
            masks (Tensor): shape (n, 1, h, w).

        Returns:
            Tensor: shape (n, 1, h, w).
        """
        masks_np = masks.cpu().numpy().astype(np.uint8)
        for i in range(masks_np.shape[0]):
            cc_num, cc = cv2.connectedComponents(masks_np[i, 0], connectivity=4)
            max_idx = -1
            max_area = 0
            for cc_idx in range(1, cc_num):
                area = (cc == cc_idx).sum()
                if area > max_area:
                    max_area = area
                    max_idx = cc_idx
            masks_np[i, 0, cc != max_idx] = 0
        return torch.from_numpy(masks_np).to(masks.device)

    def _mask2poly(self, masks, boxes):
        """Convert mask results to polygons

        Args:
            masks (Tensor): shape (n, h, w).
            boxes (Tensor): shape (n, 4).

        Returns:
            list: a list (length = image num) of list (length = mask num) of
                list (length = poly num) of numpy array
        """
        masks_np = masks.cpu().numpy().astype(np.uint8)
        boxes_np = boxes.cpu().numpy().astype(np.int32)
        assert masks_np.shape[0] == boxes_np.shape[0]
        polys = []
        valids = []
        for i in range(masks.shape[0]):
            x0, y0, x1, y1 = boxes_np[i]
            points = np.array(np.where(masks_np[i, 0] == 1)).transpose((1, 0))[:, ::-1]
            if points.shape[0] == 0:
                valids.append(False)
                polys.append(np.zeros(8, dtype=np.int32).tolist())
                continue
            hull = cv2.convexHull(points.astype(np.int32), True)
            concave_hull_pts = np.asarray([pt[0] for pt in hull])
            poly = concave_hull_pts.reshape(-1).astype(np.float32)

            poly[::2] *= (boxes_np[i, 2] - boxes_np[i, 0]) / masks_np.shape[3]
            poly[1::2] *= (boxes_np[i, 3] - boxes_np[i, 1]) / masks_np.shape[2]
            poly[::2] += x0
            poly[1::2] += y0

            polys.append(np.round(poly).astype(np.int32).tolist())
            valids.append(True)

        return polys, torch.from_numpy(np.array(valids)).to(masks.device)

    def _mask2rect(self, masks, scale_factor):
        """Convert mask results to rectangles

        Args:
            masks (Tensor): shape (n, h, w).
            boxes (Tensor): shape (n, 4).

        Returns:
            list: a list (length = image num) of list (length = mask num) of
                list (length = rect num) of numpy array
        """
        masks_np = masks.cpu().numpy().astype(np.uint8)
        rects = []
        valids = []
        for i in range(masks.shape[0]):
            points = np.array(np.where(masks_np[i] == 1)).transpose((1, 0))[:, ::-1]
            if points.shape[0] == 0:
                valids.append(False)
                rects.append(np.zeros(8, dtype=np.int32).tolist())
                continue
            rect = cv2.boxPoints(cv2.minAreaRect(np.array(points).reshape(-1, 2))).reshape(-1) / float(scale_factor)
            rects.append(np.round(rect).astype(np.int32).tolist())
            valids.append(True)

        _rects = np.asarray(rects).reshape((len(rects), 4, 2))
        side_1 = np.sqrt(np.square(_rects[:, 0] - _rects[:, 3]).sum(axis=-1))
        side_2 = np.sqrt(np.square(_rects[:, 0] - _rects[:, 1]).sum(axis=-1))
        valids_np = (side_1 > 0) & (side_2 > 0) & np.array(valids)
        valids = torch.from_numpy(np.array(valids_np)).to(masks.device)

        rects = [rect for i, rect in enumerate(rects) if valids_np[i]]

        return rects, valids

    def ins_mask_nms(self, mask_pred, scores, labels, nms_thresh):
        """NMS for multi-class bboxes.

                Args:
                    masks (Tensor): shape (n, 1, h, w).
                    scores (Tensor): shape (1,).
                    labels (Tensor): shape (1,).
                    nms_thresh (float)

                Returns:
                    Tensor: shape (n, 1, h, w).
                """
        n_samples = len(labels)
        if n_samples == 0:
            return []

        # sort and keep top nms_pre
        sort_inds = torch.argsort(scores, descending=True)

        sum_masks = mask_pred.sum((1, 2)).float()
        mask_pred = mask_pred.reshape(n_samples, -1).float()
        # inter.
        inter_matrix = torch.mm(mask_pred, mask_pred.transpose(1, 0))
        # union.
        sum_masks_x = sum_masks.expand(n_samples, n_samples)
        # iou.
        iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
        # label_specific matrix.
        cate_labels_x = labels.expand(n_samples, n_samples)
        label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

        iou_matrix = iou_matrix * label_matrix.to(iou_matrix.device)

        keep = []
        while sort_inds.size(0) > 0:
            i = sort_inds[0]
            keep.append(i)
            ious = iou_matrix[i, sort_inds[1:]]
            inds = torch.where(ious <= nms_thresh)[0]
            sort_inds = sort_inds[inds + 1]
        return torch.stack(keep, dim=0).to(mask_pred.device)
