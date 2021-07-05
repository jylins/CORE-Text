from mmdet.core import bbox2roi
from ..builder import HEADS
from .poly_roi_head import PolyRoIHead


@HEADS.register_module()
class VRMPolyRoIHead(PolyRoIHead):
    """PolyRoIHead with VRM.
    """

    def _bbox_forward(self, x, rois, nongt_inds_list=None):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats, rois, nongt_inds_list)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])

        # get non-gt indexs
        pos_is_gts = [res.pos_is_gt for res in sampling_results]
        nongt_inds_list = []
        for i in range(len(pos_is_gts)):
            gt_inds = rois.new_zeros((rois[:, 0] == i).nonzero().shape[0])
            gt_inds[:len(pos_is_gts[i])] = pos_is_gts[i]
            nongt = (gt_inds == 0)
            nongt_inds_list.append(nongt)

        bbox_results = self._bbox_forward(x, rois, nongt_inds_list)

        cls_scores_list = bbox_results['cls_score'].split([len(inds) for inds in nongt_inds_list], dim=0)
        bbox_targets = self.bbox_head.get_targets(cls_scores_list, sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
