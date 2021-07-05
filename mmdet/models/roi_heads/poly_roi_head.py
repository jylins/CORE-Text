from mmdet.core import bbox2result
from ..builder import HEADS
from .standard_roi_head import StandardRoIHead
from .test_mixins import MaskPolyTestMixin


@HEADS.register_module()
class PolyRoIHead(StandardRoIHead, MaskPolyTestMixin):
    """Polygon roi head including one bbox head and one mask head
       with a specific post process for text detection.
    """
    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        pass

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results, keeps = self.simple_test_mask_poly(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            for i in range(len(segm_results)):
                for label in range(len(bbox_results[i])):
                    bbox_results[i][label] = bbox_results[i][label][keeps[i][label]]
            # return bbox_results, segm_results
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        pass
