import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet.models.builder import HEADS, build_loss
from mmdet.core import bbox_overlaps
from .vrm_bbox_head import VRM2FCBBoxHead


@HEADS.register_module()
class CORE2FCBBoxHead(VRM2FCBBoxHead):
    """BBox Head with COntrastive RElation (CORE) Module.
    """

    def __init__(self,
                 tau=0.2,
                 min_iof=0.7,
                 loss_sim=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(CORE2FCBBoxHead, self).__init__(*args, init_cfg=init_cfg, **kwargs)
        self.tau = tau
        self.min_iof = min_iof
        self.loss_sim = build_loss(loss_sim)

        # add relation embedding
        self.rel_fcs = nn.ModuleList()
        for i in range(self.num_shared_fcs):
            rel_fc = nn.Sequential(
                nn.Linear(
                    self.shared_out_channels,
                    self.shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.shared_out_channels, 128))
            self.rel_fcs.append(rel_fc)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='rel_fcs'),
                    ])
            ]

    def forward(self, x, rois, nongt_inds=None):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        rel_feats = []
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for (fc, vrms, rel_fc) in zip(self.shared_fcs, self.vrms, self.rel_fcs):
                x = fc(x)
                skip_x = x
                for vrm in vrms:
                    x = vrm(x, rois, nongt_inds)  # vanilla relation module
                    rel_feat = rel_fc(x)
                    rel_feats.append(rel_feat)
                x = skip_x + x
                x = self.relu(x)

        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, rel_feats

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             rel_feats_list=None,
             nongt_inds_list=None,
             bboxes_list=None):
        # cls & reg loss
        losses = super(CORE2FCBBoxHead, self).loss(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            rois=rois,
            labels=labels,
            label_weights=label_weights,
            bbox_targets=bbox_targets,
            bbox_weights=bbox_weights,
            reduction_override=reduction_override)
        # instance-wise contrastive loss
        if rel_feats_list is not None and nongt_inds_list is not None and bboxes_list is not None:
            losses['loss_inscl'] = self.sim_loss(bboxes_list, nongt_inds_list, rel_feats_list)
        return losses

    def sim_loss(self, bboxes_list, nongt_inds_list, rel_feats_list):
        sim_losses = []
        num_img = len(bboxes_list)
        sim_avg_factor = 0.
        for img_id in range(num_img):
            # bboxes pre-proprocess for each image
            gt_bboxes, nongt_bboxes, valid = self._bboxes_preprocess(
                bboxes_list[img_id], nongt_inds_list[img_id])
            if not valid:
                sim_losses.append(self._zero_loss(rel_feats_list, img_id))
                continue
            num_gts = gt_bboxes.size(0)
            num_nongts = nongt_bboxes.size(0)

            # nongt positive indexes
            nongt_iou_mat = bbox_overlaps(nongt_bboxes, gt_bboxes)
            nongt_iof_mat = bbox_overlaps(nongt_bboxes, gt_bboxes, mode='iof')
            nongt_max_iou, nongt_argmax_iou = nongt_iou_mat.max(dim=1)
            nongt_iof = nongt_iof_mat[torch.arange(nongt_bboxes.size(0)), nongt_argmax_iou]
            nongt_pt_inds = (nongt_iof >= self.min_iof) & (nongt_max_iou >= 0.1) & (nongt_max_iou < 0.5)
            nongt_pos_inds = (nongt_max_iou >= 0.5) | nongt_pt_inds

            # similarity matrix
            sim_mat_list = []
            for rel_feat in rel_feats_list:
                rel_norm = F.normalize(rel_feat[img_id], dim=1)
                nongt_rel_norm = F.normalize(rel_feat[img_id][nongt_inds_list[img_id], :], dim=1)
                nongt_rel_norm = nongt_rel_norm.permute(1, 0).contiguous()
                sim_mat = torch.einsum('nc,ck->nk', [rel_norm, nongt_rel_norm]).unsqueeze(dim=0)
                sim_mat_list.append(sim_mat)

            # instance-wise contrastive loss
            for gt_id in range(num_gts):
                pos_inds, neg_inds, valid = self._get_pos_neg_inds(
                    gt_id, num_nongts, nongt_pos_inds, nongt_argmax_iou)
                if not valid:
                    sim_losses.append(self._zero_loss(rel_feats_list, img_id))
                    continue
                for sim_mat in sim_mat_list:
                    try:
                        sim_loss = self.contrastive_loss(
                            sim_mat=sim_mat[:, gt_id, :],
                            pos_inds=pos_inds,
                            neg_inds=neg_inds)
                        sim_avg_factor += 1.
                        sim_losses.append(sim_loss)
                    except Exception:
                        sim_losses.append(self._zero_loss(rel_feats_list, img_id))
                        continue
        return torch.stack(sim_losses).sum().reshape(-1) / (sim_avg_factor + 1e-3)

    def contrastive_loss(self, sim_mat, pos_inds, neg_inds):
        logits_pos = (sim_mat[:, pos_inds]).reshape(-1).unsqueeze(dim=-1)
        logits_neg = (sim_mat[:, neg_inds]).reshape(-1).unsqueeze(dim=0).repeat(logits_pos.size(0), 1)
        logits = torch.cat([logits_pos, logits_neg], dim=1) / self.tau
        labels = logits.new_zeros(logits.shape[0], dtype=torch.long)
        return self.loss_sim(logits, labels)

    def _get_pos_neg_inds(self, gt_id, num_nongts, nongt_pos_inds, nongt_argmax_iou, min_neg=512):
        if gt_id + 1 > num_nongts:
            return None, None, False
        valid_nongt_pos_inds = nongt_pos_inds & (nongt_argmax_iou == gt_id)
        valid_nongt_neg_inds = ~valid_nongt_pos_inds
        num_neg = valid_nongt_neg_inds.sum()
        if not valid_nongt_pos_inds.any() or num_neg < min_neg:
            return None, None, False
        return valid_nongt_pos_inds, valid_nongt_neg_inds, True

    def _bboxes_preprocess(self, bboxes, nongt_inds):
        if bboxes.size(0) == 0:
            return None, None, False
        nongt_bboxes = bboxes[nongt_inds, :]
        gt_inds = (~nongt_inds).nonzero().reshape(-1)
        gt_bboxes = bboxes[gt_inds, :]
        num_gts = gt_bboxes.size(0)
        num_nongts = nongt_bboxes.size(0)
        if (num_gts == 0) or (num_nongts == 0):
            return None, None, False
        return gt_bboxes, nongt_bboxes, True

    def _zero_loss(self, feats, img_id):
        return sum([feat[img_id] * 0. for feat in feats]).sum()
