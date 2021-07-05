import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import HEADS
from mmdet.core import roi2bbox, multi_apply
from .convfc_bbox_head import Shared2FCBBoxHead


@HEADS.register_module()
class VRM2FCBBoxHead(Shared2FCBBoxHead):
    """BBox Head with Vanilla Relation Module (VRM).
    """

    def __init__(self,
                 sampler_num=512,
                 sampler_pos_fraction=0.25,
                 num_vrms=[1, 1],
                 num_vrm_group=16,
                 position_embedding_channels=64,
                 wave_length=1000,
                 *args,
                 **kwargs):
        super(VRM2FCBBoxHead, self).__init__(*args, **kwargs)
        assert len(num_vrms) == self.num_shared_fcs
        self.sampler_num = sampler_num
        self.sampler_pos_fraction = sampler_pos_fraction
        self.num_rel_modules = num_vrms
        # add vanilla relation modules
        self.vrms = self._add_vrm_branch(
            self.shared_out_channels,
            num_vrms,
            num_vrm_group,
            position_embedding_channels,
            wave_length)

    def _add_vrm_branch(self,
                        in_channels,
                        num_vrms,
                        num_vrm_group,
                        position_embedding_channels,
                        wave_length):
        vrms_list = nn.ModuleList()
        for num in num_vrms:
            vrms = nn.ModuleList()
            for i in range(num):
                vrms.append(
                    VanillaRelationModule(
                        in_channels=in_channels,
                        position_embedding_channels=position_embedding_channels,
                        wave_length=wave_length,
                        num_groups=num_vrm_group))
            vrms_list.append(vrms)
        return vrms_list

    def forward(self, x, rois, nongt_inds=None):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for (fc, vrms) in zip(self.shared_fcs, self.vrms):
                x = fc(x)
                skip_x = x
                for vrm in vrms:
                    x = vrm(x, rois, nongt_inds)  # vanilla relation module
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
        return cls_score, bbox_pred

    def _get_target_single(self, cls_scores, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains all the gt_boxes,
                has shape (num_gt, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains all the gt_labels,
                has shape (num_gt).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        # online hard example mining
        ohem_weights = self._ohem_pipeline(cls_scores, labels, label_weights, num_samples, num_pos, num_neg)
        label_weights *= ohem_weights
        bbox_weights *= ohem_weights[:, None]

        return labels, label_weights, bbox_targets, bbox_weights

    def _ohem_pipeline(self, cls_scores, labels, label_weights, num_samples, num_pos, num_neg):
        ohem_weights = label_weights.new_zeros(num_samples)
        # hard positives
        expected_num_pos = int(self.sampler_pos_fraction * self.sampler_num)
        if num_pos <= expected_num_pos:
            ohem_weights[:num_pos] = 1.0
            _num_pos = num_pos
        else:
            hard_pos_inds = self._hard_mining(cls_scores[:num_pos], labels[:num_pos],
                                              label_weights[:num_pos], expected_num_pos)
            ohem_weights[:num_pos][hard_pos_inds] = 1.0
            _num_pos = expected_num_pos
        # hard negatives
        _num_neg = self.sampler_num - _num_pos
        if _num_neg > num_neg:
            ohem_weights[-num_neg:] = 1.0
        else:
            hard_neg_inds = self._hard_mining(cls_scores[-num_neg:], labels[-num_neg:],
                                              label_weights[-num_neg:], _num_neg)
            ohem_weights[-num_neg:][hard_neg_inds] = 1.0
        return ohem_weights

    def _hard_mining(self, cls_scores, labels, label_weights, num_expected):
        with torch.no_grad():
            loss = self.loss(
                cls_score=cls_scores,
                bbox_pred=None,
                rois=None,
                labels=labels,
                label_weights=label_weights,
                bbox_targets=None,
                bbox_weights=None,
                reduction_override='none')['loss_cls']
            _, topk_loss_inds = loss.topk(num_expected)
        return topk_loss_inds

    def get_targets(self,
                    cls_scores_list,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 4), the last dimension 4 represents
                  [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            cls_scores_list,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights


class VanillaRelationModule(nn.Module):

    def __init__(self,
                 in_channels,
                 position_embedding_channels,
                 wave_length,
                 num_groups):
        super(VanillaRelationModule, self).__init__()
        self.in_channels = in_channels  # 1024
        self.num_groups = num_groups  # 16
        self.position_embedding_channels = position_embedding_channels  # 64
        self.wave_length = wave_length  # 1000
        self.group_channels = in_channels // num_groups  # 64

        self.w_g = nn.Conv2d(position_embedding_channels, num_groups, 1)
        self.w_q = nn.Linear(in_channels, in_channels)
        self.w_k = nn.Linear(in_channels, in_channels)
        self.w_v = nn.Conv2d(in_channels * num_groups, in_channels, 1, groups=num_groups)

        self.position_encoder = PositionEncoder(
            position_embedding_channels=position_embedding_channels,
            wave_length=wave_length)

        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        for module_list in [self.w_q, self.w_k, self.w_g]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.constant_(self.w_v.bias, 0)

    def forward(self, feats, rois, nongt_inds_list=None):
        # get position embedding
        bboxes_list = roi2bbox(rois)

        if nongt_inds_list is None:
            nongt_inds_list = [feats.new_ones(feats.shape[0], dtype=torch.bool)]

        with torch.no_grad():
            position_embeddings_list = self.position_encoder(bboxes_list, nongt_inds_list)

        # relation module
        relation_feat_list = []
        feat_list = feats.split([len(inds) for inds in nongt_inds_list], dim=0)
        num_imgs = len(nongt_inds_list)
        for i in range(num_imgs):
            # select non-gt feature
            nongt_feat = feat_list[i][nongt_inds_list[i], :]
            if nongt_feat.size(0) == 0:
                nongt_feat = feat_list[i]
            # position g
            g_feat = position_embeddings_list[i]
            g_feat = self.relu(self.w_g(g_feat))  # [1, num_groups, num_bboxes, num_nongt]: 1x16xNxM
            aff_weight = g_feat.permute(2, 1, 3, 0).contiguous().squeeze(dim=-1)  # [num_bboxes, num_groups, num_nongt]: Nx16xM
            # multi-head q & k
            q_feat = self.w_q(feat_list[i])  # [num_bboxes, in_channels]: Nx1024
            q_feat_batch = q_feat.contiguous().view(-1, self.num_groups, self.group_channels)  # [num_bboxes, num_groups, group_channels]: Nx16x64
            q_feat_batch = q_feat_batch.permute(1, 0, 2).contiguous()  # [num_groups, num_bboxes, group_channels]: 16xNx64
            k_feat = self.w_k(nongt_feat)  # [num_nongt, in_channels]: Mx1024
            k_feat_batch = k_feat.contiguous().view(-1, self.num_groups, self.group_channels)  # [num_nongt, num_groups, group_channels]: Mx16x64
            k_feat_batch = k_feat_batch.permute(1, 2, 0).contiguous()  # [num_groups, group_channels, num_nongt]: 16x64xM
            v_data = nongt_feat
            # q & k
            aff = torch.matmul(q_feat_batch, k_feat_batch)  # [num_groups, num_bboxes, num_nongt]: 16xNxM
            aff_scale = (1.0 / aff.new_tensor(self.group_channels).float().sqrt()) * aff
            aff_scale = aff_scale.permute(1, 0, 2).contiguous()  # [num_bboxes, num_groups, num_nongt]: Nx16xM
            # q & k & g
            weighted_aff = aff_weight.clamp(min=1e-6).log() + aff_scale  # [num_bboxes, num_groups, num_nongt]: Nx16xM
            aff_softmax = F.softmax(weighted_aff, dim=2).contiguous().view(-1, nongt_feat.shape[0])  # (Nx16)xM
            # output_t: # [num_bboxes, num_groups * in_channels, 1, 1]
            output_t = torch.matmul(aff_softmax, v_data)  # [num_bboxes * num_groups, in_channels]: Nx(16x1024)
            output_t = output_t.contiguous().view(-1, self.num_groups * self.in_channels, 1, 1)  # [num_bboxes, num_groups * in_channels, 1, 1]: Nx(16x1024)
            # linear_out: [num_bboxes, in_channels, 1, 1]
            relation_feat = self.w_v(output_t).contiguous().view(-1, self.in_channels)  # [num_bboxes, in_channels]: Nx1024
            relation_feat_list.append(relation_feat)
        return torch.cat(relation_feat_list, dim=0)


class PositionEncoder(nn.Module):
    def __init__(self,
                 position_embedding_channels,
                 wave_length):
        super(PositionEncoder, self).__init__()
        self.position_embedding_channels = position_embedding_channels  # 64
        self.wave_length = wave_length  # 1000

    def _convert_to_coco_format(self, bboxes):
        x1, y1, x2, y2 = bboxes.split([1, 1, 1, 1], dim=1)
        center_Xs = (x1 + x2) * 0.5
        center_Ys = (y1 + y2) * 0.5
        Ws = x2 - x1 + 1.
        Hs = y2 - y1 + 1.
        return center_Xs, center_Ys, Ws, Hs

    def _extract_position_matrix(self, bboxes, nongt_bboxes):
        # covert to coco format
        # [num_bboxes, 1]
        Xs, Ys, Ws, Hs = self._convert_to_coco_format(bboxes)
        # [num_nongt_bboxes, 1]
        nongt_Xs, nongt_Ys, nongt_Ws, nongt_Hs = self._convert_to_coco_format(nongt_bboxes)
        # delta Xs
        delta_Xs = ((Xs - nongt_Xs.permute(1, 0)) / Ws.float()).abs().clamp(min=1e-3).log()
        delta_Ys = ((Ys - nongt_Ys.permute(1, 0)) / Hs.float()).abs().clamp(min=1e-3).log()
        delta_Ws = (Ws / nongt_Ws.permute(1, 0)).log()
        delta_Hs = (Hs / nongt_Hs.permute(1, 0)).log()
        position_mat = torch.stack([delta_Xs, delta_Ys, delta_Ws, delta_Hs], dim=-1)
        return position_mat  # [num_bboxes, num_nongt_bboxes, 4]

    def _extract_position_embedding(self, position_mat, embedding_channels, wave_length=1000):
        feat_arange = torch.arange(0, embedding_channels // 8, dtype=position_mat.dtype, device=position_mat.device)
        dim_mat = position_mat.new_full([1, ], wave_length).pow((8. / embedding_channels) * feat_arange)
        dim_mat = dim_mat.view(1, 1, 1, -1)  # [1, 1, 1, embedding_channels / 8]
        position_mat = 100. * position_mat.unsqueeze(dim=3)  # [num_bboxes, num_nongt_bboxes, 4, 1]
        div_mat = position_mat / dim_mat  # [num_bboxes, num_nongt_bboxes, 4, embedding_channels/8]
        sin_mat = div_mat.sin()  # [num_bboxes, num_nongt_bboxes, 4, embedding_channels / 8]
        cos_mat = div_mat.cos()  # [num_bboxes, num_nongt_bboxes, 4, embedding_channels / 8]
        # embedding
        embedding = torch.cat([sin_mat, cos_mat], dim=3)  # [num_bboxes, num_nongt_bboxes, 4, embedding_channels / 4]: NxMx4x16
        embedding = embedding.view(*embedding.shape[:2], -1)  # [num_bboxes, num_nongt_bboxes, embedding_channels]: NxMx64
        return embedding


    def forward(self, bboxes_list, nongt_inds_list):
        position_embeddings = []
        for i in range(len(bboxes_list)):
            bboxes = bboxes_list[i]
            # get non-gt bboxes
            nongt_bboxes = bboxes[nongt_inds_list[i], :]
            if nongt_bboxes.size(0) == 0:
                nongt_bboxes = bboxes
            # extract position matrix: [num_bboxes, num_nongt_bboxes, 4]
            position_mat = self._extract_position_matrix(bboxes, nongt_bboxes)
            # extract position embedding: [position_embedding_channels, num_bboxes, num_nongt_bboxes]
            position_embedding = self._extract_position_embedding(
                position_mat,
                self.position_embedding_channels,
                self.wave_length)  # [num_bboxes, num_nongt_bboxes, embedding_channels]: NxMx64
            position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(dim=0)  # [1, embedding_channels, num_bboxes, num_nongt_bboxes]: 64xNxM
            position_embeddings.append(position_embedding)
        return position_embeddings
