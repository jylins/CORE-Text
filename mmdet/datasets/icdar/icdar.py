import logging
import os
import os.path as osp
import shutil
import tempfile
import zipfile
import json
import time
import copy
from pycocotools.coco import COCO

import numpy as np
from mmcv.utils import print_log

from ..builder import DATASETS
from ..coco import CocoDataset
from .icdar_eval import ICDAREval


@DATASETS.register_module()
class ICDARDataset(CocoDataset):

    CLASSES = ('text',)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_masks_ann_ignore = []
        for i, ann in enumerate(ann_info):
            # if ann.get('ignore', False):
            #     continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False) or ann.get('ignore', False):
                gt_bboxes_ignore.append(bbox)
                gt_masks_ann_ignore.append(ann['segmentation'])
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            masks_ignore=gt_masks_ann_ignore,
            seg_map=seg_map)

        return ann

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    data['segmentation'] = segms[i]  # diff from coco
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def _loadRes(self, resFile):
        res = COCO()
        res.dataset['images'] = [img for img in self.coco.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str or type(resFile) == str:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.coco.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.coco.getImgIds())), \
            'Results do not correspond to current coco set'
        if 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(
                self.coco.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if 'segmentation' not in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2] * bb[3]
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(
                self.coco.dataset['categories'])
            for id, ann in enumerate(anns):
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        print('DONE (t={:0.2f}s)'.format(time.time() - tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def _convert_to_icdar_format(self, out_dir):
        '''Convert to ICDAR format
        :return: None
        '''
        result_dir = osp.join(out_dir, 'results')
        zip_path = osp.join(out_dir, 'submit.zip')
        zip_file = zipfile.ZipFile(zip_path, 'w')

        imgIds = self.coco.getImgIds()
        catIds = self.coco.getCatIds()

        for idx in range(len(imgIds)):
            img_id = imgIds[idx]
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=[img_id], catIds=catIds))
            filename = 'res_img_' + self.cocoDt.imgs[img_id]['filename'].split('/')[-1].split('.')[0].split('_')[
                -1] + '.txt'
            filename = osp.join(result_dir, filename)
            segms = np.array([dt['segmentation'] for dt in dts if
                              'segmentation' in dt.keys() and isinstance(dt['segmentation'], list)])
            scores = np.array([dt['score'] for dt in dts if 'score' in dt.keys()])

            if segms.shape[0] != 0:
                with open(filename, 'w+') as f:
                    for (segm, score) in zip(segms, scores):
                        if score > 1:
                            score = 1
                        segm = np.array(segm, dtype=np.int32)
                        segm_str = [str(res) for res in segm]
                        segm_str += [str(score)]
                        line = ','.join(segm_str) + '\r\n'
                        f.write(line)
                    f.close()
                zip_file.write(filename, osp.basename(filename))
        zip_file.close()

    def format_results(self, results, jsonfile_prefix=None, logger=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
            if osp.exists(jsonfile_prefix):
                shutil.rmtree(path=jsonfile_prefix)
            jsonfile_prefix = osp.join(jsonfile_prefix, 'results')
        os.makedirs(jsonfile_prefix)
        result_files = self.results2json(results, jsonfile_prefix)

        try:
            self.cocoDt = self._loadRes(result_files['segm'])
            self._convert_to_icdar_format(osp.dirname(jsonfile_prefix))
        except IndexError:
            print_log(
                'The testing results of the whole dataset is empty.',
                logger=logger,
                level=logging.ERROR)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='segm',
                 logger=None,
                 jsonfile_prefix=None,
                 gt_path=None):
        """Evaluation in COCO protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            gt_path (str | None): The path of ground truth. Default: None.

        Returns:
            dict[str: float]
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['segm']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix, logger)

        if tmp_dir is not None:
            jsonfile_prefix = tmp_dir.name

        eval_results = {}
        msg = f'Evaluating {metric}...'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        if metric not in result_files:
            raise KeyError(f'{metric} is not in results')

        cocoEval = ICDAREval(self.coco, self.cocoDt, metric,
                             out_dir=jsonfile_prefix,
                             gt_path=gt_path)
        cocoEval.params.catIds = self.cat_ids
        cocoEval.params.imgIds = self.img_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        all_res = cocoEval.stats[0::4]
        small_res = cocoEval.stats[1::4]
        medium_res = cocoEval.stats[2::4]
        large_res = cocoEval.stats[3::4]

        eval_results['precision'] = all_res[0]
        eval_results['recall'] = all_res[1]
        eval_results['hmean'] = all_res[2]

        eval_results['all'] = f'{all_res[0]:.4f} {all_res[1]:.4f} {all_res[2]:.4f}'
        eval_results['small'] = f'{small_res[0]:.4f} {small_res[1]:.4f} {small_res[2]:.4f}'
        eval_results['medium'] = f'{medium_res[0]:.4f} {medium_res[1]:.4f} {medium_res[2]:.4f}'
        eval_results['large'] = f'{large_res[0]:.4f} {large_res[1]:.4f} {large_res[2]:.4f}'

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results