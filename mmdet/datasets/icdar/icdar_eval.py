# modified from COCOeval in 'https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py'.

import os.path as osp
import numpy as np
import datetime
import time
from collections import defaultdict

from .tools import icdar_eval_tool
import zipfile


class ICDAREval:
    def __init__(self, icdarGt=None, icdarDt=None, iouType='segm', out_dir=None, gt_path=None):
        '''
        Initialize ICDAREval using icdar APIs for gt and dt
        :param icdarGt: icdar object with ground truth annotations
        :param icdarDt: icdar object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.icdarGt = icdarGt  # ground truth COCO API
        self.icdarDt = icdarDt  # detections COCO API
        self.params = {}  # evaluation parameters
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(iouType=iouType)  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts
        self.out_dir = out_dir
        self.gt_path = gt_path

        if not icdarGt is None:
            self.params.imgIds = sorted(icdarGt.getImgIds())
            self.params.catIds = sorted(icdarGt.getCatIds())

    def _save(self):
        '''
        Save result to submit.zip
        :return: None
        '''
        result_dir = osp.join(self.out_dir, 'results')
        zip_path = osp.join(self.out_dir, 'submit.zip')
        zip_file = zipfile.ZipFile(zip_path, 'w')

        p = self.params

        for idx in range(len(p.imgIds)):
            img_id = p.imgIds[idx]
            dts = self.icdarDt.loadAnns(self.icdarDt.getAnnIds(imgIds=[img_id], catIds=p.catIds))
            filename = 'res_img_' + self.icdarDt.imgs[img_id]['filename'].split('/')[-1].split('.')[0].split('_')[
                -1] + '.txt'
            filename = osp.join(result_dir, filename)
            segms = np.array([dt['segmentation'] for dt in dts if 'segmentation' in dt.keys() and isinstance(dt['segmentation'], list)])
            scores = np.array([dt['score'] for dt in dts if 'score' in dt.keys()])

            if segms.shape[0] != 0:
                with open(filename, 'w+') as f:
                    for (segm, score) in zip(segms, scores):
                        segm = np.array(segm, dtype=np.int32)
                        segm_str = [str(res) for res in segm]
                        segm_str += [str(score)]
                        line = ','.join(segm_str) + '\r\n'
                        f.write(line)
                    f.close()
                zip_file.write(filename, osp.basename(filename))
        zip_file.close()

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def accumulate(self, p=None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)  # 1
        K = len(p.catIds) if p.useCats else 1  # 1
        A = len(p.areaRng)  # 4
        M = len(p.maxDets)  # 1
        precision = -np.ones((T, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        hmean = -np.ones((T, K, A, M))
        scores = -np.ones((T, K, A, M))

        p = {'s': osp.join(self.out_dir, 'submit.zip'),
             'g': self.gt_path}

        res_dict = icdar_eval_tool(p)

        # all
        precision[:, :, 0, 0] = res_dict['method']['all']['precision']
        recall[:, :, 0, 0] = res_dict['method']['all']['recall']
        hmean[:, :, 0, 0] = res_dict['method']['all']['hmean']

        # small
        precision[:, :, 1, 0] = res_dict['method']['area']['small']['precision']
        recall[:, :, 1, 0] = res_dict['method']['area']['small']['recall']
        hmean[:, :, 1, 0] = res_dict['method']['area']['small']['hmean']

        # medium
        precision[:, :, 2, 0] = res_dict['method']['area']['medium']['precision']
        recall[:, :, 2, 0] = res_dict['method']['area']['medium']['recall']
        hmean[:, :, 2, 0] = res_dict['method']['area']['medium']['hmean']

        # large
        precision[:, :, 3, 0] = res_dict['method']['area']['large']['precision']
        recall[:, :, 3, 0] = res_dict['method']['area']['large']['recall']
        hmean[:, :, 3, 0] = res_dict['method']['area']['large']['hmean']

        self.eval = {
            'params': p,
            'counts': [T, K, A, M],  # [1, 1, 4, 1]
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, areaRng='all', maxDets=100):
            p = self.params
            title_type = ['Precision', 'Recall', 'F-measure']
            s_type = ['precision', 'recall', 'hmean']
            iStr = ' {:<9}  @[ IoU={:<4} | area={:>6s} | maxDets={:>3d} ] = {:0.4f}'
            titleStr = title_type[ap]
            iouStr = '{:0.2f}'.format(p.iouThrs[0])

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            mean_s = self.eval[s_type[ap]][:, :, aind, mind].mean()

            print(iStr.format(titleStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(0, areaRng='all', maxDets=self.params.maxDets[-1])
            stats[1] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[-1])
            stats[2] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[-1])
            stats[3] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[-1])
            stats[4] = _summarize(1, areaRng='all', maxDets=self.params.maxDets[-1])
            stats[5] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[-1])
            stats[6] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[-1])
            stats[7] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[-1])
            stats[8] = _summarize(2, areaRng='all', maxDets=self.params.maxDets[-1])
            stats[9] = _summarize(2, areaRng='small', maxDets=self.params.maxDets[-1])
            stats[10] = _summarize(2, areaRng='medium', maxDets=self.params.maxDets[-1])
            stats[11] = _summarize(2, areaRng='large', maxDets=self.params.maxDets[-1])
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        self.stats = summarize()


class Params:
    '''
    Params for icdar evaluation api
    '''

    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = [0.5]
        self.recThrs = [0.0]
        self.maxDets = [1000]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None