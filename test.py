'''
PointGroup test.py
Written by Li Jiang
'''

import torch
import time
import numpy as np
import random
import os

from util.config import cfg
from util.log import logger
import util.utils as utils
from tqdm import tqdm


cfg.task = 'test'

def init():
    global result_dir
    result_dir = os.path.join(cfg.exp_path, 'result', 'epoch{}_nmst{}_scoret{}_npointt{}'.format(cfg.test_epoch, cfg.TEST_NMS_THRESH, cfg.TEST_SCORE_THRESH, cfg.TEST_NPOINT_THRESH), cfg.split)
    backup_dir = os.path.join(result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'predicted_masks'), exist_ok=True)
    os.system('cp test.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    global semantic_label_idx
    semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)

if __name__ == '__main__':
    init()

    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    ##### model
    logger.info('=> creating model ...')
    logger.info('Classes: {}'.format(cfg.classes))

    if model_name == 'pointgroup':
        from model.pointgroup.pointgroup import PointGroup as Network
        from model.pointgroup.pointgroup import model_fn_decorator
    else:
        print("Error: no model version " + model_name)
        exit(0)
    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    # logger.info(model)
    logger.info('#classifier parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### model_fn (criterion)
    model_fn = model_fn_decorator(test=True)

    ##### load model
    utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda, cfg.test_epoch, dist=False, f=cfg.pretrain)

    ##### evaluate
    if cfg.dataset == 'scannetv2':
        if data_name == 'scannet':
            from data.scannetv2_inst import Dataset
            dataset = Dataset(test=True)
            dataset.valLoader()
        else:
            print("Error: no data loader - " + data_name)
            exit(0)
    dataloader = dataset.val_data_loader

    gts = []
    preds = []
    total_time = 0
    with torch.no_grad():
        model = model.eval()
        for i, batch in enumerate(tqdm(dataloader)):
            N = batch['feats'].shape[0]
            # pred_list:  for each scan:
            #                 for each instance
            #                     instance = dict(scan_id, label_id, mask, confidence)
            since = time.time()
            masks, scores, semantic_pred = model_fn(batch, model, 1)
            total_time += time.time() - since 
            pred = [] 
            for j in range(scores.shape[0]):
                for k in range(1, 19):
                    p = {}
                    p['scan_id'] = i
                    p['conf'] = scores[j]
                    gt_labels = batch['labels'].numpy()
                    fg_inds = (semantic_pred != 0) & (semantic_pred != 1) & (semantic_pred != 20)
                    mask = masks[j]
                    fg_inds = fg_inds & mask
                    pred_class = k
                    p['label_id'] = k
                    p['pred_mask'] = mask & (semantic_pred == (k + 1))
                    pred.append(p)
            gt = ((batch['labels'] - 2 + 1) * 1000 + batch['instance_labels']).numpy()
            gts.append(gt)
            preds.append(pred)
    from evaluate_semantic_instance import ScanNetEval
    eval = ScanNetEval(use_label=True)
    eval.evaluate(preds, gts)
    print('time', total_time / len(dataloader))
