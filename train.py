'''
PointGroup train.py
Written by Li Jiang
'''

import torch
import torch.optim as optim
import time, sys, os, random
from tensorboardX import SummaryWriter
import numpy as np

from util.config import cfg
from util.log import logger
import util.utils as utils
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

def init():
    # copy important files to backup
    backup_dir = os.path.join(cfg.exp_path, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp train.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    # log the config
    logger.info(cfg)

    # summary writer
    global writer
    writer = SummaryWriter(cfg.exp_path)

    # random seed
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def train_epoch(train_loader, model, model_fn, optimizer, epoch):
    iter_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    am_dict = {}

    model.train()
    start_epoch = time.time()
    end = time.time()
    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        torch.cuda.empty_cache()

        ###### adjust learning rate
        # utils.step_learning_rate(optimizer, cfg.lr, epoch - 1, cfg.step_epoch, cfg.multiplier)

        ##### prepare input and forward
        loss, _, visual_dict, meter_dict = model_fn(batch, model, epoch)

        ##### meter_dict
        for k, v in meter_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = utils.AverageMeter()
            am_dict[k].update(v)

        ##### backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ##### time and print
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        max_iter = cfg.epochs * len(train_loader)
        remain_iter = max_iter - current_iter

        iter_time.update(time.time() - end)
        end = time.time()

        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        log_str = "epoch: {}/{} iter: {}/{} data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {remain_time} ".format(
            epoch, cfg.epochs, i + 1, len(train_loader),
            data_time.val, data_time.avg, iter_time.val, iter_time.avg, remain_time=remain_time)
        log_str += "lr: {} ".format(get_lr(optimizer))
        for k, v in am_dict.items():
            log_str += '{}: {:.4f}({:.4f}) '.format(k, v.val, v.avg)
        logger.info(log_str)

        if (i == len(train_loader) - 1): print()

        for k, v in meter_dict.items():
            writer.add_scalar(k, v, current_iter)
        writer.add_scalar('lr', get_lr(optimizer), current_iter)
        writer.flush()

def eval_epoch(val_loader, model, model_fn, epoch):
    gts = []
    preds = []
    with torch.no_grad():
        model = model.eval()
        for i, batch in enumerate(tqdm(val_loader)):
            N = batch['feats'].shape[0]
            # pred_list:  for each scan:
            #                 for each instance
            #                     instance = dict(scan_id, label_id, mask, confidence)
            masks, scores, semantic_pred = model_fn(batch, model, 1)

            pred = [] 
            for j in range(scores.shape[0]):
                p = {}
                p['scan_id'] = i
                label_id = 1
                p['conf'] = scores[j]
                p['label_id'] = label_id
                gt_labels = batch['labels'].numpy()
                semantic_inds = (semantic_pred != 0) & (semantic_pred != 1) & (semantic_pred != 20)
                mask = masks[j]
                p['pred_mask'] = mask & semantic_inds
                pred.append(p)

            gt = ((batch['labels'] - 2 + 1) * 1000 + batch['instance_labels']).numpy()
            gts.append(gt)
            preds.append(pred)
    from evaluate_semantic_instance import ScanNetEval
    eval = ScanNetEval(use_label=False)
    results = eval.evaluate(preds, gts)
    for k, v in results.items():
        if isinstance(v, float):
            writer.add_scalar(k, v, epoch)
    writer.flush()


if __name__ == '__main__':
    ##### init
    init()

    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    ##### model
    logger.info('=> creating model ...')

    if model_name == 'pointgroup':
        from model.pointgroup.pointgroup import PointGroup as Network
        from model.pointgroup.pointgroup import model_fn_decorator
    else:
        print("Error: no model - " + model_name)
        exit(0)

    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    # logger.info(model)
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### optimizer
    if cfg.optim == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optim == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    ##### model_fn (criterion)
    model_fn = model_fn_decorator()
    test_model_fn = model_fn_decorator(test=True)

    ##### dataset
    if cfg.dataset == 'scannetv2':
        if data_name == 'scannet':
            import data.scannetv2_inst
            dataset = data.scannetv2_inst.Dataset()
            dataset.trainLoader()
            dataset.valLoader()
        else:
            print("Error: no data loader - " + data_name)
            exit(0)

    ##### resume
    start_epoch = utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda)
    scheduler = MultiStepLR(optimizer, milestones=cfg.milestones, gamma=0.5)

    ##### train and val
    for epoch in range(start_epoch, cfg.epochs + 1):
        # eval_epoch(dataset.val_data_loader, model, test_model_fn, epoch)
        train_epoch(dataset.train_data_loader, model, model_fn, optimizer, epoch)
        scheduler.step()

        if (epoch % cfg.save_freq == 0) or epoch == cfg.epochs:
            utils.checkpoint_save(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], epoch, cfg.save_freq, use_cuda)
            eval_epoch(dataset.val_data_loader, model, test_model_fn, epoch)
