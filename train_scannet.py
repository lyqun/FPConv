import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist

import os, sys
import argparse
import importlib
import numpy as np
import json
import tensorboard_logger as tb_log

from datasets.scannet_dataset_rgb import ScannetDataset, ScannetDatasetWholeScene
from utils.saver import Saver
from utils.switchnorm import convert_sn

np.seterr(divide='ignore', invalid='ignore')

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument("--gpu", type=str, default='6,7')
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument('--workers', type=int, default=12)
parser.add_argument("--mode", type=str, default='train')

parser.add_argument("--model", type=str, default='fpcnn_scannet_tiny_v3')
parser.add_argument("--save_dir", type=str, default='logs/test_scannet_tiny')
parser.add_argument("--config", type=str, default='./config.json')
parser.add_argument("--use_sn", action='store_true', default=False)

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay', type=float, default=0.1)
parser.add_argument('--lr_clip', type=float, default=0.000001)
parser.add_argument('--decay_step_list', type=list, default=[100, 200, 300])
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument("--resume", type=str, default=None)

parser.add_argument("--sample_rate", type=float, default=None)
parser.add_argument("--with_rgb", action='store_true', default=False)
parser.add_argument("--with_norm", action='store_true', default=False)
parser.add_argument("--num_points", type=int, default=8192)
parser.add_argument("--accum", type=int, default=24)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# load config file
with open(args.config, 'r') as f:
    _cfg = json.load(f)


NUM_CLASSES = 21
NUM_POINTS = args.num_points
saver = Saver(args.save_dir, max_files=100)
print(args)
print(_cfg)


def log_str(info):
    print(info)


def load_checkpoint(model, filename):
    if os.path.isfile(filename):
        log_str("==> Loading from checkpoint %s" % filename)
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state'])
        log_str("==> Done")
    else:
        raise FileNotFoundError

    return epoch


class CrossEntropyLossWithWeights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, predict, target, weights):
        """
        :param predict: (B,N,C)
        :param target: (B,N)
        :param weights: (B,N)
        :return:
        """
        predict = predict.view(-1, NUM_CLASSES).contiguous() # B*N, C
        target = target.view(-1).contiguous().cuda().long()  # B*N
        weights = weights.view(-1).contiguous().cuda().float() # B*N

        loss = self.cross_entropy_loss(predict, target) # B*N
        loss *= weights
        loss = torch.mean(loss)
        return loss


def train_one_epoch(model, dst_loader, optimizer, epoch, tb_log):
    model.train()
    loss_func = CrossEntropyLossWithWeights()

    repeat = args.accum // args.batch_size
    log_str(' --- train, accumulate gradients for {} times. Total bacth size is {}.'.format(repeat, args.accum))

    loss_list = []
    loss_temp_list = []
    correct_temp = 0
    seen_temp = 0
    total_correct = 0
    total_seen = 0
    optimizer.zero_grad()

    # for it, batch in tqdm(enumerate(dst_loader)):
    for it, batch in enumerate(dst_loader):

        point_set, semantic_seg, sample_weight = batch
        point_set = point_set.cuda().float()
        predict = model(point_set) # B,N,C

        loss = loss_func(predict, semantic_seg, sample_weight)
        loss_norm = loss / repeat
        loss_norm.backward()

        # accumulate gradient
        if (it + 1) % repeat == 0 or (it + 1) == len(dst_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # 1. loss
        loss_list.append(loss.item())

        # 2. accuracy
        predict = torch.argmax(predict, dim=2).cpu().numpy() # B,N
        semantic_seg = semantic_seg.numpy()
        correct = np.sum(predict == semantic_seg)
        batch_seen = predict.shape[0] * NUM_POINTS
        total_correct += correct
        total_seen += batch_seen

        # save temp data
        loss_temp_list.append(loss.item())
        correct_temp += correct
        seen_temp += batch_seen

        if (it + 1) % 100 == 0:
            log_str(' -- batch: {}/{} -- '.format(it+1, len(dst_loader)))
            log_str('accuracy: {:.4f}'.format(correct_temp / seen_temp))
            log_str('mean loss: {:.4f}'.format(np.mean(loss_temp_list)))
            loss_temp_list = []
            correct_temp = 0
            seen_temp = 0

    log_str(' -- epoch accuracy: {:.4f}'.format(total_correct / total_seen))
    log_str(' -- epoch mean loss: {:.4f}'.format(np.mean(loss_list)))
    
    if epoch % 5 == 0:
        tb_log.log_value('epoch oA', total_correct / total_seen, epoch)
        tb_log.log_value('epoch loss', np.mean(loss_list), epoch)


def eval_one_epoch(model, dst_loader, epoch, tb_log):
    model.eval()

    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

    loss_func = CrossEntropyLossWithWeights()
    loss_list = []
    with torch.no_grad():
        for it, batch in enumerate(dst_loader):
            batch_data, batch_label, batch_smpw = batch
            batch_data = batch_data.cuda().float()
            pred_val = model(batch_data) # B,N,C

            loss = loss_func(pred_val, batch_label, batch_smpw)
            loss_list.append(loss.item())
            
            # convert to numpy array
            pred_val = torch.argmax(pred_val, dim=2).cpu().numpy() # B,N
            batch_label = batch_label.numpy()
            batch_smpw = batch_smpw.numpy()

            correct = np.sum((pred_val == batch_label) & (batch_label>0) & (batch_smpw>0))
            total_correct += correct
            total_seen += np.sum((batch_label>0) & (batch_smpw>0))

            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label==l) & (batch_smpw>0))
                total_correct_class[l] += np.sum((pred_val==l) & (batch_label==l) & (batch_smpw>0))
                total_iou_deno_class[l] += np.sum(((pred_val==l) | (batch_label==l)) & (batch_smpw>0) & (batch_label>0))

    IoU = np.array(total_correct_class[1:])/(np.array(total_iou_deno_class[1:],dtype=np.float)+1e-6)
    log_str('eval point avg class IoU: %f' % (np.mean(IoU)))
    IoU_Class = 'Each Class IoU:::\n'
    for i in range(IoU.shape[0]):
        log_str('Class %d : %.4f'%(i+1, IoU[i]))
    log_str('eval loss: %f'% (np.mean(loss_list)))
    log_str('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_str('eval avg class acc: %f' % (np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6))))
    
    tb_log.log_value('Eval loss', np.mean(loss_list), epoch)
    tb_log.log_value('Eval mIoU', np.mean(IoU), epoch)
    tb_log.log_value('Eval oA', total_correct / float(total_seen), epoch)
    tb_log.log_value('Eval mA', np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6)), epoch)
    return np.mean(IoU)


def train(model, train_loader, eval_loader, tb_log, resume_epoch=0):
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr, 
                                momentum=0.98, 
                                weight_decay=args.weight_decay, 
                                nesterov=True)

    # init lr scheduler
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in args.decay_step_list:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * args.lr_decay
        return max(cur_decay, args.lr_clip / args.lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd)

    best_miou = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        # resume training epoch
        if epoch < resume_epoch: continue
        elif resume_epoch > 0:
            log_str('====== resume epoch {} ======'.format(epoch))
            log_str('====== Evaluation ======')
            miou = eval_one_epoch(model, eval_loader, epoch, tb_log)
            if miou > best_miou:
                best_miou = miou
                best_epoch = epoch
            log_str(' === Best mIoU: {}, epoch {}. === '.format(best_miou, best_epoch))
            lr_scheduler.step(epoch)
            resume_epoch = 0
            continue
        
        # training
        log_str('====== epoch {} ======'.format(epoch))
        train_one_epoch(model, train_loader, optimizer, epoch, tb_log)
        lr_scheduler.step(epoch)

        # evaluate model
        if epoch % 20 == 0: # epoch >= 30 and epoch % 5 == 0:
            # saver.save_checkpoint(model, epoch, 'saved_ckpt_{}'.format(epoch))
            log_str('====== Evaluation ======')
            miou = eval_one_epoch(model, eval_loader, epoch, tb_log)
            if miou > best_miou:
                best_miou = miou
                best_epoch = epoch
                saver.save_checkpoint(model, epoch, 'pn2_best_epoch_{}'.format(epoch))
        log_str(' === Best mIoU: {}, epoch {}. === '.format(best_miou, best_epoch))
        
if __name__ == '__main__':
    input_channels = 0
    if args.with_rgb: input_channels += 3
    if args.with_norm: input_channels += 3
    print('model input_channel: {}.'.format(input_channels))

    # model init
    MODEL = importlib.import_module('models.' + args.model)
    model = MODEL.get_model(num_class=NUM_CLASSES, input_channels=input_channels, num_pts=args.num_points)
    if args.use_sn:
        print(' --- use sn')
        model = utils.convert_sn(model)

    # resume
    from_epoch = 0
    if args.resume:
        from_epoch = load_checkpoint(model, args.resume)
    model = nn.parallel.DataParallel(model)
    model.cuda()

    # init tb_log
    tb_log.configure(os.path.join(args.save_dir, 'tensorboard'))

    # eval dataloader
    eval_dst = ScannetDatasetWholeScene(root=_cfg['scannet_pickle'], 
                                        npoints=NUM_POINTS, 
                                        split='eval',
                                        with_norm=args.with_norm,
                                        with_rgb=args.with_rgb)
    eval_loader = DataLoader(eval_dst, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    # train dataloader
    train_dst = ScannetDataset(root=_cfg['scannet_pickle'], 
                               npoints=NUM_POINTS, 
                               split='train' if args.mode == 'train' else 'eval', 
                               with_dropout=True, 
                               with_norm=args.with_norm,
                               with_rgb=args.with_rgb,
                               sample_rate=args.sample_rate)
    train_loader = DataLoader(train_dst, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              pin_memory=True, 
                              num_workers=args.workers, 
                              drop_last=True) # sync_bn will raise an unknown error with batch size of 1. 
    train(model, train_loader, eval_loader, tb_log, from_epoch)