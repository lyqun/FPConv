import os, sys
sys.path.insert(0, "/home/densechen/code/FPConv")
import argparse
import importlib
import numpy as np
import json
import time
import tensorboard_logger as tb_log

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.s3dis_dataset import S3DIS
from utils.saver import Saver

np.seterr(divide='ignore', invalid='ignore')

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument("--gpu", type=str, default='0,1,2,3')
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=101)
parser.add_argument('--workers', type=int, default=8)

parser.add_argument('--num_classes', type=int, default=13)
parser.add_argument('--eval_freq', type=int, default=10)
parser.add_argument('--start_eval_epoch', type=int, default=0)
parser.add_argument("--accum_steps", type=int, default=8)

parser.add_argument('--sample_rate_eval', type=float, default=1)
parser.add_argument('--sample_rate_train', type=float, default=0.5)
parser.add_argument('--num_pts', type=int, default=14564)
parser.add_argument('--block_size', type=float, default=2)
parser.add_argument('--test_area', type=int, default=5)

parser.add_argument("--model", type=str, default='fpcnn_s3dis')
parser.add_argument("--save_dir", type=str, default='logs/test_s3dis/')
parser.add_argument("--config", type=str, default='./config.json')
parser.add_argument('--bn_momentum', type=float, default=0.02)

parser.add_argument('--warmup_epochs', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay', type=float, default=0.5)
parser.add_argument('--lr_clip', type=float, default=0.000001)
parser.add_argument('--decay_step_list', type=list, default=[25, 50, 75])
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument("--resume", type=str, default=None)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(args)

with open(args.config, 'r') as f:
    _cfg = json.load(f)
    print(_cfg)


NUM_CLASSES = args.num_classes
NUM_POINTS = args.num_pts
saver = Saver(args.save_dir)

class WarmStart:
    """Warm up learning rate"""

    def __init__(self, optimizer, steps, lr):
        '''
        steps: Warm up steps, if it is 0, warm up is not activated
        lr: target learning rate
        '''
        self.optimizer = optimizer
        self.steps = steps
        self.iter = 0
        if steps != 0:
            self.increment = lr / steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0

    def step(self):
        self.iter += 1
        if self.iter < self.steps:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.iter * self.increment


def log_str(info):
    print(info)


def reset_bn(model, momentum=args.bn_momentum):
    '''
    Reset bn momentum
    '''
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def save_config(args, _cfg):
    '''
    Save configs currently using, along with the checkpoints, etc.
    '''
    f1 = os.path.join(args.save_dir, 'args.txt')
    f2 = os.path.join(args.save_dir, 'configs.txt')

    with open(f1, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    with open(f2, 'w') as f:
        json.dump(_cfg, f, indent=2)


def load_checkpoint(model, filename):
    if os.path.isfile(filename):
        print("==> Loading from checkpoint %s" % filename)
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state'])
        print("==> Done")
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
        predict = predict.view(-1, NUM_CLASSES).contiguous()  # B*N, C
        target = target.view(-1).contiguous().cuda().long()  # B*N
        weights = weights.view(-1).contiguous().cuda().float()  # B*N

        loss = self.cross_entropy_loss(predict, target)  # B*N
        loss *= weights
        loss = torch.mean(loss)
        return loss


def train_one_epoch(model, dst_loader, optimizer, epoch, tb_log, warmup=None):
    model.train()
    loss_func = CrossEntropyLossWithWeights()
    optimizer.zero_grad()

    loss_list = []
    total_correct = 0
    total_seen = 0
    start_time = time.time()
    for it, batch in enumerate(dst_loader):

        point_set, semantic_seg, sample_weight = batch
        point_set = point_set.cuda().float()
        predict = model(point_set)  # B,N,C

        loss = loss_func(predict, semantic_seg, sample_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if (it + 1) % args.accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if epoch <= args.warmup_epochs:
                warmup.step()

        # 1. loss
        loss_list.append(loss.item())

        # 2. accuracy
        predict = torch.argmax(predict, dim=2).cpu().numpy()  # B,N
        semantic_seg = semantic_seg.numpy()
        correct = np.sum(predict == semantic_seg)
        batch_seen = predict.shape[0] * NUM_POINTS
        total_correct += correct
        total_seen += batch_seen

        if (it + 1) % 100 == 0:
            time_cost = time.time() - start_time
            log_str(' -- batch: {}/{} -- '.format(it + 1, len(dst_loader)))
            log_str('accuracy: {:.4f}'.format(total_correct / total_seen))
            log_str('mean loss: {:.4f}'.format(np.mean(loss_list)))
            log_str('time cost: {:.2f}'.format(time_cost))
            start_time = time.time()
            iternum = epoch * len(dst_loader) + it + 1
            tb_log.log_value('Train/IterAcc', total_correct /
                             total_seen, iternum)
            tb_log.log_value('Train/IterLoss', np.mean(loss_list), iternum)
            tb_log.log_value('Train/Learning rate',
                             optimizer.param_groups[0]['lr'], iternum)

    log_str(' -- epoch accuracy: {:.4f}'.format(total_correct / total_seen))
    log_str(' -- epoch mean loss: {:.4f}'.format(np.mean(loss_list)))
    tb_log.log_value('Train/epoch oA', total_correct / total_seen, epoch)
    tb_log.log_value('Train/epoch loss', np.mean(loss_list), epoch)
    lr = optimizer.param_groups[0]['lr']
    tb_log.log_value('Learning rate', lr, epoch)


def eval_one_epoch(model, dst_loader, epoch, tb_log):
    model.eval()
    loss_func = CrossEntropyLossWithWeights()

    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

    loss_list = []
    with torch.no_grad():
        for it, batch in enumerate(dst_loader):
            batch_data, batch_label, batch_smpw = batch
            batch_data = batch_data.cuda().float()
            pred_val = model(batch_data)  # B,N,C

            loss = loss_func(pred_val, batch_label, batch_smpw)
            loss_list.append(loss.item())

            pred_val = torch.argmax(pred_val, dim=2).cpu().numpy()  # B,N
            batch_label = batch_label.numpy()
            batch_smpw = batch_smpw.numpy()
            aug_data = batch_data.cpu().numpy()

            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += np.sum((batch_label >= 0) & (batch_smpw > 0))

            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label == l)
                                              & (batch_smpw > 0))
                total_correct_class[l] += np.sum((pred_val == l)
                                                 & (batch_label == l) & (batch_smpw > 0))
                total_iou_deno_class[l] += np.sum(
                    ((pred_val == l) | (batch_label == l)) & (batch_smpw > 0))

    
    IoU = np.array(total_correct_class /
                   np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
    avg_acc = np.mean(np.array(total_correct_class) /
                      (np.array(total_seen_class, dtype=np.float) + 1e-6))

    if np.mean(loss_list) > 20:
        saver.save_checkpoint(
            model, epoch, 'loss_explosion_epoch_{}'.format(epoch))

    log_str('eval point avg class IoU: %f' % (np.mean(IoU)))
    IoU_Class = 'Each Class IoU:::\n'
    for i in range(IoU.shape[0]):
        log_str('Class %d : %.4f' % (i + 1, IoU[i]))
    log_str('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_str('eval avg class acc: %f' % (avg_acc))

    tb_log.log_value('Eval/mIoU', np.mean(IoU), epoch)
    tb_log.log_value('Eval/oA', total_correct / float(total_seen), epoch)
    tb_log.log_value('Eval/mA', avg_acc, epoch)
    tb_log.log_value('Eval/Loss', np.mean(loss_list), epoch)
    return np.mean(IoU)


def train(model, train_loader, eval_loader, tb_log, resume_epoch=0):
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr, 
                                momentum=0.98, 
                                weight_decay=args.weight_decay, 
                                nesterov=True)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    warmup_steps = int(args.warmup_epochs * len(train_loader) / args.accum_steps)
    warmup = WarmStart(optimizer, warmup_steps, args.lr)

    best_miou = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        if epoch < resume_epoch:
            continue
        elif resume_epoch > 0:
            log_str('====== resume epoch {} ======'.format(epoch))
            log_str('====== Evaluation ======')
            miou = eval_one_epoch(model, eval_loader, epoch, tb_log)
            if miou > best_miou:
                best_miou = miou
                best_epoch = epoch
            log_str(' === Best mIoU: {}, epoch {}. === '.format(
                best_miou, best_epoch))
            if epoch >= args.warmup_epochs:
                lr_scheduler.step(epoch)

            resume_epoch = 0
            continue

        log_str('====== epoch {} ======'.format(epoch))
        train_one_epoch(model, train_loader, optimizer, epoch, tb_log, warmup)

        if epoch >= args.warmup_epochs:
            lr_scheduler.step(epoch)

        if (epoch >= args.start_eval_epoch and epoch % args.eval_freq == 0) or \
           (epoch > 80 and epoch % 2 == 0):
            log_str('====== Evaluation ======')
            miou = eval_one_epoch(model, eval_loader, epoch, tb_log)
            if miou > best_miou:
                best_miou = miou
                best_epoch = epoch
            saver.save_checkpoint(
                model, epoch, 'pn2_best_epoch_{}'.format(epoch))
        log_str(' === Best mIoU: {}, epoch {}. === '.format(
            best_miou, best_epoch))


if __name__ == '__main__':
    save_config(args, _cfg)
    MODEL = importlib.import_module('models.' + args.model)

    input_channels = 6
    print('model input_channel: {}.'.format(input_channels))
    model = MODEL.get_model(num_class=NUM_CLASSES, input_channels=input_channels)

    reset_bn(model)

    # resume
    from_epoch = 0
    if args.resume:
        from_epoch = load_checkpoint(model, args.resume)
    print("resume from {}".format(from_epoch))
    model = nn.parallel.DataParallel(model)
    model.cuda()

    # init tb_log
    tb_log.configure(os.path.join(args.save_dir, 'tensorboard'))
    
    # Sample rate is the num of voting
    eval_dst = S3DIS(split='eval', 
                    data_root=_cfg['s3dis_data_root'], 
                    num_point=args.num_pts,
                    test_area=args.test_area, 
                    block_size=args.block_size, 
                    sample_rate=args.sample_rate_eval, 
                    transform=None, 
                    if_normal=True)
    eval_loader = DataLoader(eval_dst, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    
    train_dst = S3DIS(split='train', 
                      data_root=_cfg['s3dis_data_root'], 
                      num_point=args.num_pts,
                      test_area=args.test_area, 
                      block_size=args.block_size, 
                      sample_rate=args.sample_rate_train, 
                      transform=None, 
                      if_normal=True)
    train_loader = DataLoader(train_dst, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    # train model
    train(model, train_loader, eval_loader, tb_log, from_epoch)
