import torch
from torch.utils.data import DataLoader

import numpy as np
import argparse
import importlib
import os
import sys
import json
from utils.switchnorm import convert_sn

from datasets.scannet_dataset_rgb_test import ScannetDatasetWholeScene_evaluation
np.seterr(divide='ignore', invalid='ignore')

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument("--gpu", type=str, default='6,7')
parser.add_argument("--batch_size", type=int, default=48)

parser.add_argument("--with_rgb", action='store_true', default=False)
parser.add_argument("--with_norm", action='store_true', default=False)
parser.add_argument("--use_sn", action='store_true', default=False)

parser.add_argument("--model", type=str, default='fpcnn_scannet_tiny_v3')
parser.add_argument("--weight_dir", type=str, default=None)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--config", type=str, default='./config.json')
parser.add_argument("--skip_exist", type=bool, default=False)
parser.add_argument("--num_points", type=int, default=8192)
parser.add_argument("--mode", type=str, default='eval')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(args)

# load config files
with open(args.config, 'r') as f:
    _cfg = json.load(f)
    print(_cfg)


NUM_CLASSES = 21
NUM_POINTS = args.num_points # 8192 # 10240 + 1024
SEM_LABELS = None
class_dict = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]) # 21 (0: unknown)


def load_checkpoint(model, filename):
    """
    Load model from file.

    Args:
        model: (todo): write your description
        filename: (str): write your description
    """
    if os.path.isfile(filename):
        print("==> Loading from checkpoint %s" % filename)
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state'])
        print("==> Done")
    else:
        print(filename)
        raise FileNotFoundError
    return epoch


def vote(predict, vote_num, pred, points_idx):
    ''' numpy array
    :param predict: (pn,21) float
    :param vote_num: (pn,1) int
    :param pred: (bs,np,21) float
    :param points_idx: (bs,np) int
    '''
    bs, np = points_idx.shape
    for i in range(bs):
        for j in range(np):
            pred_ = pred[i, j, :] # 21
            pidx_ = points_idx[i, j] # int
            predict[pidx_, :] += pred_
            vote_num[pidx_, 0] += 1
    return predict, vote_num


def write_to_file(path, probs):
    '''
    :param path: path to save predicted label
    :param probs: N,22
    '''
    file_name = path + ('.txt' if args.mode == 'test' else '.npy')
    if args.skip_exist and os.path.isfile(file_name):
        print(' -- file exists, skip', file_name)
        return
    if args.mode == 'test':
        predict = np.argmax(probs[:, 1:], axis=1) # pn
        predict += 1
        predict = class_dict[predict]
        with open(file_name, 'w') as f:
            f.write(str(predict[0]))
            for pred in predict[1:]:
                f.write('\n{}'.format(pred))
    else:
        np.save(file_name, probs)
    print(' -- save file to ====>', file_name)


def test(model, dst_loader, pn_list, scene_list):
    '''
    :param pn_list: sn (list => int), the number of points in a scene
    :param scene_list: sn (list => str), scene id
    '''
    model.eval()
    total_seen = 0
    total_correct = 0
    total_seen_class = [0] * NUM_CLASSES
    total_correct_class = [0] * NUM_CLASSES
    total_iou_deno_class = [0] * NUM_CLASSES

    scene_num = len(scene_list)
    for scene_index in range(scene_num):
        print(' ======= {}/{} ======= '.format(scene_index, scene_num))
        # scene_index = 0
        scene_id = scene_list[scene_index]
        point_num = pn_list[scene_index]
        predict = np.zeros((point_num, NUM_CLASSES), dtype=np.float32) # pn,21
        vote_num = np.zeros((point_num, 1), dtype=np.int) # pn,1
        for batch_data in dst_loader:
            pc, seg, smpw, pidx= batch_data
            pc = pc.cuda().float()
            pred = model(pc) # B,N,C
            pred = torch.nn.functional.softmax(pred, dim=2)

            pred = pred.cpu().detach().numpy()
            pidx = pidx.numpy() # B,N
            predict, vote_num = vote(predict, vote_num, pred, pidx)


        predict = predict / vote_num
        if args.save_dir is not None:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_path = os.path.join(args.save_dir, '{}'.format(scene_id))
            write_to_file(save_path, predict)
        
        if args.mode != 'test':
            predict = np.argmax(predict[:, 1:], axis=1) # pn
            predict += 1
            labels = SEM_LABELS[scene_index]
            total_seen += np.sum(labels > 0) # point_num
            total_correct += np.sum((predict == labels) & (labels > 0))
            print('accuracy: ', total_correct / total_seen)
            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((labels == l) & (labels > 0))
                total_correct_class[l] += np.sum((predict == l) & (labels == l))
                total_iou_deno_class[l] += np.sum(((predict == l) & (labels > 0)) | (labels == l))

    if args.mode != 'test':
        IoU = np.array(total_correct_class[1:])/(np.array(total_iou_deno_class[1:],dtype=np.float)+1e-6)
        print('eval point avg class IoU: %f' % (np.mean(IoU)))
        IoU_Class = 'Each Class IoU:::\n'
        for i in range(IoU.shape[0]):
            print('Class %d : %.4f'%(i+1, IoU[i]))
        print('eval accuracy: %f'% (total_correct / float(total_seen)))
        print('eval avg class acc: %f' % (np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6))))


if __name__ == '__main__':
    input_channels = 0
    if args.with_rgb: input_channels += 3
    if args.with_norm: input_channels += 3
    # Initialize Model and Data Loader
    MODEL = importlib.import_module('models.' + args.model)
    model = MODEL.get_model(num_class=NUM_CLASSES, input_channels=input_channels, num_pts=args.num_points)
    if args.use_sn:
        print(' --- use sn')
        model = convert_sn(model)
    
    load_checkpoint(model, args.weight_dir)
    model.cuda()
    model = torch.nn.parallel.DataParallel(model)

    test_dst = ScannetDatasetWholeScene_evaluation(root=_cfg['scannet_pickle'], 
                                                   scene_list_dir=_cfg['scene_list'], 
                                                   split=args.mode, 
                                                   block_points=NUM_POINTS, 
                                                   with_rgb=args.with_rgb,
                                                   with_norm=args.with_norm)
    pn_list = test_dst.point_num
    scene_list = test_dst.scene_list
    SEM_LABELS = test_dst.semantic_labels_list

    test_loader = DataLoader(test_dst, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    with torch.no_grad():
        test(model, test_loader, pn_list, scene_list)
    