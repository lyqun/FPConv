import os, sys
sys.path.insert(0, "/home/densechen/code/FPConv")
import json
import numpy as np
import argparse
import importlib

import torch
from torch.utils.data import DataLoader
from datasets.s3dis_dataset_test import S3DISWholeScene_evaluation

np.seterr(divide='ignore', invalid='ignore')

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--model", type=str, default='fpcnn_s3dis')

parser.add_argument("--stride", type=float, default=0.5)
parser.add_argument("--block_size", type=float, default=2)
parser.add_argument("--test_area", type=int, default=5)
parser.add_argument("--num_pts", type=int, default=14564)

parser.add_argument("--weight_dir", type=str, default=None) # checkpoint path
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--config", type=str, default='./config.json')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(args)

with open(args.config, 'r') as f:
    _cfg = json.load(f)
    print(_cfg)

SEM_LABELS = None
NUM_CLASSES = 13
NUM_POINTS = args.num_pts

class_name_path = os.path.join('utils/s3dis_meta/class_names.txt')
g_classes = [x.rstrip() for x in open(class_name_path)]
class_dict = np.arange(13)
if args.save_dir is not None:
    os.makedirs(args.save_dir, exist_ok=True)


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


def vote(predict, pred, points_idx, vote_num_point):
    ''' numpy array
    :param predict: (pn,21) float
    :param pred: (bs,np,21) float
    :param points_idx: (bs,np) int
    :param vote_num_point: (pn, 1) times that points are overlapped
    '''
    bs, np = points_idx.shape
    for i in range(bs):
        for j in range(np):
            pred_ = pred[i, j, :]  # 21
            pidx_ = points_idx[i, j]  # int
            predict[pidx_, :] += pred_
    return predict


def write_to_file(path, labels):
    '''
    :param path: path to save predicted label
    :param labels: n (list => int)
    '''
    np.save(path, labels)


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
        scene_id = scene_list[scene_index]
        point_num = pn_list[scene_index]
        predict = np.zeros((point_num, NUM_CLASSES), dtype=np.float32)  # pn,21
        vote_num_point = np.zeros((point_num, 1), dtype=np.float32)
        for batch_data in dst_loader: # tqdm(dst_loader):
            pc, seg, pidx = batch_data
            pc = pc.cuda().float()
            with torch.no_grad():
                pred = model(pc)  # B,N,C
            pred = torch.softmax(pred, dim=2)
            pred = pred.cpu().detach().numpy()
            seg = seg.data.numpy()
            pidx = pidx.numpy()  # B,N
            predict = vote(predict, pred, pidx, vote_num_point)
        predict = np.argmax(predict, axis=1)

        # Save predictions
        if args.save_dir is not None:
            save_path = os.path.join(args.save_dir, scene_id)
            write_to_file(save_path, predict)
            print('Save predicted label to {}.'.format(save_path))

        labels = SEM_LABELS[scene_index]
        total_seen += np.sum(labels >= 0)  # point_num
        total_correct += np.sum((predict == labels) & (labels >= 0))
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum((labels == l) & (labels >= 0))
            total_correct_class[l] += np.sum((predict == l)
                                             & (labels == l))
            total_iou_deno_class[l] += np.sum(
                ((predict == l) & (labels >= 0)) | (labels == l))

        print('Batch eval accuracy: %f' %
              (total_correct / float(total_seen)))

    IoU = np.array(
        total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
    print('eval point avg class IoU: %f' % (np.mean(IoU)))
    for i in range(IoU.shape[0]):
        print('%s : %.4f' % (g_classes[i], IoU[i]))
    
    print('eval accuracy: %f' % (total_correct / float(total_seen)))
    print('eval avg class acc: %f' % (np.mean(np.array(
          total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))


if __name__ == '__main__':
    # Initialize Model and Data Loader
    MODEL = importlib.import_module('models.' + args.model)
    model = MODEL.get_model(num_class=NUM_CLASSES, input_channels=6)
    load_checkpoint(model, args.weight_dir)
    model = torch.nn.parallel.DataParallel(model)
    model.cuda()

    test_dst = S3DISWholeScene_evaluation(root=_cfg['s3dis_data_root'],
                                          split='test',
                                          test_area=args.test_area,
                                          block_points=NUM_POINTS,
                                          block_size=args.block_size,
                                          stride=args.stride,
                                          with_rgb=True)
    pn_list = test_dst.point_num
    scene_list = test_dst.scene_list
    SEM_LABELS = test_dst.semantic_labels_list

    test_loader = DataLoader(test_dst, batch_size=args.batch_size,
                             shuffle=False, pin_memory=True, num_workers=0)
    test(model, test_loader, pn_list, scene_list)
