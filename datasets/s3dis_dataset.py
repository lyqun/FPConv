import os
import numpy as np
import sys
from torch.utils.data import Dataset


class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0, transform=None, if_normal=True):
        super().__init__()
        print('Initiating DataLoader....')
        self.if_normal = if_normal
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [
                room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [
                room for room in rooms if 'Area_{}'.format(test_area) in room]
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        for room_name in rooms_split:
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            # xyzrgb, N*6; l, N
            points, labels = room_data[:, 0:6], room_data[:, 6]
            points[:, 0:3] -= np.amin(points, axis=0)[0:3]

            coord_min, coord_max = np.amin(points, axis=0)[
                :3], np.amax(points, axis=0)[:3]

            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(
                coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)

        # Generate label weights
        self.labelweights = self.__gen_labelweights(self.room_labels)

        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend(
                [index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        np.random.seed(123)
        np.random.shuffle(self.room_idxs)

        print('Num of labels: ', len(self.room_labels))
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __gen_labelweights(self, labels):
        labelweights = np.zeros(13)
        for seg in labels:
            tmp, _ = np.histogram(seg, range(14))
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        # self.labelweights = 1/np.log(1.2+labelweights)
        return np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        while True:
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size /
                                  2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size /
                                  2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (
                points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(
                point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(
                point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        if self.if_normal:
            current_points = np.zeros((self.num_point, 9))  # num_point * 9
            current_points[:, 6] = selected_points[:, 0] / \
                self.room_coord_max[room_idx][0]
            current_points[:, 7] = selected_points[:, 1] / \
                self.room_coord_max[room_idx][1]
            current_points[:, 8] = selected_points[:, 2] / \
                self.room_coord_max[room_idx][2]
            current_points[:, 0:6] = selected_points
        else:
            current_points = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(
                current_points, current_labels)

        sampleweights = self.labelweights[current_labels.astype(np.uint8)]
        return current_points, current_labels, sampleweights

    def __len__(self):
        return len(self.room_idxs)


if __name__ == '__main__':
    data_root = '/home/zizheng/data/s3dis/stanford_indoor3d_all_classes'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    import psutil
    print("Before loading, the memory usage is ", psutil.virtual_memory())
    point_data = S3DIS(split='train', data_root=data_root, num_point=num_point,
                       test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch
    import time
    import random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    print("After loading, the memory usage is ", psutil.virtual_memory())

    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(
        point_data, batch_size=32, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (points, target, weight) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i + 1,
                                           len(train_loader), time.time() - end))
            print('Size of points: ', points.size())
            points_np = points.cpu().data.numpy()
            points_np_block1 = points_np[0, ...]
            minp = points_np_block1[:, 0].min()
            maxp = points_np_block1[:, 0].max()
            print('weight is ', weight)
            print('Min in x is {}, Max in x is {}'.format(minp, maxp))
            print('Min in y is {}, Max in y is {}'.format(
                points_np_block1[:, 1].min(), points_np_block1[:, 1].max()))

            print("In loop, the memory usage is ", psutil.virtual_memory())
            sys.exit(0)
