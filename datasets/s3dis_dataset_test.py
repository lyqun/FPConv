import pickle
import os
import sys
import numpy as np
import torch.utils.data as torch_data


class S3DISWholeScene_evaluation(torch_data.IterableDataset):
    # prepare to give prediction on each points
    def __init__(self, root=None, split='test', test_area=5, num_class=13, block_points=8192, block_size=1.5, stride=0.5, with_rgb=True):
        print('test area:', test_area)
        self.root = root
        self.split = split
        self.with_rgb = with_rgb
        self.block_points = block_points
        self.block_size = block_size
        self.stride = stride
        self.point_num = []
        self.temp_data = []
        self.temp_index = 0
        self.now_index = 0
        self.scene_points_list = []
        self.semantic_labels_list = []

        rooms = sorted(os.listdir(root))
        rooms = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        for room_name in rooms:
            room_path = os.path.join(root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            # xyzrgb, N*6; l, N
            points, labels = room_data[:, 0:6], room_data[:, 6]
            points[:, 0:3] -= np.amin(points, axis=0)[0:3]
            self.scene_points_list.append(points)
            self.semantic_labels_list.append(labels)

        self.scene_list = [i.replace('.npy', '') for i in rooms]

        for seg in self.semantic_labels_list:
            self.point_num.append(seg.shape[0])

    def reset(self):
        self.temp_data = []
        self.temp_index = 0
        self.now_index = 0

    def __iter__(self):
        if self.now_index >= len(self.scene_points_list):
            print(' ==== reset dataset index ==== ')
            self.reset()
        self.gen_batch_data()
        return self

    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def split_data(self, data, idx):
        new_data = []
        for i in range(len(idx)):
            new_data += [data[idx[i]]]
        return new_data

    def nearest_dist(self, block_center, block_center_list):
        num_blocks = len(block_center_list)
        dist = np.zeros(num_blocks)
        for i in range(num_blocks):
            dist[i] = np.linalg.norm(
                block_center_list[i] - block_center, ord=2)  # i->j
        return np.argsort(dist)[0]

    def gen_batch_data(self):
        index = self.now_index
        self.now_index += 1
        self.temp_data = []
        self.temp_index = 0

        print(' ==== generate batch data of {} ==== '.format(
            self.scene_list[index]))

        delta = self.stride
        if self.with_rgb:
            point_set_ini = self.scene_points_list[index]
        else:
            point_set_ini = self.scene_points_list[index][:, 0:3]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set_ini[:, 0:3], axis=0)
        coordmin = np.min(point_set_ini[:, 0:3], axis=0)
        nsubvolume_x = np.ceil(
            (coordmax[0] - coordmin[0]) / delta).astype(np.int32)
        nsubvolume_y = np.ceil(
            (coordmax[1] - coordmin[1]) / delta).astype(np.int32)
        point_sets = []
        semantic_segs = []
        sample_weights = []
        point_idxs = []
        block_center = []
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin + [i * delta, j * delta, 0]
                curmax = curmin + [self.block_size, self.block_size, 0]

                curchoice = np.where((point_set_ini[:, 0] >= curmin[0]) & (point_set_ini[:, 0] <= curmax[0]) & (
                    point_set_ini[:, 1] >= curmin[1]) & (point_set_ini[:, 1] <= curmax[1]))[0]

                cur_point_set = point_set_ini[curchoice, :]
                cur_semantic_seg = semantic_seg_ini[curchoice]

                bc = (curmin[0:2] + curmax[0:2]) / 2.0
                cur_point_set[:, 0] -= bc[0]
                cur_point_set[:, 1] -= bc[1]
                current_points = np.zeros((cur_point_set.shape[0], 9))
                current_points[:, 6] = cur_point_set[:, 0] / coordmax[0]
                current_points[:, 7] = cur_point_set[:, 1] / coordmax[1]
                current_points[:, 8] = cur_point_set[:, 2] / coordmax[2]
                current_points[:, 0:6] = cur_point_set

                if len(cur_semantic_seg) == 0:
                    continue
                point_sets.append(current_points)  # 1xNx3/6
                semantic_segs.append(cur_semantic_seg)  # 1xN
                point_idxs.append(curchoice)  # 1xN
                block_center.append(bc)

        # merge small blocks
        num_blocks = len(point_sets)
        block_idx = 0
        while block_idx < num_blocks:
            if point_sets[block_idx].shape[0] > (self.block_points / 2):
                block_idx += 1
                continue

            small_block_data = point_sets[block_idx].copy()
            small_block_seg = semantic_segs[block_idx].copy()
            small_block_idxs = point_idxs[block_idx].copy()
            small_block_center = block_center[block_idx].copy()
            point_sets.pop(block_idx)
            semantic_segs.pop(block_idx)
            point_idxs.pop(block_idx)
            block_center.pop(block_idx)
            nearest_block_idx = self.nearest_dist(
                small_block_center, block_center)
            point_sets[nearest_block_idx] = np.concatenate(
                (point_sets[nearest_block_idx], small_block_data), axis=0)
            semantic_segs[nearest_block_idx] = np.concatenate(
                (semantic_segs[nearest_block_idx], small_block_seg), axis=0)
            point_idxs[nearest_block_idx] = np.concatenate(
                (point_idxs[nearest_block_idx], small_block_idxs), axis=0)
            num_blocks = len(point_sets)

        # divide large blocks
        num_blocks = len(point_sets)
        div_blocks = []
        div_blocks_seg = []
        # div_blocks_smpw = []
        div_blocks_idxs = []
        div_blocks_center = []
        for block_idx in range(num_blocks):
            cur_num_pts = point_sets[block_idx].shape[0]

            point_idx_block = np.array([x for x in range(cur_num_pts)])
            if point_idx_block.shape[0] % self.block_points != 0:
                makeup_num = self.block_points - \
                    point_idx_block.shape[0] % self.block_points
                np.random.shuffle(point_idx_block)
                point_idx_block = np.concatenate(
                    (point_idx_block, point_idx_block[0:makeup_num].copy()))

            np.random.shuffle(point_idx_block)

            sub_blocks = list(self.chunks(point_idx_block, self.block_points))

            div_blocks += self.split_data(point_sets[block_idx], sub_blocks)
            div_blocks_seg += self.split_data(
                semantic_segs[block_idx], sub_blocks)
            div_blocks_idxs += self.split_data(
                point_idxs[block_idx], sub_blocks)
            div_blocks_center += [block_center[block_idx].copy()
                                  for i in range(len(sub_blocks))]

        for i in range(len(div_blocks)):
            point_set = div_blocks[i]
            if self.with_rgb:
                point_set[:, 3:6] /= 255.0
            self.temp_data.append(
                (point_set, div_blocks_seg[i], div_blocks_idxs[i]))

    def __next__(self):
        if self.temp_index >= len(self.temp_data):
            raise StopIteration()
        else:
            idx = self.temp_index
            self.temp_index += 1
            return self.temp_data[idx]


if __name__ == '__main__':
    test_dst = S3DISWholeScene_evaluation(root='/home/zizheng/data/s3dis/stanford_indoor3d_all_classes',
                                          split='test',
                                          test_area=5,
                                          block_points=8192,
                                          with_rgb=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    loader = torch_data.DataLoader(
        test_dst, batch_size=12, shuffle=False, pin_memory=True, num_workers=0)
    for i, data in enumerate(loader):
        a, b, d = data
        print(a.shape)
        print(np.max(a[0, :, 0].data.cpu().numpy()) -
              np.min(a[0, :, 0].data.cpu().numpy()))

    for i, data in enumerate(loader):
        a, b, d = data
        print(a.shape)
