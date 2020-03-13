import pickle
import os
import sys
import numpy as np
import torch.utils.data as torch_data

class ScannetDatasetWholeScene_evaluation(torch_data.IterableDataset):
    #prepare to give prediction on each points
    def __init__(self, root=None, scene_list_dir=None, split='test', num_class=21, block_points=10240, with_norm=True, with_rgb=True):
        super().__init__()
        print(' ---- load data from', root)
        self.block_points = block_points
        self.indices = [0, 1, 2]
        if with_norm: self.indices += [3, 4, 5]
        if with_rgb: self.indices += [6, 7, 8]
        print('load scannet <TEST> dataset <{}> with npoint {}, indices: {}.'.format(split, block_points, self.indices))
        
        
        self.point_num = []
        self.temp_data = []
        self.temp_index = 0
        self.now_index = 0
        
        data_filename = os.path.join(root, 'scannet_%s_rgb21c_pointid.pickle' % (split))
        with open(data_filename, 'rb') as fp:
            self.scene_points_list = pickle.load(fp)
            self.semantic_labels_list = pickle.load(fp)
            self.scene_points_id = pickle.load(fp)
            self.scene_points_num = pickle.load(fp)
            file_path = os.path.join(scene_list_dir, 'scannetv2_{}.txt'.format(split))

        num_class = 21
        if split == 'test' or split == 'eval' or split == 'train':
            self.labelweights = np.ones(num_class)
            for seg in self.semantic_labels_list:
                self.point_num.append(seg.shape[0])
            
            with open(file_path) as fl:
                self.scene_list = fl.read().splitlines()
        else:
            raise ValueError('split must be test or eval, {}'.format(split))

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
            dist[i] = np.linalg.norm(block_center_list[i] - block_center, ord = 2) #i->j
        return np.argsort(dist)[0]

    def gen_batch_data(self):
        index = self.now_index
        self.now_index += 1
        self.temp_data = []
        self.temp_index = 0

        print(' ==== generate batch data of {} ==== '.format(self.scene_list[index]))

        delta = 0.5
        # if self.with_rgb:
        point_set_ini = self.scene_points_list[index]
        # else:
        #     point_set_ini = self.scene_points_list[index][:, 0:3]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set_ini[:, 0:3],axis=0)
        coordmin = np.min(point_set_ini[:, 0:3],axis=0)
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/delta).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/delta).astype(np.int32)
        point_sets = []
        semantic_segs = []
        sample_weights = []
        point_idxs = []
        block_center = []
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin+[i*delta,j*delta,0]
                curmax = curmin+[2,2,coordmax[2]-coordmin[2]]
                curchoice = np.sum((point_set_ini[:,0:3]>=(curmin-0.2))*(point_set_ini[:,0:3]<=(curmax+0.2)),axis=1)==3
                curchoice_idx = np.where(curchoice)[0]
                cur_point_set = point_set_ini[curchoice,:]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                if len(cur_semantic_seg)==0:
                    continue
                mask = np.sum((cur_point_set[:,0:3]>=(curmin-0.001))*(cur_point_set[:,0:3]<=(curmax+0.001)),axis=1)==3
                sample_weight = self.labelweights[cur_semantic_seg]
                sample_weight *= mask # N
                point_sets.append(cur_point_set) # 1xNx3/6
                semantic_segs.append(cur_semantic_seg) # 1xN
                sample_weights.append(sample_weight) # 1xN
                point_idxs.append(curchoice_idx) #1xN
                block_center.append((curmin[0:2] + curmax[0:2]) / 2.0)

        # merge small blocks
        num_blocks = len(point_sets)
        block_idx = 0
        while block_idx < num_blocks:
            if point_sets[block_idx].shape[0] > self.block_points // 2:
                block_idx += 1
                continue
            
            small_block_data = point_sets[block_idx].copy()
            small_block_seg = semantic_segs[block_idx].copy()
            small_block_smpw = sample_weights[block_idx].copy()
            small_block_idxs = point_idxs[block_idx].copy()
            small_block_center = block_center[block_idx].copy()
            point_sets.pop(block_idx)
            semantic_segs.pop(block_idx)
            sample_weights.pop(block_idx)
            point_idxs.pop(block_idx)
            block_center.pop(block_idx)
            nearest_block_idx = self.nearest_dist(small_block_center, block_center)
            point_sets[nearest_block_idx] = np.concatenate((point_sets[nearest_block_idx], small_block_data), axis = 0)
            semantic_segs[nearest_block_idx] = np.concatenate((semantic_segs[nearest_block_idx], small_block_seg), axis = 0)
            sample_weights[nearest_block_idx] = np.concatenate((sample_weights[nearest_block_idx], small_block_smpw), axis = 0)
            point_idxs[nearest_block_idx] = np.concatenate((point_idxs[nearest_block_idx], small_block_idxs), axis = 0)
            num_blocks = len(point_sets)

        #divide large blocks
        num_blocks = len(point_sets)
        div_blocks = []
        div_blocks_seg = []
        div_blocks_smpw = []
        div_blocks_idxs = []
        div_blocks_center = []
        for block_idx in range(num_blocks):
            cur_num_pts = point_sets[block_idx].shape[0]

            point_idx_block = np.array([x for x in range(cur_num_pts)])
            if point_idx_block.shape[0]%self.block_points != 0:
                makeup_num = self.block_points - point_idx_block.shape[0]%self.block_points
                np.random.shuffle(point_idx_block)
                point_idx_block = np.concatenate((point_idx_block,point_idx_block[0:makeup_num].copy()))

            np.random.shuffle(point_idx_block)

            sub_blocks = list(self.chunks(point_idx_block, self.block_points))

            div_blocks += self.split_data(point_sets[block_idx], sub_blocks)
            div_blocks_seg += self.split_data(semantic_segs[block_idx], sub_blocks)
            div_blocks_smpw += self.split_data(sample_weights[block_idx], sub_blocks)
            div_blocks_idxs += self.split_data(point_idxs[block_idx], sub_blocks)
            div_blocks_center += [block_center[block_idx].copy() for i in range(len(sub_blocks))]

        for i in range(len(div_blocks)):
            selected_points = div_blocks[i]
            point_set = np.zeros([self.block_points, 9])
            point_set[:, :3] = selected_points[:, :3] # xyz
            for k in range(3): # normalized_xyz
                point_set[:, 3 + k] = (selected_points[:, k] - coordmin[k]) / (coordmax[k] - coordmin[k])
            point_set[:, 6:] = selected_points[:, 3:] / 255.0 # rgb

            point_set = point_set[:, self.indices]
            self.temp_data.append((point_set, div_blocks_seg[i], div_blocks_smpw[i], div_blocks_idxs[i]))


    def __next__(self):
        if self.temp_index >= len(self.temp_data):
            raise StopIteration()
        else:
            idx = self.temp_index
            self.temp_index += 1
            return self.temp_data[idx]


