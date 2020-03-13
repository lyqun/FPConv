import pickle
import os
import sys
import numpy as np
import torch.utils.data as torch_data

class ScannetDataset(torch_data.Dataset):
    def __init__(self, root=None, npoints=10240, split='train', with_dropout=False, with_norm=False, with_rgb=False, sample_rate=None):
        super().__init__()
        print(' ---- load data from', root)
        self.npoints = npoints
        self.with_dropout = with_dropout

        self.indices = [0, 1, 2]
        if with_norm: self.indices += [3, 4, 5]
        if with_rgb: self.indices += [6, 7, 8]
        print('load scannet dataset <{}> with npoint {}, indices: {}.'.format(split, npoints, self.indices))

        data_filename = os.path.join(root, 'scannet_%s_rgb21c_pointid.pickle' % (split))
        with open(data_filename, 'rb') as fp:
            self.scene_points_list = pickle.load(fp)
            self.semantic_labels_list = pickle.load(fp)
            scene_points_id = pickle.load(fp)
            num_point_all = pickle.load(fp)
        
        if split == 'train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            # self.labelweights = 1/np.log(1.2+labelweights)
            self.labelweights = np.power(np.amax(labelweights[1:]) / labelweights, 1 / 3.0)
        elif split == 'eval' or split == 'test':
            self.labelweights = np.ones(21)
        else:
            raise ValueError('split must be train or eval.')

        if sample_rate is not None:
            num_point = npoints
            sample_prob = num_point_all / np.sum(num_point_all)
            num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
            room_idxs = []
            for index in range(len(self.scene_points_list)):
                repeat_times = round(sample_prob[index] * num_iter)
                repeat_times = int(max(repeat_times, 1))
                room_idxs.extend([index] * repeat_times)
            self.room_idxs = np.array(room_idxs)
            np.random.seed(123)
            np.random.shuffle(self.room_idxs)
        else:
            self.room_idxs = np.arange(len(self.scene_points_list))

        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))
    
    def __getitem__(self, index):
        index = self.room_idxs[index]

        data_set = self.scene_points_list[index]
        point_set = data_set[:, :3]
        semantic_seg = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set, axis=0)
        coordmin = np.min(point_set, axis=0)
        smpmin = np.maximum(coordmax-[2, 2, 3.0], coordmin)
        smpmin[2] = coordmin[2]
        smpsz = np.minimum(coordmax-smpmin,[2,2,3.0])
        smpsz[2] = coordmax[2]-coordmin[2]
        isvalid = False
        # randomly choose a point as center point and sample <n_points> points in the box area of center-point
        for i in range(10):
            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:]
            curmin = curcenter - [1, 1, 1.5]
            curmax = curcenter + [1, 1, 1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set >= (curmin - 0.2)) * (point_set <= (curmax + 0.2)), axis=1) == 3
            cur_point_set = point_set[curchoice, :]
            cur_data_set = data_set[curchoice, :]
            cur_semantic_seg = semantic_seg[curchoice]
            if len(cur_semantic_seg) == 0:
                continue
            mask = np.sum((cur_point_set >= (curmin - 0.01)) * (cur_point_set <= (curmax + 0.01)), axis=1) == 3
            vidx = np.ceil((cur_point_set[mask, :] - curmin) / (curmax - curmin) * [31.0, 31.0, 62.0])
            vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 + vidx[:, 1] * 62.0 + vidx[:, 2])
            isvalid = np.sum(cur_semantic_seg > 0) / len(cur_semantic_seg) >= 0.7 and len(vidx) / 31.0 / 31.0 / 62.0 >= 0.02
            if isvalid:
                break
        
        choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        sample_weight = self.labelweights[semantic_seg]
        sample_weight *= mask

        selected_points = cur_data_set[choice, :] # np * 6, xyz + rgb
        point_set = np.zeros((self.npoints, 9)) # xyz, norm_xyz, rgb

        point_set[:, :3] = selected_points[:, :3] # xyz
        for i in range(3): # normalized_xyz
            point_set[:, 3 + i] = (selected_points[:, i] - coordmin[i]) / (coordmax[i] - coordmin[i])
        point_set[:, 6:] = selected_points[:, 3:] / 255.0 # rgb

        if self.with_dropout:
            dropout_ratio = np.random.random() * 0.875 # 0 ~ 0.875
            drop_idx = np.where(np.random.random((self.npoints)) <= dropout_ratio)[0]

            point_set[drop_idx, :] = point_set[0, :]
            semantic_seg[drop_idx] = semantic_seg[0]
            sample_weight[drop_idx] *= 0

        point_set = point_set[:, self.indices]
        return point_set, semantic_seg, sample_weight

    def __len__(self):
        return len(self.room_idxs)
        # return len(self.scene_points_list)

class ScannetDatasetWholeScene(torch_data.IterableDataset):
    def __init__(self, root=None, npoints=10240, split='train', with_norm=True, with_rgb=True):
        super().__init__()
        print(' ---- load data from', root)
        self.npoints = npoints
        
        self.indices = [0, 1, 2]
        if with_norm: self.indices += [3, 4, 5]
        if with_rgb: self.indices += [6, 7, 8]
        print('load scannet <whole scene> dataset <{}> with npoint {}, indices: {}.'.format(split, npoints, self.indices))
        
        self.temp_data = []
        self.temp_index = 0
        self.now_index = 0
        
        data_filename = os.path.join(root, 'scannet_%s_rgb21c_pointid.pickle' % (split))
        with open(data_filename, 'rb') as fp:
            self.scene_points_list = pickle.load(fp)
            self.semantic_labels_list = pickle.load(fp)
        if split == 'train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            # self.labelweights = 1 / np.log(1.2 + labelweights)
            self.labelweights = np.power(np.amax(labelweights[1:]) / labelweights, 1 / 3.0)
        elif split == 'eval' or split == 'test':
            self.labelweights = np.ones(21)
    
    def get_data(self):
        idx = self.temp_index
        self.temp_index += 1
        return self.temp_data[idx]

    def reset(self):
        self.temp_data = []
        self.temp_index = 0
        self.now_index = 0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.now_index >= len(self.scene_points_list) and self.temp_index >= len(self.temp_data):
            raise StopIteration()
        
        if self.temp_index < len(self.temp_data):
            return self.get_data()

        index = self.now_index
        self.now_index += 1
        self.temp_data = []
        self.temp_index = 0

        data_set_ini = self.scene_points_list[index]
        point_set_ini = data_set_ini[:,:3]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set_ini,axis=0)
        coordmin = np.min(point_set_ini,axis=0)
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/2).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/2).astype(np.int32)
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        isvalid = False
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin+[i*2,j*2,0]
                curmax = coordmin+[(i+1)*2,(j+1)*2,coordmax[2]-coordmin[2]]
                curchoice = np.sum((point_set_ini>=(curmin-0.2))*(point_set_ini<=(curmax+0.2)),axis=1)==3
                cur_point_set = point_set_ini[curchoice,:]
                cur_data_set = data_set_ini[curchoice,:]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                if len(cur_semantic_seg)==0:
                    continue
                mask = np.sum((cur_point_set >= (curmin - 0.001)) * (cur_point_set <= (curmax + 0.001)), axis=1) == 3
                
                choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=len(cur_semantic_seg) < self.npoints)
                semantic_seg = cur_semantic_seg[choice] # N
                mask = mask[choice]
                if sum(mask) / float(len(mask)) < 0.01:
                    continue
                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask # N
                
                selected_points = cur_data_set[choice, :] # Nx6
                point_set = np.zeros([self.npoints, 9])
                point_set[:, :3] = selected_points[:, :3] # xyz
                for k in range(3): # normalized_xyz
                    point_set[:, 3 + k] = (selected_points[:, k] - coordmin[k]) / (coordmax[k] - coordmin[k])
                point_set[:, 6:] = selected_points[:, 3:] / 255.0 # rgb

                point_set = point_set[:, self.indices]
                self.temp_data.append((point_set, semantic_seg, sample_weight))

        return self.get_data()

