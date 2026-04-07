import numpy as np
import glob
import torch.utils.data as data
import torch
import glob


class HEMDataset(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

        self.file_names = []
        self.total_point_num = 0
        for filename in sorted(glob.glob(cfg.root)):
            if filename.endswith('.npz'):
                self.file_names.append('{}'.format(filename))
                self.total_point_num += int(filename.split('_')[-1].split('.')[0])
        self.root = cfg.root
        self.index = 0
        self.datalen = 0
        self.dataBuffer = []
        self.fileIndx = 0
        # self.tree_point_num = cfg.tree_size * cfg.context_size
        self.context_size = cfg.context_size
        assert self.file_names, 'no file found!'
        # self.max_time_each_file = self.total_point_num//(self.context_size*len(self.file_names))
        self.max_time_each_file = 0
        self.cur_times = self.max_time_each_file
        self.cur_max_level = 0
    
    def count_ones_in_binary_numpy_fast(self, label: np.ndarray) -> np.ndarray:
        # 转换为 uint64 方便拆分字节
        x = np.array(label).astype(np.int16)+1
        # 先展开成字节
        bytes_view = x.astype(np.uint8).reshape(x.shape + (-1,))
        # 用 unpackbits 得到每个字节的二进制位
        bits = np.unpackbits(bytes_view, axis=-1)
        # 求和就是 1 的数量
        return bits.sum(axis=-1)

    def __getitem__(self, index):   
        # N, K, 6
        # occupancy, level, octant, x, y, z
        if self.cur_times >= self.max_time_each_file:
            file_idx = index % len(self.file_names) # randomly select file
            self.cur_data = np.load(self.file_names[file_idx], allow_pickle=True)['arr_0']
            self.cur_data[:, :, 0] -= 1
            self.cur_max_level = max(self.cur_data[:, -1, 1])

            self.cur_times = 0
            cur_file_len = int(self.file_names[file_idx].split('_')[-1].split('.')[0])
            self.max_time_each_file = cur_file_len // self.context_size
            self.order = torch.randperm(self.max_time_each_file)

        cur_idx = self.order[self.cur_times]
        data = np.copy(self.cur_data[cur_idx*self.context_size:cur_idx*self.context_size + self.context_size])
        
        pos = data[:, -1, 5:]
        pos_max, pos_min = pos.max(), pos.min()

        pos = ((pos-pos_min) / (pos_max-pos_min)).astype(np.float32).transpose((1, 0))

        extent = data[:,-1, 3:5] 

        extent = ((extent-pos_min) / (pos_max-pos_min)).astype(np.float32)

        data = data[:, :, :3]
        
        data = np.concatenate((data[:, :, 1:], data[:, :, :1]), axis=2)
        label = np.copy(data[:, -1, 2])
        num = (self.count_ones_in_binary_numpy_fast(label)-1)/1.0
        # print(num.max(),num.min(),label.max(),label.min())
        self.cur_times += 1

        return data, extent, pos, label, num # data: level, octant, occ

    def __len__(self):
        return self.total_point_num//self.context_size
