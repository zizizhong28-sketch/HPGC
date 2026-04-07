import os
import uuid
import numpy as np
import torch.utils.data as data
from data_preproc.data_preprocess import glsproc_pc
from pathlib import Path
import data_preproc.pt as pointCloud
from utils import get_psnr


class EncodeDataset(data.Dataset):
    """ImageFolder can be used to load images where there are no labels."""

    def __init__(self, test_files = None, context_size = None, data_type = 'kitti', level_wise=True, lidar_level=12, preproc_path=''):
        self.test_files = test_files

        self.context_size = context_size
        self.data_type = data_type
        self.level_wise = level_wise
        self.lidar_level = lidar_level
        self.preproc_path = preproc_path
        if not os.path.exists('temp'):
            os.mkdir('temp')

    def __getitem__(self, index):

        ori_file = self.test_files[index]
        npy_path, pt, chamfer, node_num, psnr = self.preproc(ori_file)

        oct_seq = np.load(npy_path + ".npz", allow_pickle=True)['arr_0']
        padding = np.zeros([self.context_size - 1, oct_seq.shape[1], oct_seq.shape[2]]).astype(np.int64)
        padding[:, :, 0] = 255
        ids_pad = np.ones([self.context_size - 1]).astype(np.int64) * -1

        oct_seq[:, :, 0] -= 1
        whole_ids = np.arange(len(oct_seq)).astype(np.int64)
        max_level = max(oct_seq[:, -1, 1])
        data = []
        extent = []
        pos = []
        ids = []
        cur_level = 1 if self.level_wise else 100
        cur_level_start = 0
        for i in range(len(oct_seq)):
            if oct_seq[i, -1, 1] > cur_level:
                data.append(np.vstack((padding[:, :, :3], oct_seq[cur_level_start:i, :, :3])))
                extent.append(np.vstack((padding[:, :, 3:5].astype(np.float32), (oct_seq[cur_level_start:i, :, 3:5] / (2**max_level)).astype(np.float32))))
                pos.append(np.vstack((padding[:, :, 5:].astype(np.float32), (oct_seq[cur_level_start:i, :, 5:] / (2**max_level)).astype(np.float32))))
                ids.append(np.hstack((ids_pad, whole_ids[cur_level_start:i] - cur_level_start)))
                cur_level_start = i
                cur_level = oct_seq[i, -1, 1]
        data.append(np.vstack((padding[:, :, :3], oct_seq[cur_level_start:, :, :3])))
        extent.append(np.vstack((padding[:, :, 3:5].astype(np.float32), (oct_seq[cur_level_start:, :, 3:5] / (2**max_level)).astype(np.float32))))
        pos.append(np.vstack((padding[:, :, 5:].astype(np.float32), (oct_seq[cur_level_start:, :, 5:] / (2**max_level)).astype(np.float32))))
        ids.append(np.hstack((ids_pad, whole_ids[cur_level_start:] - cur_level_start)))
        return ids, pos, extent, data, oct_seq, len(pt), node_num, chamfer, psnr
    
    def preproc(self, ori_file):
        ori_path = Path(ori_file)
        out_file = ori_path.parent / ori_path.stem / Path(".npz")
        if out_file.exists():
            return str(out_file)

        if self.data_type == 'kitti':
            peak = '59.70'
        elif self.data_type == 'ford':
            peak = '30000'

        tmp_test_file = "temp/pcerror_results" + str(uuid.uuid4()) + ".txt"
        out_file, quantized_pc, pc, nodenum= glsproc_pc(
            ori_file,
            self.preproc_path,
            ori_path.parents[1].name+'_'+ori_path.stem,
            quant_size = 50 if self.data_type == 'kitti' else 100000,
            lidar_level=self.lidar_level,
            Layer_indexs=[],
            cylin=True,
            resize='exp',
            m=90,
            n=0.015,
            z_rate = 1,
            type = self.data_type, 
        )

        pointCloud.pcerror(pc, quantized_pc, None, "-r " + peak, tmp_test_file)
        return out_file, pc, pointCloud.distChamfer(pc, quantized_pc), nodenum, get_psnr(tmp_test_file)[0]

    def __len__(self):
        return len(self.test_files)
