import os
import uuid
import open3d as o3d
import numpy as np
import torch.utils.data as data
from data_preproc.data_preprocess import glsproc_pc
from pathlib import Path
import data_preproc.pt as pointCloud
from utils import get_psnr
from metrics.utils import gnp


class EncodeEHEMDataset(data.Dataset):
    """ImageFolder can be used to load images where there are no labels."""

    def __init__(
        self,
        test_files,
        context_size,
        data_type,
        level_wise=True,
        lidar_level=12,
        preproc_path='',
        use_scaling=True,
    ):
        self.test_files = test_files
        self.context_size = context_size
        self.data_type = data_type
        self.level_wise = level_wise
        self.lidar_level = lidar_level
        self.preproc_path = preproc_path
        self.use_scaling = use_scaling
        self.i = 0
        if not os.path.exists('temp'):
            os.mkdir('temp')

    def __getitem__(self, index):
        ori_file = self.test_files[index]
        npy_path, pc, chamfer, bin_num, gnp, psnr = self.preproc(ori_file)
        z_offset = 0
        oct_seq = np.load(npy_path + ".npz", allow_pickle=True)['arr_0']

        oct_seq[:, :, 0] -= 1
        whole_ids = np.arange(len(oct_seq)).astype(np.int64)
        data = []
        poss = []
        extent = []
        pos_mm = []
        ids = []
        cur_level = 1
        cur_level_start = 0
        for i in range(len(oct_seq)):
            if oct_seq[i, -1, 1] > cur_level:
                level_data = oct_seq[cur_level_start:i, :, :3]
                level_data = np.concatenate((level_data[:, :, 1:], level_data[:, :, :1]), axis=2)
                data.append(level_data)
                cur_extent = oct_seq[cur_level_start:i, -1, 3:5]
                cur_pos = oct_seq[cur_level_start:i, -1, 5:]
                pos_max, pos_min = cur_pos.max(), cur_pos.min()
                extent.append(((cur_extent-pos_min) / (pos_max - pos_min + 1e-9)).astype(np.float32))
                poss.append(((cur_pos - pos_min) / (pos_max - pos_min + 1e-9)).astype(np.float32).transpose((1, 0)))
                pos_mm.append((pos_min, pos_max)) 

                ids.append(whole_ids[cur_level_start:i] - cur_level_start)
                cur_level_start = i
                cur_level = oct_seq[i, -1, 1]

        level_data = oct_seq[cur_level_start:, :, :3]
        level_data[:, :, 1] = np.clip(level_data[:, :, 1], None, self.lidar_level)
        level_data = np.concatenate((level_data[:, :, 1:], level_data[:, :, :1]), axis=2)  # level, octant, occupancy
        data.append(level_data)

        cur_pos = oct_seq[cur_level_start:, -1, 5:]
        cur_extent = oct_seq[cur_level_start:, -1, 3:5]
        pos_max, pos_min = cur_pos.max(), cur_pos.min()
        extent.append((cur_extent).astype(np.float32))
        poss.append(((cur_pos - pos_min) / (pos_max - pos_min + 1e-9)).astype(np.float32).transpose((1, 0)))
        pos_mm.append((pos_min, pos_max))

        ids.append(whole_ids[cur_level_start:] - cur_level_start)
        return ids, poss, extent, pos_mm, data, oct_seq, len(pc), pc, bin_num, z_offset, chamfer, gnp, psnr

    def preproc(self, ori_file, self_metrics = True):
        ori_path = Path(ori_file)
        out_file = ori_path.parent / ori_path.stem / Path(".npz")
        if out_file.exists():
            return str(out_file)

        if self.data_type == 'kitti' or self.data_type == 'nuscenes':
            peak = '59.70'
        elif self.data_type == 'ford':
            peak = '30000'

        tmp_test_file = "temp/pcerror_results" + str(uuid.uuid4()) + ".txt"
        out_file, quantized_pc, pc, nodenum= glsproc_pc(
            ori_file,
            self.preproc_path,
            ori_path.parents[1].name+'_'+ori_path.stem,
            quant_size = 80 if self.data_type == 'kitti' else 120 if self.data_type == 'nuscenes' else (2**17),
            lidar_level=self.lidar_level,
            Layer_indexs=[],
            cylin=True,
            datatype = self.data_type, 
            mode = 'test'
        )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(quantized_pc)
        o3d.io.write_point_cloud('demo/recon/{}.ply'.format(ori_path.parents[1].name+'_'+ori_path.stem), pcd)
        pcd.points = o3d.utility.Vector3dVector(pc)
        
        o3d.io.write_point_cloud('demo/origin/{}.ply'.format(ori_path.parents[1].name+'_'+ori_path.stem), pcd)
        
        if self_metrics:
            pointCloud.pcerror(pc, quantized_pc, None, "-r " + peak, tmp_test_file)
            gnp_value = gnp(pc, quantized_pc, 59.7)
            return out_file, pc, pointCloud.distChamfer(pc, quantized_pc), nodenum, gnp_value, get_psnr(tmp_test_file)[0]

    def __len__(self):
        return len(self.test_files)
