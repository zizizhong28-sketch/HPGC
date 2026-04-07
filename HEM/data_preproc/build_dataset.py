import os
import uuid
import glob
import sys
sys.path.append('.')
from data_preproc.data_preprocess import glsproc_pc
from pathlib import Path
import data_preproc.pt as pointCloud
from utils import get_psnr
from joblib import Parallel, delayed
from tqdm import tqdm
from metrics.utils import gnp

class BuildDataset():
    """ImageFolder can be used to load images where there are no labels."""

    def __init__(self, test_files, context_size, data_type, level_wise=True, lidar_level=12, quant_size=50,save_dir = 'data/',layer_indexs = [],cylin=True,resize='exp',num_workers = 4):
        self.test_files = test_files
        self.context_size = context_size
        self.data_type = data_type
        self.level_wise = level_wise
        self.lidar_level = lidar_level
        self.quant_size = 80 if data_type == 'kitti' else 120 if data_type == 'nuscenes' else 2**17
        self.save_dir = save_dir
        self.layer_indexs = layer_indexs 
        self.cylin = cylin
        self.num_workers = num_workers if num_workers > 0 else 1
        if not os.path.exists('temp'):
            os.mkdir('temp')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self.test_files = sorted(glob.glob(test_files))
        

    def begin_build(self,mode:str='train'):
        """
        使用 joblib 并行处理所有文件。
        """
        print(f" Starting to build dataset using {self.num_workers} workers...")

        # 使用 Parallel 和 delayed 来并行化 preproc 方法的调用
        # tqdm 会为处理过程显示一个漂亮的进度条
        Parallel(n_jobs=self.num_workers)(
            delayed(self.preproc)(ori_file, mode) for ori_file in tqdm(self.test_files, desc="{} : Processing files on depth {}".format(mode,self.lidar_level))
        )

        print("All files processed.")
        # 返回一个包含所有文件处理结果的列表
        # 每个元素是 preproc 函数返回的一个元组
        # return results

    def preproc(self, ori_file, mode = 'test', self_metrics=True):

        self.save_dir = os.path.join(self.save_dir,mode)
        self.save_dir = os.path.join(self.save_dir,str(self.lidar_level))

        ori_path = Path(ori_file)
        # out_name =  ori_path.parents[1].name+'_'+ori_path.stem
        # # 检查是否存在以 out_path 为前缀的文件
        # existing_files = list(Path(self.save_dir).glob(out_name + "*"))
        # if not existing_files:

        tmp_test_file = "temp/pcerror_results" + str(uuid.uuid4()) + ".txt"
        if self.data_type == 'kitti' or self.data_type == 'nuscenes':
            peak = '59.70'
        elif self.data_type == 'ford':
            peak = '30000'

        
        out_file, quantized_pc, pc, nodenum= glsproc_pc(
            
            ori_file,
            self.save_dir,
            ori_path.parents[1].name+'_'+ori_path.stem,
            datatype=self.data_type,
            quant_size = self.quant_size,
            lidar_level=self.lidar_level,
            Layer_indexs=self.layer_indexs,
            cylin=True,
            mode = mode, 
            save = True,
        )
        # print(quantized_pc)
        # exit(-1)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pc)
        # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        # # pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))

        # o3d.io.write_point_cloud('temp/data/normal_pc.ply', pcd, write_ascii=True)
        psnr_new = gnp(pc, quantized_pc, peak_value=30000 if self.data_type == 'ford' else 59.7)
        pointCloud.pcerror(pc, quantized_pc, None, "-r " + peak, tmp_test_file)
        psnr_old, _  = get_psnr(tmp_test_file)
        cd = pointCloud.distChamfer(pc, quantized_pc)

        # print(psnrd1)
        with open(os.path.join(self.save_dir, "psnr_results.txt"), "a") as f:
            f.write(f"{psnr_old}, {psnr_new}, {cd}, {nodenum}\n")
        return len(pc), pc, cd,  nodenum


def main():

    # dataset = BuildDataset(
    #     test_files='../ford/Ford_01_q_1mm/*.ply',
    #     data_type='ford',
    #     save_dir='data/ford',
    #     context_size=32768,
    #     lidar_level=14,
    #     num_workers=16,
    # )

    # dataset = BuildDataset(
    #     test_files='../../datasets/SemanticKITTIDataset/sequences/0*/velodyne/*.bin',
    #     data_type='kitti',
    #     save_dir='data/kitti',
    #     context_size=32768,
    #     lidar_level=13,
    #     num_workers=16,
    # )
    
    dataset = BuildDataset(
        test_files='../../datasets/NuScenes_lidar/train/LIDAR_TOP/*.bin',
        data_type='nuscenes',
        save_dir='data/NuScenes',
        context_size=32768,
        lidar_level=15,
        num_workers=16
    )

    dataset.begin_build(mode='test')

if __name__ == '__main__':
    main()
#     return len(self.test_files)