import argparse
import glob
import sys
sys.path.append('.')
from data_preproc.Octree import gen_K_parent_seq, mullevel_gen_octree, DeOctree
import data_preproc.pt as pointCloud
import numpy as np
import math
import os
import open3d as o3d

import sys
# 获取当前工作目录
current_dir = os.getcwd()
# 构造包含 .so 文件的路径（假设 .so 文件位于 pybind 文件夹下）
pybind_dir = os.path.join(current_dir, "pybind")
# 将 pybind 文件夹路径添加到 sys.path
sys.path.append(pybind_dir)
from data_preproc import fastutils
from scipy.stats import norm

from pyntcloud import PyntCloud


def power_transform(x, upper_bound, alpha=2.0):

    x = np.asarray(x, dtype=float)

    return upper_bound * (x / upper_bound)**alpha

def power_inverse(y, upper_bound, alpha=2.0):
    """
    反变换（与 power_transform 对应）。在调用时请使用与变换相同的 a,b,mu,alpha。
    """
    y = np.asarray(y, dtype=float)

    return (upper_bound) * (y / upper_bound)**(1/alpha)

def log_transform(x, base, scale_factor = 1):

    x = np.asarray(x,dtype=float)

    return scale_factor * np.log1p(x / scale_factor) / np.log(base)


def log_inverse(y, base, scale_factor = 1):

    y = np.asarray(y, dtype= float)

    return  scale_factor * (np.power(base, y / scale_factor) - 1)


# 哈希表
def xor_point_clouds_hash_table(cloud1, cloud2):
    set1 = set(map(tuple, cloud1))
    set2 = set(map(tuple, cloud2))
    return np.array(list(set1.symmetric_difference(set2)))

def read_ply_pyntcloud(file_path):
    """快速读取PLY点云并返回点坐标数组"""
    # 读取PLY文件（支持二进制和ASCII格式）
    cloud = PyntCloud.from_file(file_path)
    
    # 获取点坐标（x, y, z）
    points = cloud.points[['x', 'y', 'z']].to_numpy()
    
    return points

def uniform_sampling(point_cloud, ratio=0.9):
    """
    对n×3的点云数组进行均匀采样
    
    参数:
        point_cloud: 形状为(n, 3)的numpy数组，代表点云数据
        ratio: 采样比例，默认为0.9（保留90%的点）
        
    返回:
        sampled_cloud: 采样后的点云数组
    """
    # 获取点云中点的数量
    n_points = point_cloud.shape[0]
    
    # 计算需要采样的点数量
    sample_size = int(n_points * ratio)
    
    # 防止采样数量为0（当点云数量极少时）
    if sample_size == 0 and n_points > 0:
        sample_size = 1
    
    # 随机选择索引
    np.random.seed(42)  # 设置随机种子以确保结果可复现
    indices = np.random.choice(n_points, sample_size, replace=False)
    
    # 进行采样
    sampled_cloud = point_cloud[indices, :]
    
    return sampled_cloud


def density_based_sampling(point_cloud, ratio=0.9, voxel_size=0.1, min_points_per_voxel=1):
    """
    基于密度的LiDAR点云采样：密集区域多保留，稀疏区域（边缘）少删减
    
    参数:
        point_cloud: 形状为(n, 3)的numpy数组，LiDAR点云
        ratio: 总体采样比例（保留90%的点）
        voxel_size: 体素大小（控制局部区域粒度，单位与点云坐标一致，如米）
        min_points_per_voxel: 每个体素最少保留的点数（保护边缘稀疏区域）
        
    返回:
        sampled_cloud: 采样后的点云数组
    """
    n_total = point_cloud.shape[0]
    n_sample_total = int(n_total * ratio)
    
    # 防止采样数为0或超过原始点数
    if n_sample_total <= 0:
        return point_cloud[:1]
    if n_sample_total >= n_total:
        return point_cloud.copy()
    
    # 1. 体素划分：计算每个点所属的体素索引（将连续坐标离散化为整数体素ID）
    # 对坐标取整（除以体素大小后），作为体素的唯一标识
    voxel_indices = (point_cloud / voxel_size).astype(int)
    # 将三维索引转换为一维字符串ID（方便分组）
    voxel_ids = [f"{x}_{y}_{z}" for x, y, z in voxel_indices]
    
    # 2. 按体素分组，统计每个体素的点数
    from collections import defaultdict
    voxel_points = defaultdict(list)  # 键：体素ID，值：该体素内的点索引
    for idx, vid in enumerate(voxel_ids):
        voxel_points[vid].append(idx)
    
    # 3. 计算每个体素应采样的点数
    voxel_sampling = []  # 存储每个体素的采样点索引
    total_points = n_total
    remaining = n_sample_total  # 剩余需要采样的总点数
    
    # 先处理所有体素，计算每个体素的理论采样数（按比例）
    voxel_counts = {vid: len(pts) for vid, pts in voxel_points.items()}
    total_counts = sum(voxel_counts.values())
    
    # 按体素点数占比分配采样名额
    for vid, pts in voxel_points.items():
        count = len(pts)
        # 理论采样数 = 总采样数 * (该体素点数 / 总点数)
        n_sample = int(remaining * (count / total_counts))
        # 确保不超过实际点数，且不低于最小保留数
        n_sample = max(min(n_sample, count), min_points_per_voxel)
        voxel_sampling.append((pts, n_sample))
        remaining -= n_sample
    
    # 4. 如果还有剩余名额（因取整导致），分配给点数最多的体素
    if remaining > 0:
        # 按体素内点数排序，优先给点数多的体素分配剩余名额
        voxel_sampling.sort(key=lambda x: len(x[0]), reverse=True)
        for i in range(len(voxel_sampling)):
            pts, n = voxel_sampling[i]
            add = min(remaining, len(pts) - n)  # 最多加到该体素的实际点数
            if add <= 0:
                continue
            voxel_sampling[i] = (pts, n + add)
            remaining -= add
            if remaining == 0:
                break
    
    # 5. 在每个体素内随机采样（也可改为均匀采样）
    sampled_indices = []
    for pts, n in voxel_sampling:
        # 从体素内的点中随机选n个（保证局部均匀）
        if len(pts) <= n:
            sampled_indices.extend(pts)  # 点数不足时全保留
        else:
            sampled_indices.extend(np.random.choice(pts, n, replace=False))
    
    # 6. 生成采样后的点云
    sampled_cloud = point_cloud[sampled_indices]
    return sampled_cloud

def glsproc_pc(
    inp_path,
    out_dir,
    out_name,
    datatype='kitti',
    lidar_level=12,
    quant_size = 50,
    Layer_indexs=[],
    rotation=False,
    cylin=True,
    mode  ='test',
    save = True
):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 量化计算当前点和前一帧
    # now PC

    if datatype == 'kitti':

        points = np.fromfile(inp_path, dtype=np.float32, count=-1).reshape([-1, 4])[:,:3]
        
    elif datatype == 'ford':
        points = read_ply_pyntcloud(inp_path)
    elif datatype == 'nuscenes':
        points = np.fromfile(inp_path, dtype=np.float32, count=-1).reshape([-1, 5])[:,:3]
        
    else:
        raise ValueError(f"Unsupported data type: {datatype}")
    
    oct_center = np.array([quant_size/2,quant_size/2,quant_size/2])
    # log = False
    # m = 6.5
    # n = -0.5
    # m = 400
    # n = 0.004
    ref_pt = points
        
    points = ref_pt

    alpha = .5 if datatype == 'ford' else .7 if datatype == 'kitti' else .7
    base = 10000

    # points = cart2cylin(ref_pt, base, alpha)
    points = cart2spher(ref_pt)
    bin_num = np.round(points[:, 0].max())
    # if cylin:
        
    #     bin_num = np.round(points[:, 0].max() / qs) + 1
    #     # qs = np.array([qs, 2 * math.pi / (bin_num - 1), qs])[True]
    #     qs = np.array([qs, 2 * math.pi / (bin_num - 1), math.pi / (bin_num - 1)])[True]

    P_quant = points/np.array([1, 2 * math.pi / bin_num , math.pi / bin_num])

    # 建树，特征提取+保存
    # octree,root,node_numlist = fastutils.GenOctree(P_quant,lidar_level,np.array([50,50,45]),quant_size,z_rate,cylin=True)

    octree,root,node_numlist = fastutils.GenOctree(P_quant,lidar_level,oct_center,quant_size,cylin=True)
    
    pc_struct = gen_K_parent_seq(octree, 4, node_numlist, layer_indexs=Layer_indexs)

    out_pc = np.concatenate((pc_struct["Seq"][:, :, True], pc_struct["Level"], pc_struct["Extents"],pc_struct["Pos"]), axis=2)

    out_file = os.path.join(out_dir, out_name + "_" + str(out_pc.shape[0]))

    if save:
        np.savez_compressed(out_file + '.npz', out_pc)
    

    out_points = np.array(fastutils.octree2pointcloud(root))

    # print('*'*10+f'max_r:{np.max(P_quant[:,:2])}'+'*'*10)

    out_points = out_points * np.array([1, 2 * math.pi / bin_num , math.pi / bin_num])
    # out_points = cylin2cart(out_points,base,alpha)
    out_points = spher2cart(out_points)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(out_points)

    # ply_out_file = os.path.join(out_dir, 'plys')
    # ply_out_file = os.path.join(ply_out_file, out_name.split('_')[-1])
    # o3d.io.write_point_cloud(ply_out_file + '.ply', pcd)
    # node_num = sum(node_numlist)
    # print('data size & nodenum:', out_pc.shape, node_num)
    node_num = out_pc.shape[0] #non-leaf node

    return [out_file, out_points, ref_pt, node_num]



def cart2cylin(points, quant_size, alpha):
    """
    如果输入是 (B, N, 3)，则对每个 batch 独立地做最大值归一化；否则对所有点共用一个最大值。
    
    参数
    ----
    points : ndarray, shape=(N,3) 或 (B,N,3)
    resize : {'exp', 'sqrt', 'frac'} 或 None
    m : float
    n : float

    返回
    ----
    cyl : ndarray, 同 points 形状，最后一维是 (rho, phi, theta)
    """
    pts = np.asarray(points)
    if pts.ndim not in (2, 3) or pts.shape[-1] != 3:
        raise ValueError("仅支持形状 (N,3) 或 (B,N,3) 的输入。")
    
    # 拆分坐标
    x = pts[..., 0]
    y = pts[..., 1]
    z = pts[..., 2]

    # 计算 rho, phi, theta
    rho = np.sqrt(x**2 + y**2)                         # (..., N)
    phi = np.arctan2(y, x + 1e-9)
    phi[phi < 0] += 2 * math.pi                        # 归到 [0, 2π)
    # theta = np.arccos(z / np.where(rho == 0, 1.0, rho))
    theta = np.arccos(np.clip(z / np.where(rho == 0, 1.0, rho), -1.0, 1.0))

    # 非线性缩放

    rho_t = power_transform(rho,quant_size,alpha)

    # 归一化：如果是 (B,N) 形状，axis=1；否则全局 scalar
    if pts.ndim == 3:
        # rho.shape == (B, N)
        orig_max   = rho.max(axis=1, keepdims=True)      # (B,1)
        scaled_max = rho_t.max(axis=1, keepdims=True)    # (B,1)
    else:
        # rho 是 (N,)
        orig_max   = rho.max()                           # scalar
        scaled_max = rho_t.max()                         # scalar

        rho = rho_t * (orig_max / (scaled_max + 1e-9))

    # 重组输出，最后一维放 (rho, phi, theta)
    if pts.ndim == 2:
        return np.stack([rho, phi, theta], axis=1)      # (N,3)
    else:
        return np.stack([rho, phi, theta], axis=2)      # (B,N,3)


def cylin2cart(points, quant_size, alpha):
    """
    将 坐标 (rho, phi, theta) 还原到 Cartesian (x, y, z)。
    如果 resize 不为 None，且输入是 (B, N, 3)，则对每个 batch 分别：
      1. 从归一化后的 rho 反推 orig_max = max(rho)
      2. 按 forward 映射公式计算 scaled_max
      3. rho_t = rho_norm * (scaled_max / orig_max)
      4. 对 rho_t 做逆映射，得到原始 rho
    对于 (N,3)，则用全局最大值。

    参数
    ----
    points : ndarray, shape=(N,3) or (B,N,3)
    resize : {'exp','sqrt','frac'} or None
    m, n    : 同 forward 中的缩放参数

    返回
    ----
    xyz : ndarray, 同 points 形状，最后一维为 (x,y,z)
    """
    pts = np.asarray(points)
    if pts.ndim not in (2, 3) or pts.shape[-1] != 3:
        raise ValueError("仅支持 (N,3) 或 (B,N,3) 的输入。")

    # 拆分
    rho_norm = pts[..., 0]
    phi       = pts[..., 1]
    theta     = pts[..., 2]

    # 先反归一化再做逆映射

    eps = 1e-9
    # 1) 计算 orig_max（归一化后 rho 的最大值）
    if pts.ndim == 3:
        # (B, N)
        orig_max   = rho_norm.max(axis=1, keepdims=True)    # (B,1)
    else:
        orig_max   = rho_norm.max()                         # scalar

    # 2) 对应的 scaled_max = max_forward_mapping(orig_max)
    def calc_scaled_max(rho_m):
        return power_transform(rho_m,quant_size,alpha)

    scaled_max = calc_scaled_max(orig_max)

    # 3) 反归一化：rho_t = rho_norm * (scaled_max / orig_max)
    rho_t = rho_norm * (scaled_max / (orig_max + eps))

    # 4) 逆映射得到原始 rho
    rho = power_inverse(rho_t,quant_size, alpha)


    # 最后从 (rho, phi, theta) 计算 (x,y,z)
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    # 注意 forward 时用了 theta = arccos(z / rho)
    # => z = rho * cos(theta)
    z = rho * np.cos(theta)

    # 重组输出
    if pts.ndim == 2:
        return np.stack([x, y, z], axis=1)     # (N,3)
    else:
        return np.stack([x, y, z], axis=2)     # (B,N,3)

def cart2spher(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x + 1e-9)
    theta = np.arccos(z / rho)
    return np.vstack((rho, phi, theta)).transpose(1, 0)


def spher2cart(points):
    rho, phi, theta = points[:, 0], points[:, 1], points[:, 2]
    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)
    
    return np.vstack((x, y, z)).transpose(1, 0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="kitti", choices=["kitti", "ford", 'nuscenes'])
    parser.add_argument("--ori_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--parts", type=str, default="-1/-1")
    parser.add_argument("--cylin", action="store_true", help="whether using cylindrical coordinate")
    parser.add_argument("--spher", action="store_true", help="whether using spherical coordinate")
    return parser.parse_args()