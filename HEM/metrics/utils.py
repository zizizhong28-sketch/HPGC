from sklearn.linear_model import RANSACRegressor
import numpy as np
import math
from scipy.spatial import cKDTree
from scipy.linalg import lstsq

def xyz2sph(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x + 1e-9)
    phi[np.where(phi < 0)[0]] += 2 * math.pi
    theta = np.arccos(z / rho)
    return np.vstack((rho, phi, theta)).transpose(1, 0)

def sph2xyz(points):
    rho, phi, theta = points[:, 0], points[:, 1], points[:, 2]
    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)
    return np.vstack((x, y, z)).transpose(1, 0)

class pcd_Tree():
    def __init__(self, pcd, workers=-1):
        self.pcd = pcd
        self.tree = cKDTree(pcd)
        self.workers = workers

    def search(self, target_points, r=0.1):
        indices = np.concatenate(self.tree.query_ball_point(target_points, r),axis=0).astype(float)
        indices = indices[np.isfinite(indices)].astype(int)
        return indices

    def reg_self(self):
        # 查询最近邻，k=2表示查找最近的两个点
        distances, indices = self.tree.query(self.pcd, k=2)
        return distances[:,-1], indices[:,-1]

    def reg_else_pcd(self, target_points, k = 1):

        # 查询最近邻，k=1表示查找最近的一个点
        distances, indices = self.tree.query(target_points, k=k)
        if k == 1:
            return distances, indices
        else:
            return distances[:,-1], indices[:,-1]
    
    def reg_outliner(self, neighbor_points, std = 2):
        # 查询最近邻，k=1表示查找最近的一个点
        distances, indices = self.tree.query(self.pcd, k=neighbor_points)
        kth_distances = distances[:, -1]
    
        # 计算统计阈值
        mean_distance = np.mean(kth_distances)
        std_distance = np.std(kth_distances)
        threshold = mean_distance + std * std_distance
        
        # 生成内点掩码
        inlier_mask = kth_distances < threshold
        inlier = self.pcd[inlier_mask]
        return inlier

    def plane_distance(self, points, plane_params):

        A, B, C, D = plane_params
        return np.abs(A*points[:,0] + B*points[:,1] + C*points[:,2] + D) / np.sqrt(A**2 + B**2 + C**2)

def FEC(points, co, min_n):
    tree = cKDTree(points)
    num_points = points.shape[0]
    labels = [0] * num_points  # initalize all point label as 0
    segLab = 1  # Segment label

    for i in range(num_points):
        if labels[i] == 0:  # if Pi.lable =0

            eps = np.linalg.norm(points[i],ord=2)*co

            _ , indices = tree.query(points[i], eps = eps, k = min_n)  # find all points in eps-neighborhood of Pi
            minSegLab = segLab
            for j in indices:
                # if Nonzero(Pnn.lab)
                if (labels[j] > 0) and (labels[j] < minSegLab):
                    minSegLab = labels[j]  # minSegLab = min(N0nzero(Pnn.lab),SegLab)
            for j in indices:  # foreach pj in Pnn do
                tempLab = labels[j]
                if tempLab > minSegLab:  # if pj.lab > minSeglab then
                    for k in range(num_points):  # foreach pk.lab in P do
                        if labels[k] == tempLab:  # if pk.lab = Pj.lab then
                            labels[k] = minSegLab  # pk.lab = minSegLab

                labels[j] = minSegLab  # 将所有邻近点分类

        segLab += 1
    centoid = []
    for i in np.unique(labels):
        centoid.append(np.mean(points[labels == i], axis=0))
    return centoid, np.array(labels)  

def plane_fitting_ransac(points, percentage = 40, distance_threshold=0.1):
        z_threshold = np.percentile(points[:,2],percentage)
        """
        :param points: 输入点云数组，形状为[n, 3]
        :param z_threshold: 初始高度筛选阈值, 默认0.2米
        :param distance_threshold: 平面距离阈值, 默认0.1米
        :return: 返回非地面点的索引
        """
        
        candidate_mask = points[:,2] < z_threshold
        candidates = points[candidate_mask]
        
        if len(candidates) < 3:
            return np.zeros(len(points), dtype=bool)
        
        # 2. 使用RANSAC拟合平面模型（z = ax + by + c）
        X = candidates[:, :2]  # 使用x,y作为特征
        y = candidates[:, 2]   # 预测z值
        
        ransac = RANSACRegressor()
        ransac.fit(X, y)
        
        # 3. 计算平面方程参数
        a, b = ransac.estimator_.coef_
        c = ransac.estimator_.intercept_
        
        # 4. 计算所有点到平面的距离
        numerator = np.abs(a * points[:, 0] + b * points[:, 1] - points[:, 2] + c)
        denominator = np.sqrt(a**2 + b**2 + 1)
        distances = numerator / denominator
        
        # 5. 根据距离阈值确定地面点
        nonground_mask = distances > distance_threshold
        nonground_points = points[nonground_mask]
        return nonground_points

def plane_fitting(points, distance_threshold=0.5):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    A = np.vstack((x, y, np.ones_like(x))).T
    [a, b, c] = lstsq(A, z)[0]
    distances = np.abs( (a * x + b * y - z + c))
    denominator = np.sqrt(a**2 + b**2 + 1)
    distances = distances / denominator
    
    # 5. 根据距离阈值确定地面点
    nonground_mask = distances > distance_threshold
    nonground_points = points[nonground_mask]
    return nonground_points


def psnr(origin_pcd, recon_pcd, peak_value = 59.7):
    ori_pcd = np.asarray(origin_pcd.points)
    recon_pcd = np.asarray(recon_pcd.points)

    ori_tree = pcd_Tree(ori_pcd)
    recon_tree = pcd_Tree(recon_pcd)
    dist_rec2ori, _ = ori_tree.reg_else_pcd(recon_pcd)
    dist_ori2rec, _ = recon_tree.reg_else_pcd(ori_pcd)
    dist_rec2ori = np.asarray(dist_rec2ori)
    dist_ori2rec = np.asarray(dist_ori2rec)
    mse1 = np.mean((dist_rec2ori)**2)
    mse2 = np.mean((dist_ori2rec)**2)
    mse = np.min([mse1, mse2])
    if mse == 0:
        return 'No loss'
    else:
        psnr = 10 * np.log10((peak_value**2)/mse)
        return psnr

def gnp(origin_pcd, recon_pcd, peak_value = 59.7):
    """
    GNP: Global Normalized PSNR - Full-Reference LPC Quality Metric
    
    Combines density-adaptive global weighting with regional keypoint analysis.
    Paper: HPGC (IEEE TPAMI 2026)
    """
    scale = 32
    ori_pcd = np.asarray(origin_pcd)
    recon_pcd = np.asarray(recon_pcd)

    ori_tree = pcd_Tree(ori_pcd)
    recon_tree = pcd_Tree(recon_pcd)

    # 分布越靠外，距离系数越大，重要性越低
    dist_rec2ori, _ = ori_tree.reg_else_pcd(recon_pcd) 
    r_co_rec = np.log(np.linalg.norm(recon_pcd, ord=2, axis=1)+0.0001)+1e-4

    dist_ori2rec, _ = recon_tree.reg_else_pcd(ori_pcd)
    r_co_ori = np.log(np.linalg.norm(ori_pcd, ord=2, axis=1)+0.0001)+1e-4

    # 最近邻距离越小， dense越大，密度系数越小， 重要性越低
    near_ori, _ = ori_tree.reg_self()
    near_rec, _ = recon_tree.reg_self()
    den_ori = np.log10(near_ori+1)+1e-4
    den_rec = np.log10(near_rec+1)+1e-4

    dist_rec2ori = dist_rec2ori/r_co_rec*den_rec
    dist_ori2rec = dist_ori2rec/r_co_ori*den_ori

    mse1 = np.mean(dist_rec2ori**2)
    mse2 = np.mean(dist_ori2rec**2)

    mse = np.min([mse1, mse2])
    psnr_global = 10 * np.log10((peak_value**2)/mse)

    rest = ori_tree.reg_outliner(30, std=1)
    threshold = np.percentile(rest[:,2], 50)
    rest = rest[rest[:,2] > threshold]
    local_peak = peak_value/scale
    centoid , _ = FEC(rest, 0.1, int(rest.shape[0]**(1/3)))
    if len(centoid) == 0:
        return psnr_global
    idx_ori_local = np.unique(ori_tree.search(centoid, local_peak))
    if len(idx_ori_local) == 0:
        return psnr_global
    ori_local = ori_tree.pcd[idx_ori_local]
    dist1_local, _ = recon_tree.reg_else_pcd(ori_local)
    mse_local = np.mean(dist1_local**2)+1e-6
    psnr_local = 10 * np.log10((local_peak**2)/mse_local)
    return (psnr_global+psnr_local)/2


def r_psnr(origin_pcd, recon_pcd, peak_value = 59.7):
    """
    Backward compatibility alias for gnp().
    
    DEPRECATED: Please use gnp() instead for consistency with the paper.
    """
    return gnp(origin_pcd, recon_pcd, peak_value)

