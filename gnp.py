"""
GNP: Global Normalized PSNR - A Full-Reference LPC-Specific Quality Metric

This metric combines density-adaptive global weighting with regional keypoint 
analysis to homogenize error assessment across varying point densities while 
preserving sensitivity to local geometric fidelity.

Key components:
- Global Evaluation: Adaptive distance coefficient (alpha) and density coefficient (beta)
                    to balance error contributions across the scene
- Local Evaluation: Adaptive Fast Euclidean Cluster (AFEC) on filtered object clustering
                    for precise local fidelity measurement
"""

from sklearn.linear_model import RANSACRegressor
import numpy as np
import math
from scipy.spatial import cKDTree
from scipy.linalg import lstsq


def xyz2sph(points):
    """Convert Cartesian coordinates to spherical coordinates."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x + 1e-9)
    phi[np.where(phi < 0)[0]] += 2 * math.pi
    theta = np.arccos(z / rho)
    return np.vstack((rho, phi, theta)).transpose(1, 0)


def sph2xyz(points):
    """Convert spherical coordinates to Cartesian coordinates."""
    rho, phi, theta = points[:, 0], points[:, 1], points[:, 2]
    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)
    return np.vstack((x, y, z)).transpose(1, 0)


class pcd_Tree():
    """KD-Tree wrapper for efficient point cloud operations."""
    
    def __init__(self, pcd, workers=-1):
        self.pcd = pcd
        self.tree = cKDTree(pcd)
        self.workers = workers

    def search(self, target_points, r=0.1):
        """Find all points within radius r of target points."""
        indices = np.concatenate(self.tree.query_ball_point(target_points, r), axis=0).astype(float)
        indices = indices[np.isfinite(indices)].astype(int)
        return indices

    def reg_self(self):
        """Query nearest neighbor distances for self points (k=2)."""
        distances, indices = self.tree.query(self.pcd, k=2)
        return distances[:, -1], indices[:, -1]

    def reg_else_pcd(self, target_points, k=1):
        """Query nearest neighbor distances to target points."""
        distances, indices = self.tree.query(target_points, k=k)
        if k == 1:
            return distances, indices
        else:
            return distances[:, -1], indices[:, -1]

    def reg_outliner(self, neighbor_points, std=2):
        """Identify outlier points based on k-th nearest neighbor distance."""
        distances, indices = self.tree.query(self.pcd, k=neighbor_points)
        kth_distances = distances[:, -1]
        
        mean_distance = np.mean(kth_distances)
        std_distance = np.std(kth_distances)
        threshold = mean_distance + std * std_distance
        
        inlier_mask = kth_distances < threshold
        inlier = self.pcd[inlier_mask]
        return inlier

    def plane_distance(self, points, plane_params):
        """Calculate point-to-plane distances."""
        A, B, C, D = plane_params
        return np.abs(A * points[:, 0] + B * points[:, 1] + C * points[:, 2] + D) / np.sqrt(A**2 + B**2 + C**2)


def FEC(points, co, min_n):
    """
    Fast Euclidean Clustering (FEC) for point cloud segmentation.
    
    Args:
        points: Point cloud array (N, 3)
        co: Clustering epsilon coefficient
        min_n: Minimum points in neighborhood
        
    Returns:
        centroids: List of cluster centroids
        labels: Cluster label for each point
    """
    tree = cKDTree(points)
    num_points = points.shape[0]
    labels = [0] * num_points
    segLab = 1

    for i in range(num_points):
        if labels[i] == 0:
            eps = np.linalg.norm(points[i], ord=2) * co
            _, indices = tree.query(points[i], eps=eps, k=min_n)
            minSegLab = segLab
            for j in indices:
                if (labels[j] > 0) and (labels[j] < minSegLab):
                    minSegLab = labels[j]
            for j in indices:
                tempLab = labels[j]
                if tempLab > minSegLab:
                    for k in range(num_points):
                        if labels[k] == tempLab:
                            labels[k] = minSegLab
                labels[j] = minSegLab
        segLab += 1
    
    centroids = []
    for i in np.unique(labels):
        centroids.append(np.mean(points[labels == i], axis=0))
    return centroids, np.array(labels)


def plane_fitting_ransac(points, percentage=40, distance_threshold=0.1):
    """
    Ground plane fitting using RANSAC.
    
    Args:
        points: Input point cloud array (N, 3)
        percentage: Z-axis percentile threshold for initial filtering
        distance_threshold: Distance threshold for plane inliers
        
    Returns:
        nonground_mask: Boolean mask of non-ground points
    """
    z_threshold = np.percentile(points[:, 2], percentage)
    
    candidate_mask = points[:, 2] < z_threshold
    candidates = points[candidate_mask]
    
    if len(candidates) < 3:
        return np.zeros(len(points), dtype=bool)
    
    X = candidates[:, :2]
    y = candidates[:, 2]
    
    ransac = RANSACRegressor()
    ransac.fit(X, y)
    
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_
    
    numerator = np.abs(a * points[:, 0] + b * points[:, 1] - points[:, 2] + c)
    denominator = np.sqrt(a**2 + b**2 + 1)
    distances = numerator / denominator
    
    nonground_mask = distances > distance_threshold
    return nonground_mask


def plane_fitting(points, distance_threshold=0.5):
    """Plane fitting using least squares."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    A = np.vstack((x, y, np.ones_like(x))).T
    [a, b, c] = lstsq(A, z)[0]
    distances = np.abs((a * x + b * y - z + c))
    denominator = np.sqrt(a**2 + b**2 + 1)
    distances = distances / denominator
    
    nonground_mask = distances > distance_threshold
    return points[nonground_mask]


def psnr(origin_pcd, recon_pcd, peak_value=59.7):
    """
    Standard PSNR for point clouds using bidirectional Chamfer distance.
    
    Args:
        origin_pcd: Original point cloud with .points attribute
        recon_pcd: Reconstructed point cloud with .points attribute
        peak_value: Peak value for PSNR calculation
        
    Returns:
        PSNR value or 'No loss' if MSE is zero
    """
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
        return 10 * np.log10((peak_value**2) / mse)


def gnp(origin_pcd, recon_pcd, peak_value=59.7):
    """
    GNP: Global Normalized PSNR
    
    A full-reference LPC-specific quality metric that combines:
    1. Global Evaluation: Density-adaptive weighting based on radial distance
       and local point density
    2. Local Evaluation: Regional keypoint analysis for precise local fidelity
    
    This metric addresses the bias of standard PSNR in evaluating non-uniform
    LiDAR point clouds by:
    - Using distance coefficient (alpha = 1/r) to compensate for radius-dependent sparsity
    - Using density coefficient (beta = log(eta+1)+eps) to balance local point concentration
    - Combining global and local evaluations for comprehensive quality assessment
    
    Paper: HPGC (IEEE TPAMI 2026), Section IV-E "Learning"
    
    Args:
        origin_pcd: Original point cloud (numpy array, Nx3)
        recon_pcd: Reconstructed point cloud (numpy array, Nx3)
        peak_value: Peak value for PSNR calculation (default: 59.7 for KITTI, 30000 for Ford)
        
    Returns:
        GNP value combining global and local evaluations
    """
    scale = 32
    ori_pcd = np.asarray(origin_pcd)
    recon_pcd = np.asarray(recon_pcd)

    ori_tree = pcd_Tree(ori_pcd)
    recon_tree = pcd_Tree(recon_pcd)

    # ========== Global Evaluation ==========
    # Distance coefficient: Points farther from origin have larger coefficients,
    # indicating lower importance (compensate for radius-dependent sparsity)
    dist_rec2ori, _ = ori_tree.reg_else_pcd(recon_pcd)
    r_co_rec = np.log(np.linalg.norm(recon_pcd, ord=2, axis=1) + 0.0001) + 1e-4

    dist_ori2rec, _ = recon_tree.reg_else_pcd(ori_pcd)
    r_co_ori = np.log(np.linalg.norm(ori_pcd, ord=2, axis=1) + 0.0001) + 1e-4

    # Density coefficient: Smaller nearest neighbor distance means higher density,
    # which results in smaller density coefficient and lower importance
    near_ori, _ = ori_tree.reg_self()
    near_rec, _ = recon_tree.reg_self()
    den_ori = np.log10(near_ori + 1) + 1e-4
    den_rec = np.log10(near_rec + 1) + 1e-4

    # Apply adaptive weighting
    dist_rec2ori = dist_rec2ori / r_co_rec * den_rec
    dist_ori2rec = dist_ori2rec / r_co_ori * den_ori

    mse1 = np.mean(dist_rec2ori**2)
    mse2 = np.mean(dist_ori2rec**2)
    mse = np.min([mse1, mse2])
    psnr_global = 10 * np.log10((peak_value**2) / mse)

    # ========== Local Evaluation ==========
    # Focus on semantically meaningful objects using outlier detection
    rest = ori_tree.reg_outliner(30, std=1)
    threshold = np.percentile(rest[:, 2], 50)
    rest = rest[rest[:, 2] > threshold]
    local_peak = peak_value / scale
    
    centroids, _ = FEC(rest, 0.1, int(rest.shape[0]**(1/3)))
    if len(centroids) == 0:
        return psnr_global
    
    idx_ori_local = np.unique(ori_tree.search(centroids, local_peak))
    if len(idx_ori_local) == 0:
        return psnr_global
    
    ori_local = ori_tree.pcd[idx_ori_local]
    dist1_local, _ = recon_tree.reg_else_pcd(ori_local)
    mse_local = np.mean(dist1_local**2) + 1e-6
    psnr_local = 10 * np.log10((local_peak**2) / mse_local)
    
    return (psnr_global + psnr_local) / 2


# Backward compatibility alias
def r_psnr(origin_pcd, recon_pcd, peak_value=59.7):
    """
    Backward compatibility alias for gnp().
    
    DEPRECATED: Please use gnp() instead for consistency with the paper.
    """
    return gnp(origin_pcd, recon_pcd, peak_value)
