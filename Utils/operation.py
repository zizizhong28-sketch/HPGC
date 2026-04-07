import torch
from pytorch3d.ops.knn import knn_points
import math


def build_knn_indices(points, k, d): # M, K, 3
    _, idx, _ = knn_points(points, points, K= k * d, return_sorted=True) # M, K, k
    return idx[:, :,0: k * d :d]

from pytorch3d.ops.sample_farthest_points import sample_farthest_points
def NeighborSample(batch_x, K, no_anchor=False, ratio=1.5):
    _, N, _ = batch_x.shape
    M = N*2//K
    # Sampling
    if N < 10000 or no_anchor:
        bones = sample_farthest_points(batch_x, K=M)[0] # (1, M, 3)
    else:
        sample_anchor = batch_x.clone()[:, torch.randperm(N)[:M*16], :]
        bones = sample_farthest_points(sample_anchor, K=M)[0] # (1, M, 3)
    # Query
    _, _, local_windows = knn_points(bones, batch_x, K=int(K*ratio), return_nn=True)
    bones, local_windows = bones[0], local_windows[0]
    return bones, local_windows

def shuffle_indices(points, ref_points):
    """
    Shuffle point indices based on nearest neighbor to reference points.

    Args:
        points: Source points (N, 3)
        ref_points: Reference points (M, 3)

    Returns:
        Index tensor mapping points to nearest reference
    """
    dist = torch.cdist(points.cpu(), ref_points.cpu())
    cloest_idx = torch.argmin(dist, dim=0).cuda()
    return cloest_idx

def self_distance(pos):
    """
    Compute self-distance for each point in the point cloud.

    Args:
        pos: Input points (B, N, 3)

    Returns:
        Self-distance for each point (B, N)
    """
    dist = knn_points(pos, pos, K=2, return_nn=False).dists[:, :, 1]
    dist = torch.sqrt(dist)
    return dist

def gaussian_feature(feature, mu, sigma):
    sigma = sigma.clamp(1e-10, 1e10)
    total_bits = torch.tensor(0.0, device=feature.device)
    probs = []
    if len(mu.shape)==2:
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
    elif len(mu.shape)==3:
        distri_num = mu.shape[1]
        feature = feature.view(feature.shape[0], distri_num, feature.shape[1] // distri_num)
        for i in range(distri_num):
            gaussian = torch.distributions.laplace.Laplace(mu[:, i, :], sigma[:, i, :])
            temp_probs = gaussian.cdf(feature[:, i] + 0.5) - gaussian.cdf(feature[:, i] - 0.5)
            probs.append(temp_probs)
            total_bits += torch.sum(torch.clamp(-1.0 * torch.log(temp_probs + 1e-10) / math.log(2.0), 0, 50))
        probs = torch.stack(probs)  # (B, distri_num, N)
    return total_bits, probs


def cdf_range(mu, sigma, L):
    M, d = sigma.shape
    mu = mu.unsqueeze(-1).repeat(1, 1, L)
    sigma = sigma.unsqueeze(-1).repeat(1, 1, L).clamp(1e-10, 1e10)
    gaussian = torch.distributions.laplace.Laplace(mu, sigma)
    flag = torch.arange(0, L).to(sigma.device).view(1, 1, L).repeat((M, d, 1))
    cdf = gaussian.cdf(flag + 0.5)

    spatial_dimensions = cdf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=cdf.dtype, device=cdf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    cdf_with_0 = cdf_with_0.clamp(max=1.)
    
    return cdf_with_0


def quantize_values(cdf_float, needs_normalization):
    """
    Quantize floating-point CDF to int16.

    Adapted from torchac library.
    """
    Lp = cdf_float.shape[-1]
    factor = torch.tensor(
        2, dtype=torch.float32, device=cdf_float.device).pow_(16)
    new_max_value = factor
    if needs_normalization:
        new_max_value = new_max_value - (Lp - 1)
    cdf_float = cdf_float.mul(new_max_value)
    cdf_float = cdf_float.round()
    cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
    if needs_normalization:
        r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
        cdf.add_(r)
    return cdf

def AdaptiveAlign(local_windows, bones):
    n_local_windows = local_windows - bones.unsqueeze(-2)  # (M, K, 3)
    sampled_self_dist = self_distance(bones.unsqueeze(0))[0].view(-1, 1, 1) # -> (M, 1, 1)
    sampled_self_dist = sampled_self_dist[sampled_self_dist[:, 0, 0] != 0]
    sampled_self_dist = sampled_self_dist.mean()
    n_local_windows = n_local_windows / sampled_self_dist # -> (M, K, 3)
    return n_local_windows


def InverseAlign(n_local_windows, bones):
    sampled_self_dist = self_distance(bones.unsqueeze(0))[0].view(-1, 1, 1) # -> (M, 1, 1)
    sampled_self_dist = sampled_self_dist[sampled_self_dist[:, 0, 0] != 0]
    sampled_self_dist = sampled_self_dist.mean()
    n_local_windows = n_local_windows * sampled_self_dist # -> (M, K, 3)
    local_windows = n_local_windows + bones.unsqueeze(-2)
    return local_windows