"""
LWC: Learning-based Point Cloud Compression
local prediction module for point cloud geometry coding.
"""

import torch
import torch.nn as nn

import Utils.operation as op
from Utils.nn import FoldNetDecoder, ResidualEncoder, LocalParam


class LWC(nn.Module):
    """
    Learning-based Wedge-based Compression network.

    Predicts local points around skeleton bones using:
    - Feature squeeze: Extract features from aligned windows
    - Entropy model: Learn probability distribution for coding
    - local param: Parameterize distribution for rate control
    - Feature stretch & point generator: Reconstruct local points
    """

    def __init__(self, channel, bottleneck_channel, distri_num=2, dilated_list=4):
        super(LWC, self).__init__()

        self.feature_squeeze = ResidualEncoder(input_dim=3, n_layers=3, embed_dim=channel, output_dim=2 * channel)
        self.maxpool = nn.AdaptiveAvgPool1d(1)
        self.feature_sample = nn.Linear(channel*2, bottleneck_channel)

        self.entropy_Model = ResidualEncoder(input_dim=3, n_layers=3, embed_dim=channel, output_dim=channel)
        self.parameterization = LocalParam(input_dim=channel, embed_dim=channel, bottleneck_channel=bottleneck_channel, distri_num = distri_num)

        self.feature_stretch = ResidualEncoder(input_dim=bottleneck_channel, n_layers=3, embed_dim=channel, output_dim=channel)
        self.point_generator = FoldNetDecoder(channel=channel, fold_channel=8, R_max=256, r=4)

    def propagate(self, batch_x, K):
        """
        Forward pass: encode and decode point cloud.

        Args:
            batch_x: Input point cloud (1, N, 3)
            K: Number of points per bone

        Returns:
            Tuple of (reconstructed points, bitrate)
        """
        N = batch_x.shape[1]

        # Sample skeleton bones and local windows
        bones, local_windows = op.NeighborSample(batch_x, K)
        aligned_windows = op.AdaptiveAlign(local_windows, bones)  # M, K, 3

        # Feature extraction
        knn_idx_list = op.build_knn_indices(aligned_windows, 16, 4)
        feature = self.feature_squeeze(x=aligned_windows, knn_idx_list=knn_idx_list)  # M, K, C
        max_pooled_feature = self.maxpool(feature.transpose(-1, -2)).squeeze(-1)  # M, C

        feature = self.feature_sample(max_pooled_feature)

        # Add uniform noise for stochastic quantization
        quantized_compact_fea = feature + torch.nn.init.uniform_(
            torch.zeros_like(feature), -0.5, 0.5
        )

        # Entropy model: learn distribution for compression
        knn_idx_list = op.build_knn_indices(bones.unsqueeze(0), 8, 4)
        feature = self.entropy_Model(x=bones.unsqueeze(0), knn_idx_list=knn_idx_list)  # M, c*2

        mu, sigma = self.parameterization(feature)

        # Compute rate (bits) for the quantized features
        bitrate, _ = op.gaussian_feature(quantized_compact_fea, mu.squeeze(0), sigma.squeeze(0))
        bitrate = bitrate / N

        # Reconstruct surface points
        feature = self.feature_stretch(
            quantized_compact_fea.unsqueeze(0), bones.unsqueeze(0), knn_idx_list
        ).squeeze(0)  # 1, M, C
        rec_windows = self.point_generator(feature, K)
        rec_windows = op.InverseAlign(rec_windows, bones)
        rec_batch_x = rec_windows.view(1, -1, 3)

        return rec_batch_x, bitrate