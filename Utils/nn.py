import torch
import torch.nn as nn
from pytorch3d.ops.knn import knn_gather
    
class FoldLayer(nn.Module):
    def __init__(self, in_channel, fold_ratio, out_channel):
        super(FoldLayer, self).__init__()
        self.fold_ratio = fold_ratio
        self.out_channel = out_channel
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, fold_ratio*out_channel),
        )

    def forward(self, fea):
        """
        Forward pass with folding operation.

        Args:
            fea: Input features (..., in_channel)

        Returns:
            Folded features (..., fold_ratio, out_channel)
        """
        output_shape = fea.shape[:-1]+(self.fold_ratio,self.out_channel,)
        fea = self.mlp(fea).reshape(output_shape)
        return fea


class FoldNetDecoder(nn.Module):
    def __init__(self, channel, fold_channel, R_max, r):
        super(FoldNetDecoder, self).__init__()
        self.R_max = R_max
        self.r = r
        self.folding_base = FoldLayer(in_channel=channel, fold_ratio=R_max, out_channel=fold_channel)
        self.folding_pro = FoldLayer(in_channel=channel+fold_channel, fold_ratio=r, out_channel=3)

    def forward(self, local_features, K):
        """
        Generate point cloud from local features.

        Args:
            local_features: Feature tensor (M, C)
            K: Number of points per bone

        Returns:
            Reconstructed points (M*K, 3)
        """
        M = local_features.shape[0]

        fea = self.folding_base(local_features)  # M, C -> M, R_max, fold_channel
        fea = fea[:, torch.randperm(self.R_max)[:K//self.r], :]

        local_features = local_features.unsqueeze(1).repeat((1, fea.shape[1], 1))
        cat_fea = torch.cat((local_features, fea), dim=-1)

        xyz = self.folding_pro(cat_fea)
        xyz = xyz.view(M, -1, 3)

        return xyz
    
class LocalBlock(nn.Module):
    def __init__(self, in_channel, mlps, relu):
        super(LocalBlock, self).__init__()

        mlps.insert(0, in_channel)
        self.mlp_Modules = nn.ModuleList()
        for i in range(len(mlps) - 1):
            if relu[i]:
                    mlp_Module = nn.Sequential(
                        nn.Linear(mlps[i], mlps[i+1]),
                        nn.ReLU(inplace=True),
                        )
            else:
                mlp_Module = nn.Sequential(
                    nn.Linear(mlps[i], mlps[i+1]),
                    )
            self.mlp_Modules.append(mlp_Module)

        self.act = nn.ELU(inplace=True)

    def forward(self, points):
        """
        Forward pass through local feature extraction.

        Args:
            points: Input point features [B, ..., N, C]

        Returns:
            Aggregated features [B, ..., D]
        """
        for m in self.mlp_Modules:
            points = m(points)  # M, K, k, C
        points = torch.sum(points, dim=-2, keepdim=False)
        return self.act(points)

class ResidualEncoder(nn.Module):
    def __init__(self, input_dim, n_layers, embed_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_layer = n_layers
        self.Encoding = nn.ModuleList()
        self.Encoding.append(LocalBlock(input_dim + 9, [embed_dim//4, embed_dim//2, embed_dim], [True, True, True]))
        for i in range(1, n_layers):
            self.Encoding.append(LocalBlock(embed_dim + 9, [embed_dim], [True]))
        self.output_emb = nn.Sequential(
            nn.Linear(self.n_layer * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, output_dim)
        )
    
    def forward(self, x, xyz=None, knn_idx_list=None):
        """
        Multi-layer residual feature encoding with KNN neighborhood.

        Args:
            x: Input points (1, M, 3)
            xyz: Position coordinates for relative features (default: same as x)
            knn_idx_list: K-nearest neighbor indices

        Returns:
            Encoded features (1, M, C)
        """
        feature_list = []
        if xyz is None:
            xyz = x
        for i in range(self.n_layer):
            feature = knn_gather(x, knn_idx_list)  # M K k C
            position = knn_gather(xyz, knn_idx_list)  # M K k 3
            substract = position - xyz.unsqueeze(-2)  # M K k 3
            distance = torch.sqrt(substract * substract)  # M K k 1
            feature = torch.concatenate([feature, position, substract, distance], dim=-1)  # 1, M, k, C+7
            if i == 0:
                x = self.Encoding[i](feature)  # 1, M, k, C+7 -> 1, M, C
            else:
                x = self.Encoding[i](feature) + x
            feature_list.append(x)
        feature = torch.concatenate(feature_list, dim=-1)  # 1, M, C * n_layer
        feature = self.output_emb(feature)  # 1, M, C
        return feature
    
class LocalParam(nn.Module):
    def __init__(self, input_dim, embed_dim, bottleneck_channel, distri_num=2):
        super().__init__()
        self.bottleneck_channel = bottleneck_channel
        self.distri_num = distri_num
        self.paramlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, bottleneck_channel * 2)
        )
    def forward(self, input):
        """
        Parameterize distribution for entropy coding.

        Args:
            input: Input features (M, K, input_dim)

        Returns:
            mu: Distribution means
            sigma: Distribution standard deviations
        """
        mu_sigma = self.paramlp(input)  # M, K, input_dim
        mu_sigma = mu_sigma.view(
            mu_sigma.shape[0], mu_sigma.shape[1],
            self.distri_num * 2,
            self.bottleneck_channel // self.distri_num
        )  # M, K, bottleneck_channel * 2
        mu = mu_sigma[:, :, ::2]
        sigma = torch.exp(mu_sigma[:, :, 1::2])
        return mu, sigma