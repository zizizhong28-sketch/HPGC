import torch
import torch.nn as nn
import math

def clamp_indices(input_indices, max_idx):
    """将索引值限制在 [0, max_idx-1] 范围内"""
    return torch.clamp(input_indices, 0, max_idx - 1)


def get_graph_feature(x, k=1, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    # 计算每个批次的中心点 (batch_size, num_dims, 1)
    center = torch.mean(x, dim=2, keepdim=True)
    
    # 扩展中心点以匹配每个点 (batch_size, num_dims, num_points)
    center_expanded = center.expand(-1, -1, num_points)
    
    # 计算相对特征
    relative_feature = x - center_expanded
    
    # 组合相对特征和原始特征
    feature = torch.cat((relative_feature, x), dim=1)  # (batch_size, 2*num_dims, num_points)
    
    # 调整形状以匹配原始输出 (batch_size, 2*num_dims, num_points, k=1)
    feature = feature.unsqueeze(3).contiguous()
    
    return feature


class GeoFeatGenerator(nn.Module):
    def __init__(self, k=10, max_level=17):
        super(GeoFeatGenerator, self).__init__()
        self.k = k
        
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d((64+96)*2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d((128+64)*2, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.occ_enc = nn.Embedding(257, 16) # TODO former method is 128, temporarily set to 16 here
        self.level_enc = nn.Embedding(max_level, 4)
        self.octant_enc = nn.Embedding(9, 4)
        self.extent_enc = nn.Linear(2, 16)
        self.mlp2 = nn.Sequential(
            nn.Linear(96, 80),
            nn.LeakyReLU(),
            nn.Linear(80, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
        )
        self.edge_mlp1 = nn.Sequential(
            nn.Linear(448, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
        )
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x, extent,pos):
        bsz, csz = x.shape[:2]
        occ = x[:, :, 2::3].long()
        level = x[:, :, ::3].long()
        octant = x[:, :, 1::3].long()
        occ_embed = self.occ_enc(occ).reshape(bsz, csz, -1)
        level_embed = self.level_enc(level).reshape(bsz, csz, -1)
        octant_embed = self.octant_enc(octant).reshape(bsz, csz, -1)
        extent_embed = self.extent_enc(extent).reshape(bsz, csz, -1)
        x = torch.concat((occ_embed, level_embed, octant_embed,extent_embed),2)

        k = min(self.k, pos.shape[2])
        pos = get_graph_feature(pos, k=k)
        pos = self.conv1(pos)
        pos1 = pos.max(dim=-1, keepdim=False)[0]

        pos = get_graph_feature(torch.concat((pos1, x.transpose(1, 2)), 1), k=k) 
        pos = self.conv2(pos)
        pos2 = pos.max(dim=-1, keepdim=False)[0]

        x = self.mlp2(x)

        pos = get_graph_feature(torch.concat((pos2, x.transpose(1, 2)), 1), k=k)
        pos = self.conv3(pos)
        pos3 = pos.max(dim=-1, keepdim=False)[0]

        x = self.mlp3(x)

        ec = self.edge_mlp1(torch.concat((pos1, pos2, pos3), 1).transpose(1, 2))
        ec = self.edge_mlp2(torch.concat((pos3.transpose(1, 2), ec), 2))

        return torch.concat((x, ec), 2) # output dim: 256

    def embed_occ(self, occ):
        occ = occ.long()
        return self.occ_enc(occ)


