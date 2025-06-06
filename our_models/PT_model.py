import torch
from torch import nn

from utils.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL1
from our_models.model_Transformer import PCTransformer
from our_models.build import MODELS


def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc


class Fold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2

@MODELS.register_module()
class PT_model(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.trans_dim
        self.knn_layer = config.knn_layer
        self.num_pred = config.num_pred
        self.num_query = config.num_query

        self.fold_step = int(pow(self.num_pred//self.num_query, 0.5) + 0.5)
        self.base_model = PCTransformer(in_chans = 3, embed_dim = self.trans_dim, depth = [6, 8], drop_rate = 0., num_query = self.num_query, knn_layer = self.knn_layer)
        
        self.foldingnet = Fold(self.trans_dim, step = self.fold_step, hidden_dim = 256)  # rebuild a cluster point

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        # Adjust the input dimension to match the concatenated feature size
        # Input will be point_feature (trans_dim) + coarse (3)
        self.reduce_map = nn.Linear(self.trans_dim + 3, self.trans_dim)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()

    def get_loss(self, ret, gt):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)
        return loss_coarse, loss_fine

    def forward(self, xyz):
        coarse, point_feature = self.base_model(xyz) # B 128 3, B 128 512
        # print(coarse.size())
        # print(fps_xyz.size())

        B, M, _ = coarse.shape

        # global_feature = self.increase_dim(point_feature.transpose(1,2)).transpose(1,2) # B M 1024
        # global_feature = torch.max(global_feature, dim=1)[0] # B 1024

        # print(global_feature.size())
        # print(point_feature.size())
        # print(coarse.size())

        # Fix dimension issues by properly preparing tensors for concatenation
        # Get feature dimensions - need to handle actual shape of point_feature
        # Print for debugging
        # print(f"point_feature shape: {point_feature.shape}")
        
        # Properly reshape point_feature based on its actual dimensions
        if len(point_feature.shape) == 3:  # If shape is [B, N, C]
            B, N, C = point_feature.shape
            point_feat_expanded = point_feature  # Already in correct shape
        else:  # If shape is [B, C]
            # Repeat point_feature for each point in coarse
            point_feat_expanded = point_feature.unsqueeze(1).expand(-1, M, -1)  # B, M, C
        
        # Concatenate with coarse coordinates
        rebuild_feature = torch.cat([
            point_feat_expanded,  # B, M, feat_dim
            coarse               # B, M, 3
        ], dim=-1)  # B, M, feat_dim+3

        rebuild_feature = self.reduce_map(rebuild_feature.reshape(B*M, -1)) # B*M C
        # # NOTE: try to rebuild pc

        # NOTE: foldingNet
        # Add debug prints to understand tensor shapes
        print(f"rebuild_feature shape: {rebuild_feature.shape}")
        
        # Apply foldingnet to get relative coordinates
        relative_xyz = self.foldingnet(rebuild_feature)
        print(f"relative_xyz after foldingnet: {relative_xyz.shape}")
        
        relative_xyz = relative_xyz.reshape(B, M, 3, -1)    # B M 3 S
        print(f"relative_xyz after reshape: {relative_xyz.shape}")
        print(f"coarse shape: {coarse.shape}")
        
        # The issue is that coarse is actually a feature vector of size 384, not just XYZ coordinates
        # Let's remove the debug prints now that we understand the shapes
        
        # For folding and rebuilding, we don't need to add the full feature vector
        # We can just use the relative coordinates from the folding network directly
        fold_step = self.fold_step
        S = fold_step * fold_step  # Typically 25 (5x5 grid) - matches 4th dimension of relative_xyz
        
        # Reshape just the xyz part of the tensor - no addition needed
        rebuild_points = relative_xyz.transpose(2, 3).reshape(B, -1, 3)  # B, M*S, 3
        print(f"rebuild_points shape: {rebuild_points.shape}")
        # print(rebuild_points.size())
        # cat the input
        # inp_sparse = fps(xyz, self.num_query)
        # coarse_point_cloud = torch.cat([coarse_point_cloud, inp_sparse], dim=1).contiguous()
        # rebuild_points = torch.cat([rebuild_points, xyz],dim=1).contiguous()
        # print(coarse.size())
        # print(rebuild_points.size())
        ret = (coarse, rebuild_points)
        return ret



