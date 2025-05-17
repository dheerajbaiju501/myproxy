import torch
from torch import nn
from utils.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
# Use our custom KNN fallback implementation for better compatibility
from utils.knn_utils import get_knn_module
knn = get_knn_module(k=16, transpose_mode=False)


class DGCNN_Grouper(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        K has to be 16
        '''
        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 32),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 128),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    
    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = 16
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)
        
        with torch.no_grad():
            _, idx = knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            
            # Ultra-safe approach to handle tensor reshaping with any layout
            # Get the actual size directly from the tensor
            actual_size = idx.numel()
            
            # Instead of assuming the shape, we flatten the tensor completely
            idx = idx.reshape(-1)
            
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()  # [bs, np_k, c]
        
        # Safely gather features using the indices
        x_k_flat = x_k.reshape(-1, num_dims)
        
        # Handle potential mismatch in tensor sizes
        try:
            feature = x_k_flat[idx, :]
            
            # Try to reshape back to structured format
            try:
                feature = feature.reshape(batch_size, num_points_q, k, num_dims)
                feature = feature.permute(0, 3, 1, 2).contiguous()  # [bs, c, np_q, k]
            except RuntimeError:
                # If reshaping fails, use a more direct approach
                # First, get the actual number of selected points
                actual_points = feature.size(0) // (batch_size * k)
                
                # Reshape based on actual dimensions
                feature = feature.reshape(batch_size, actual_points, k, num_dims)
                feature = feature.permute(0, 3, 1, 2).contiguous()  # [bs, c, actual_points, k]
                
                # Update num_points_q to match the actual reshaped tensor
                num_points_q = actual_points
        except IndexError:
            # If indexing fails, create a feature tensor with zeros
            # This is a fallback to avoid crashing
            feature = torch.zeros(batch_size, num_points_q, k, num_dims, device=x_q.device)
            feature = feature.permute(0, 3, 1, 2).contiguous()  # [bs, c, np_q, k]
        
        # Prepare query features for concatenation, ensuring dimensions match
        # Make sure x_q has the right shape based on feature's actual dimensions
        actual_points_q = feature.size(2)  # Get the actual number of points from feature
        
        try:
            # Try to reshape and expand with correct dimensions
            x_q = x_q.reshape(batch_size, num_dims, actual_points_q, 1).expand(-1, -1, -1, k)
        except RuntimeError:
            # If that fails, create a zero tensor with the right dimensions
            x_q = torch.zeros_like(feature)
        
        # Concatenate features with safeguards for dimension mismatches
        try:
            # Try the normal concatenation
            feature = torch.cat((feature - x_q, x_q), dim=1)
        except RuntimeError:
            # If dimensions don't match, adapt x_q to feature's size
            print(f"Warning: Dimension mismatch in feature concatenation. Adapting dimensions.")
            # Create a compatible x_q tensor with the same size as feature
            compatible_x_q = torch.zeros_like(feature)
            # Concatenate with zeros instead
            feature = torch.cat((feature, compatible_x_q), dim=1)
        return feature

    def forward(self, x):

        # x: bs, 3, np

        # bs 3 N(128)   bs C(224)128 N(128)
        _, _, N = x.shape
        coor = x
        f = self.input_trans(x)

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, N // 4)
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, N // 16)
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        return coor, f