import torch
import numpy as np

def furthest_point_sample_fallback(xyz, npoint):
    """
    A PyTorch-native implementation of furthest point sampling.
    
    Args:
        xyz: (B, N, 3) tensor containing point coordinates
        npoint: int, number of points to sample
        
    Returns:
        centroids: (B, npoint) tensor containing indices of sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape
    
    # Ensure we don't sample more points than available
    npoint = min(npoint, N)
    
    # Initialize output tensor containing indices of sampled points
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    
    # Randomly choose the first point
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        # Use the farthest point as the current centroid
        centroids[:, i] = farthest
        
        # Get the coordinates of the current centroid
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        
        # Compute distances between centroid and all points
        dist = torch.sum((xyz - centroid) ** 2, -1)
        
        # Update distances with minimum between current and previous
        mask = dist < distance
        distance[mask] = dist[mask]
        
        # Find the point with the largest distance as the next centroid
        farthest = torch.max(distance, -1)[1]
    
    return centroids

def patch_furthest_point_sample():
    """
    Patch the furthest_point_sample function in pointnet2_utils with our fallback.
    
    This function will temporarily override the furthest_point_sample function
    in utils.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils with our PyTorch-native
    implementation. It does so by monkey-patching the module.
    """
    from utils.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
    
    # Store the original function
    original_fps = pointnet2_utils.furthest_point_sample
    
    # Create a wrapper to maintain API compatibility
    def fps_wrapper(xyz, npoint):
        """Wrapper around the FPS function to maintain compatibility"""
        try:
            # Try to use the original CUDA implementation first
            return original_fps(xyz, npoint)
        except Exception as e:
            print(f"CUDA FPS failed: {e}")
            print("Using PyTorch native FPS fallback instead")
            return furthest_point_sample_fallback(xyz, npoint)
    
    # Replace the function in the module
    pointnet2_utils.furthest_point_sample = fps_wrapper
    
    print("Patched furthest_point_sample with PyTorch fallback implementation")
