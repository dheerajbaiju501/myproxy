import torch
import numpy as np

class KNNFallback:
    """Fallback implementation of KNN that works on any device.
    This is a PyTorch-native implementation that can be used when KNN_CUDA fails.
    """
    def __init__(self, k=16, transpose_mode=False):
        self.k = k
        self.transpose_mode = transpose_mode
    
    def forward(self, ref, query):
        """
        Args:
            ref: reference points, [B, C, N] or [B, N, C] depending on transpose_mode
            query: query points, [B, C, M] or [B, M, C] depending on transpose_mode

        Returns:
            dist: distances, [B, k, M] where k is always self.k
            idx: indices, [B, k, M] where k is always self.k
        """
        if self.transpose_mode:
            ref = ref.transpose(1, 2).contiguous()    # [B, N, C]
            query = query.transpose(1, 2).contiguous()  # [B, M, C]
        
        batch_size = ref.shape[0]
        num_ref_points = ref.shape[1]  # N
        num_query_points = query.shape[1]  # M
        
        # Get available k (might be smaller than self.k)
        available_k = min(self.k, num_ref_points)
        
        # Compute pairwise distances
        ref_sqr = torch.sum(ref ** 2, dim=2, keepdim=True)  # [B, N, 1]
        query_sqr = torch.sum(query ** 2, dim=2, keepdim=True).transpose(1, 2)  # [B, 1, M]
        
        # Compute the dot product
        inner = torch.matmul(ref, query.transpose(1, 2))  # [B, N, M]
        
        # Compute squared distances using the formula d^2 = |x|^2 + |y|^2 - 2*x*y
        dist = ref_sqr + query_sqr - 2 * inner  # [B, N, M]
        
        # Replace any negative distances (due to numerical issues) with zero
        dist = torch.clamp(dist, min=0.0)
        
        # Transpose to match KNN_CUDA's output format
        dist = dist.transpose(1, 2)  # [B, M, N]
        
        # Get available_k smallest distances for each query point
        temp_dist, temp_idx = torch.topk(dist, k=available_k, dim=2, largest=False, sorted=True)
        
        # Create final output tensors with the exact k specified
        final_dist = torch.zeros((batch_size, num_query_points, self.k), device=dist.device)
        final_idx = torch.zeros((batch_size, num_query_points, self.k), device=dist.device, dtype=temp_idx.dtype)
        
        # Fill available neighbors
        final_dist[:, :, :available_k] = temp_dist
        final_idx[:, :, :available_k] = temp_idx
        
        # If we don't have enough neighbors, duplicate the last one
        if available_k < self.k:
            # Duplicate the last valid neighbor for remaining positions
            final_dist[:, :, available_k:] = temp_dist[:, :, -1].unsqueeze(-1).expand(-1, -1, self.k - available_k)
            final_idx[:, :, available_k:] = temp_idx[:, :, -1].unsqueeze(-1).expand(-1, -1, self.k - available_k)
        
        # Transpose back to match expected output format
        final_dist = final_dist.transpose(1, 2)  # [B, k, M]
        final_idx = final_idx.transpose(1, 2)  # [B, k, M]
        
        # Return distances and indices
        return torch.sqrt(final_dist), final_idx
    
    def __call__(self, ref, query):
        return self.forward(ref, query)

def get_knn_module(k=16, transpose_mode=False):
    """
    Get the appropriate KNN module based on availability.
    Falls back to PyTorch implementation if KNN_CUDA is not available or fails.
    
    Args:
        k: number of nearest neighbors
        transpose_mode: whether the inputs are transposed
        
    Returns:
        knn: KNN module
    """
    try:
        from knn_cuda import KNN
        knn = KNN(k=k, transpose_mode=transpose_mode)
        
        # Test if it works on a small tensor
        test_tensor = torch.rand(1, 3, 10, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        try:
            knn(test_tensor, test_tensor)
            print("Using KNN_CUDA implementation")
            return knn
        except Exception as e:
            print(f"KNN_CUDA test failed: {e}")
            print("Falling back to PyTorch KNN implementation")
            return KNNFallback(k=k, transpose_mode=transpose_mode)
            
    except ImportError:
        print("KNN_CUDA not available, using PyTorch KNN implementation")
        return KNNFallback(k=k, transpose_mode=transpose_mode)
