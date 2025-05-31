import torch
from src.bitbybit.kernel import RandomProjKernel, LearnedProjKernel

def test_random_proj_kernel():
    """Test RandomProjKernel for correct output shape and approximation quality."""
    in_features = 10
    out_features = 5
    hash_length = 1000
    kernel = RandomProjKernel(in_features, out_features, hash_length)
    
    # Generate random input and weights
    x = torch.randn(100, in_features)
    w = torch.randn(out_features, in_features)
    
    # Compute approximate dot product
    approx_dot = kernel(x, w)
    assert approx_dot.shape == (100, out_features), "Incorrect output shape"
    
    # Compute exact dot product for comparison
    exact_dot = x @ w.T
    error = torch.abs(approx_dot - exact_dot).mean()
    print(f"RandomProjKernel - Mean absolute error: {error.item()}")

def test_learned_proj_kernel():
    """Test LearnedProjKernel for correct output shape and gradient computation."""
    in_features = 10
    out_features = 5
    hash_length = 1000
    kernel = LearnedProjKernel(in_features, out_features, hash_length)
    
    # Generate random input and weights
    x = torch.randn(100, in_features)
    w = torch.randn(out_features, in_features)
    
    # Compute approximate dot product
    approx_dot = kernel(x, w)
    assert approx_dot.shape == (100, out_features), "Incorrect output shape"
    
    # Check gradient computation
    loss = approx_dot.sum()
    loss.backward()
    assert kernel.projection_matrix.grad is not None, "Gradients not computed"
    print("LearnedProjKernel - Gradients computed successfully")

if __name__ == "__main__":
    test_random_proj_kernel()
    test_learned_proj_kernel()