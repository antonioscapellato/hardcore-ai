import torch
import matplotlib.pyplot as plt
import math
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


def get_threshold(hash_length, dim):
    """
    Define an analytical threshold based on hash_length and input dimension.
    Based on the fact that the variance of the random projection approximation
    for dot products scales as O(dim / hash_length), we set the threshold for
    mean absolute error to dim / sqrt(hash_length).
    """
    return 2*float(dim) / math.sqrt(hash_length)


def test_kernel_sweep(): 
    """
    Sweep across a range of input dimensions and hash_length values,
    compute errors, and plot them if all tests pass.
    """
    # List of input/output dimensions to test
    dims_list = [8, 16, 32]
    # Hash lengths: powers of 2 from 32 to 4096
    hash_length_list = [2 ** i for i in range(5, 13)]

    num_train_steps = 100
    learning_rate = 0.01

    plt.figure(figsize=(10, 8))

    for dim in dims_list:
        in_features = dim
        out_features = dim

        # Generate consistent training and test data
        x_train = torch.randn(1000, in_features)
        x_test = torch.randn(100, in_features)
        w = torch.randn(out_features, in_features)
        y_train = x_train @ w.T
        y_test = x_test @ w.T

        random_errors = []
        learned_errors = []

        # Test RandomProjKernel across hash lengths
        for hash_length in hash_length_list:
            kernel = RandomProjKernel(in_features, out_features, hash_length)
            approx_dot = kernel(x_test, w)
            error = torch.abs(approx_dot - y_test).mean().item()
            threshold = get_threshold(hash_length, in_features)
            assert error < threshold, (
                f"RandomProjKernel error {error} exceeds threshold {threshold} "
                f"for dim {dim}, hash_length {hash_length}"
            )
            random_errors.append(error)
            print(f"RandomProjKernel - dim={dim}, hash_length={hash_length}, error={error}")

        # Test LearnedProjKernel across hash lengths with training
        for hash_length in hash_length_list:
            kernel = LearnedProjKernel(in_features, out_features, hash_length)
            optimizer = torch.optim.SGD([kernel.projection_matrix], lr=learning_rate)
            # Train the kernel to approximate the exact dot product
            for step in range(num_train_steps):
                optimizer.zero_grad()
                approx_dot_train = kernel(x_train, w)
                loss = ((approx_dot_train - y_train) ** 2).mean()
                loss.backward()
                optimizer.step()
            # Evaluate on test data
            approx_dot_test = kernel(x_test, w)
            error = torch.abs(approx_dot_test - y_test).mean().item()
            threshold = get_threshold(hash_length, in_features)
            assert error < threshold, (
                f"LearnedProjKernel error {error} exceeds threshold {threshold} "
                f"for dim {dim}, hash_length {hash_length}"
            )
            learned_errors.append(error)
            print(f"LearnedProjKernel - dim={dim}, hash_length={hash_length}, error={error}")

        # Plot errors for this dimension
        plt.plot(hash_length_list, random_errors, label=f'Random d={dim}', marker='o')
        plt.plot(hash_length_list, learned_errors, label=f'Learned d={dim}', marker='s')

    plt.xlabel('Hash Length')
    plt.ylabel('Mean Absolute Error')
    plt.title('Approximation Error vs. Hash Length for Various Dimensions')
    plt.xscale('log', base=2)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    test_random_proj_kernel()
    test_learned_proj_kernel()
    test_kernel_sweep()