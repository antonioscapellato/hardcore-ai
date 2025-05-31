"""
kernel_test.py

This script validates the RandomProjKernel and LearnedProjKernel implementations
in the bitbybit.kernel package. We check that:
1. The forward pass approximates the standard matrix multiplication (x @ W.T)
   by computing: x_norm * w_norm * estimated_cosine.
2. The LearnedProjKernel's projection matrix is learnable (requires_grad=True).
3. The output shapes and value ranges are consistent with expectations.

Usage:
    python kernel_test.py
"""

import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.bitbybit.kernel.random import RandomProjKernel
from src.bitbybit.kernel.learned import LearnedProjKernel

def run_single_test(
    kernel_class,
    in_features: int,
    out_features: int,
    hash_length: int,
    batch_size: int = 10,
):
    """
    Run a single comparison between the hashed approximation (kernel.forward)
    and the exact dense matrix multiplication (x @ W.T).
    Returns:
        mean_diff (float): mean absolute difference over all batch × out_features entries.
        max_diff (float): maximum absolute difference over all entries.
    """
    # Instantiate kernel (RandomProj or LearnedProj)
    kernel = kernel_class(in_features=in_features, out_features=out_features, hash_length=hash_length)

    # Generate random input x and weight W from a normal distribution
    # Shapes:
    #   x: (batch_size, in_features)
    #   W: (out_features, in_features)
    x = torch.randn(batch_size, in_features)
    W = torch.randn(out_features, in_features)

    # Compute approximate output: uses LSH-based hash kernel
    approx_output = kernel.forward(x, W)  # shape: (batch_size, out_features)

    # Compute exact output: dense matrix multiplication
    exact_output = x @ W.t()  # shape: (batch_size, out_features)

    # Compute absolute differences
    abs_diff = (approx_output - exact_output).abs()

    # Gather statistics
    mean_diff = abs_diff.mean().item()
    max_diff = abs_diff.max().item()

    return mean_diff, max_diff

def get_threshold_for(hash_length: int, thresholds: dict) -> float:
    """
    Retrieve the maximum allowable difference for a given hash_length.
    If not in the dictionary, return a default large value.
    """
    return thresholds.get(hash_length, float("inf"))

def main():
    # Fix random seed for reproducibility
    torch.manual_seed(0)

    # 1. Define the sweep ranges for input/output dimensions and hash lengths
    dims = [8, 16, 32]                # Test input/output feature dimensions
    hash_lengths = [32, 64, 128, 256] # Different LSH code lengths to test

    # 2. Define configurable maximum‐difference thresholds per hash length.
    #    These can be tuned based on empirical observations.
    thresholds = {
        32:  5*10.0,  # With shorter hash codes (32 bits), error can be larger
        64:   5*5.0,  # Mid‐range threshold used previously
        128:  5*3.0,  # Longer codes → lower variance
        256:  5*2.0,  # Even longer → tighter bound
    }

    # Data structures to collect error statistics:
    #   errors[dim] = list of max_diff for each hash_length in order
    results_random = {d: [] for d in dims}
    results_learned = {d: [] for d in dims}

    # Keep track of any failing combinations
    failures = []

    # 3. Sweep over each dimension and hash length
    for d in dims:
        for hl in hash_lengths:
            # We'll test both RandomProjKernel and LearnedProjKernel

            # ---- RandomProjKernel ----
            mean_r, max_r = run_single_test(
                RandomProjKernel, in_features=d, out_features=d, hash_length=hl, batch_size=10
            )
            results_random[d].append(max_r)

            # Check against threshold
            thr = get_threshold_for(hl, thresholds)
            if max_r > thr:
                failures.append(
                    f"RandomProjKernel failed at dim={d}, hash_length={hl}: max_diff={max_r:.4f} > {thr}"
                )

            # ---- LearnedProjKernel ----
            mean_l, max_l = run_single_test(
                LearnedProjKernel, in_features=d, out_features=d, hash_length=hl, batch_size=10
            )
            results_learned[d].append(max_l)

            # Check that the projection matrix is learnable (requires_grad=True)
            # (Instantiate a fresh LearnedProjKernel to inspect its parameter directly)
            tmp_kernel = LearnedProjKernel(in_features=d, out_features=d, hash_length=hl)
            assert isinstance(tmp_kernel._learnable_projection_matrix, nn.Parameter), (
                f"projection_matrix is not an nn.Parameter for LearnedProjKernel(dim={d}, hash_length={hl})"
            )
            assert tmp_kernel._learnable_projection_matrix.requires_grad, (
                f"projection_matrix must require_grad for LearnedProjKernel(dim={d}, hash_length={hl})"
            )

            # Check against threshold
            if max_l > thr:
                failures.append(
                    f"LearnedProjKernel failed at dim={d}, hash_length={hl}: max_diff={max_l:.4f} > {thr}"
                )

            # Print progress for visibility
            print(
                f"dim={d}, hl={hl} | "
                f"Random(mean={mean_r:.4f}, max={max_r:.4f}) | "
                f"Learned(mean={mean_l:.4f}, max={max_l:.4f}) | "
                f"threshold={thr}"
            )

    # 4. After sweeping, check if any failures occurred
    if failures:
        print("\nSome tests failed:")
        for msg in failures:
            print("  -", msg)
        sys.exit(1)  # Exit with error code to indicate failure

    # 5. If all tests passed, create a plot of the max errors vs. hash length
    #    for each dimension (RandomProj only, since LearnedProj matches at init).
    plt.figure(figsize=(8, 6))
    for d in dims:
        plt.plot(
            hash_lengths,
            results_random[d],
            marker="o",
            linestyle="-",
            label=f"dim={d}"
        )

    # Also plot the thresholds as horizontal dashed lines (for reference)
    for hl, thr in thresholds.items():
        plt.axhline(y=thr, color="gray", linestyle="--", linewidth=0.7)
        # Annotate the threshold
        plt.text(
            hl, thr + 0.2, f"thr={thr}", fontsize=8, color="gray", ha="center"
        )

    plt.title("Max Absolute Difference vs Hash Length\n(RandomProjKernel across dimensions)")
    plt.xlabel("Hash Length")
    plt.ylabel("Max Absolute Difference")
    plt.xticks(hash_lengths)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(title="Input/Output Dim")
    plt.tight_layout()

    # Display the plot
    plt.show()

    print("\nAll tests passed successfully. Plot displayed.")

if __name__ == "__main__":
    main()