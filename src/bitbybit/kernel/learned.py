import torch
import torch.nn as nn
import math

from ._base import _HashKernel


class LearnedProjKernel(_HashKernel):

    def __init__(
        self, in_features: int, out_features: int, hash_length: int, **kwargs
    ) -> None:
        super().__init__(in_features, out_features, hash_length)

        # LSH projection matrix (learnable)
        initial_proj_mat = torch.randn(hash_length, self.in_features)
        self._learnable_projection_matrix = nn.Parameter(initial_proj_mat)

        # Temperature for sigmoid relaxation (start high, can be decayed externally)
        self.tau: float = 1.0

    @property
    def projection_matrix(self) -> torch.Tensor:
        return self._learnable_projection_matrix

    class SignSTE(torch.autograd.Function):
        """Straight-Through Estimator for the sign function to enable gradient flow."""
        @staticmethod
        def forward(ctx, input):
            # Forward pass: apply sign function to get binary codes {-1, 1}
            return torch.sign(input)

        @staticmethod
        def backward(ctx, grad_output):
            # Backward pass: pass gradients through as if sign were the identity function
            return grad_output

    def _compute_codes_internal(self, unit_vectors: torch.Tensor) -> torch.Tensor:
        # Continuous relaxation via sigmoid (Binary Concrete style without Gumbel noise)
        # Input: unit_vectors (..., in_features), projection_matrix (hash_length, in_features)
        # Output: codes in (-1, +1) by mapping sigmoid(z / tau)
        # Store the intermediate sigmoid for quantization penalty
        z = unit_vectors @ self.projection_matrix.T  # shape: (..., hash_length)
        # Compute "soft" probabilities in (0,1)
        sigmoid_out = torch.sigmoid(z / self.tau)
        # Map to (-1, +1)
        codes = 2.0 * sigmoid_out - 1.0
        # Store for later penalty computation
        self._last_sigmoid = sigmoid_out
        return codes

    def _estimate_cosine_internal(
        self, codes_1: torch.Tensor, codes_2_matmuled: torch.Tensor
    ) -> torch.Tensor:
        # Estimate cosine similarity (same as RandomProjKernel)
        # codes_1: (B, K), codes_2_matmuled: (K, N_out), K is hash_length
        # Compute hamming distance from hash codes and approximate cosine
        K = self.hash_length
        S = codes_1 @ codes_2_matmuled  # Shape: (B, N_out)
        HD = (K - S) / 2.0  # Hamming distance
        theta_approx = (math.pi / K) * HD  # Angle approximation (DeepCAM Eq. 3)
        cos_est = torch.cos(theta_approx)  # Cosine similarity (DeepCAM Eq. 4)
        return cos_est

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"hash_length={self.hash_length}, type=learned_projection (STE TODO)"
        )