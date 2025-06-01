import torch
import torch.nn as nn
import math

from ._base import _HashKernel


class LearnedProjKernel(_HashKernel):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hash_length: int,
        initial_beta: float = 1.0,
        **kwargs
    ) -> None:
        super().__init__(in_features, out_features, hash_length)
        self.beta = initial_beta

        # LSH projection matrix (learnable)
        initial_proj_mat = torch.randn(hash_length, self.in_features)
        self._learnable_projection_matrix = nn.Parameter(initial_proj_mat)

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
        # Compute hash codes using learnable projection matrix
        # During training, use tanh with temperature beta for differentiability
        # During evaluation, use sign to get binary codes
        # Input: unit_vectors (..., in_features), projection_matrix (hash_length, in_features)
        # Output shape: (..., hash_length)
        projection = unit_vectors @ self.projection_matrix.T
        if self.training:
            codes = torch.tanh(self.beta * projection)
        else:
            codes = torch.sign(projection)
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