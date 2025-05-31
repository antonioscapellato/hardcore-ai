import torch
import torch.nn as nn
import math

from ._base import _HashKernel

class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        # e.g. hard-tanh surrogate
        grad = grad_out * (x.abs() <= 1).to(grad_out.dtype)
        return grad

class LearnedProjKernel(_HashKernel):

    def __init__(
        self, in_features: int, out_features: int, hash_length: int, **kwargs
    ) -> None:
        super().__init__(in_features, out_features, hash_length)

        # LSH projection matrix (learnable)
        initial_proj_mat = torch.randn(hash_length, self.in_features)
        self._learnable_projection_matrix = nn.Parameter(initial_proj_mat)

    @property
    def projection_matrix(self) -> torch.Tensor:
        return self._learnable_projection_matrix

    def _compute_codes_internal(self, unit_vectors: torch.Tensor) -> torch.Tensor:
        """
        Computes binary hash codes for unit input vectors using self.projection_matrix.
        Args:
            vectors (torch.Tensor): Input tensor of shape (..., features_dim).
                                            features_dim must match self.projection_matrix's second dim.
        Returns:
            torch.Tensor: Binary codes {-1, 1} of shape (..., self.hash_length).
        """
        x_proj = unit_vectors @ self.projection_matrix.T # (B, N_in) @ (N_in, K) -> (B, K)
        return SignSTE.apply(x_proj) # type: ignore

    def _estimate_cosine_internal(
        self, codes_1: torch.Tensor, codes_2_matmuled: torch.Tensor
    ) -> torch.Tensor:
        K = self.hash_length
        dot = codes_1 @ codes_2_matmuled
        return (K - dot) / 2 # Hamming distance, est cos(theta)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"hash_length={self.hash_length}, type=learned_projection (STE TODO)"
        )
