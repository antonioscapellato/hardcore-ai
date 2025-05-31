import torch
import math

from ._base import _HashKernel


class RandomProjKernel(_HashKernel):
    _random_projection_matrix: torch.Tensor  # type: ignore (K, N_feat_of_vector_to_hash)

    def __init__(
        self, in_features: int, out_features: int, hash_length: int, **kwargs
    ) -> None:
        super().__init__(in_features, out_features, hash_length)

        initial_proj_mat = torch.randn(hash_length, self.in_features)
        self.register_buffer("_random_projection_matrix", initial_proj_mat)

    @property
    def projection_matrix(self) -> torch.Tensor:
        return self._random_projection_matrix

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
        return torch.sign(x_proj)

    def _estimate_cosine_internal(self, codes_1: torch.Tensor, codes_2_matmuled: torch.Tensor) -> torch.Tensor:
        """
        Estimates cosine similarity based on hash codes.
        Args:
            codes_1 (torch.Tensor): Hash codes for first set of vectors (e.g., B, K).
            codes_2_matmuled (torch.Tensor): Hash codes for second set, transposed for matmul (e.g., K, M).
        Returns:
            torch.Tensor: Estimated cosine similarities (e.g., B, M).
        """
        dot = codes_1 @ codes_2_matmuled
        return (codes_1.shape[-1] - dot) // 2 # Hamming distance, est cos(theta)


    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"hash_length={self.hash_length}, type=random_projection"
        )
