import torch
import math

from ._base import _HashKernel


class RandomProjKernel(_HashKernel):

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
        Computes binary hash codes for unit input vectors using fixed random projections.
        Args:
            unit_vectors (torch.Tensor): Input tensor of shape (..., in_features),
                                         assumed to be L2-normalized already.
        Returns:
            torch.Tensor: Binary codes in {-1, 1} of shape (..., hash_length).
        """
        # Project each unit vector onto the random projection matrix.
        # projection_matrix has shape (hash_length, in_features).
        # unit_vectors has shape (..., in_features). We perform:
        #    projections = unit_vectors @ projection_matrix.T
        # which yields shape (..., hash_length).
        projections = unit_vectors @ self.projection_matrix.t()  # shape: (..., hash_length)

        # Convert projections to binary codes in {-1, +1} by taking the sign.
        # torch.where(projections >= 0, +1.0, -1.0) yields a tensor of the same shape.
        codes = torch.where(
            projections >= 0,
            torch.tensor(1.0, device=projections.device, dtype=projections.dtype),
            torch.tensor(-1.0, device=projections.device, dtype=projections.dtype),
        )
        return codes

    def _estimate_cosine_internal(
        self, codes_1: torch.Tensor, codes_2_matmuled: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimates cosine similarity based on hash codes.
        Args:
            codes_1 (torch.Tensor): Hash codes for first set of vectors, shape (B, K).
            codes_2_matmuled (torch.Tensor): Hash codes for second set (transposed),
                                             shape (K, M).
        Returns:
            torch.Tensor: Estimated cosine similarities, shape (B, M), values in [-1, 1].
        """
        # Compute the dot product between codes_1 and codes_2_matmuled:
        #   dot = (B, K) @ (K, M) --> (B, M). Since codes are ±1,
        #   dot ∈ [-K, +K]. We then normalize by hash_length (K).
        dot = torch.matmul(codes_1, codes_2_matmuled)  # shape: (B, M)

        # Normalize by hash_length to approximate cosine similarity in [-1, +1].
        return dot / float(self.hash_length)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"hash_length={self.hash_length}, type=random_projection"
        )