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
        # Compute binary hash codes using fixed random projections
        # Input: unit_vectors (..., in_features), projection_matrix (hash_length, in_features)
        # Operation: sign(unit_vectors @ projection_matrix.T) as per DeepCAM Page 2
        # Output shape: (..., hash_length) with values {-1, 1}
        projection = unit_vectors @ self.projection_matrix.T
        codes = torch.sign(projection)
        return codes

    def _estimate_cosine_internal(
        self, codes_1: torch.Tensor, codes_2_matmuled: torch.Tensor
    ) -> torch.Tensor:
        # Estimate cosine similarity between two sets of hash codes
        # codes_1: (B, K), codes_2_matmuled: (K, N_out), where K is hash_length
        # Step 1: Compute sum of products (S) efficiently via matrix multiplication
        # Step 2: Calculate hamming distance (HD) as (K - S) / 2, since codes are {-1, 1}
        # Step 3: Approximate angle theta = (pi / K) * HD (DeepCAM Eq. 3)
        # Step 4: Return cos(theta) for geometric dot-product (DeepCAM Eq. 4)
        K = self.hash_length
        S = codes_1 @ codes_2_matmuled  # Shape: (B, N_out), S = sum of a_i * b_i
        HD = (K - S) / 2.0  # Hamming distance, float for precision
        theta_approx = (math.pi / K) * HD  # Angle approximation
        cos_est = torch.cos(theta_approx)  # Cosine similarity
        return cos_est

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"hash_length={self.hash_length}, type=random_projection"
        )