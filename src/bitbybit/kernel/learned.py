import torch
import torch.nn as nn
import math

from ._base import _HashKernel


class LearnedProjKernel(_HashKernel):
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
        
    def __init__(
        self, in_features: int, 
        out_features: int, 
        hash_length: int, 
        query_bits: int = 128,
        k_top: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(in_features, out_features, hash_length)
        self.query_bits = min(query_bits, hash_length)
        self.k_top = k_top
        
        self.beta = kwargs.get('initial_beta', 1.0)

        # LSH projection matrix (learnable)
        # Initialize with Xavier/Glorot uniform for better training stability
        initial_proj_mat = torch.randn(hash_length, self.in_features)
        nn.init.xavier_uniform_(initial_proj_mat)
        self._learnable_projection_matrix = nn.Parameter(initial_proj_mat)
        
        # Pre-compute the weight codes once (no grad needed)
        with torch.no_grad():
            eye_w = torch.eye(out_features, in_features)
            full_codes = self._compute_codes_internal(eye_w)          # (N_out, K)
            self.register_buffer("_codes_full", full_codes)           # full  ±1
            self.register_buffer("_codes_query",                      # prefix used to route
                                 full_codes[:, : self.query_bits])

    @property
    def projection_matrix(self) -> torch.Tensor:
        return self._learnable_projection_matrix

    def _compute_codes_internal(self, unit_vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute binary hash codes using learnable projection matrix.
        
        Args:
            unit_vectors: (..., in_features) - normalized input vectors
            
        Returns:
            codes: (..., hash_length) with values {-1, 1}
        """
        # Use STE to make the sign operation differentiable for optimization
        projection = unit_vectors @ self.projection_matrix.T
        codes = self.SignSTE.apply(projection)
        return codes

    def _estimate_cosine_internal(
        self, codes_1: torch.Tensor, codes_2_matmuled: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate cosine similarity from binary hash codes.
        
        Args:
            codes_1: (B, K) - query hash codes
            codes_2_matmuled: (K, N_out) - transposed key hash codes
            
        Returns:
            cos_est: (B, N_out) - estimated cosine similarities
        """
        K = self.hash_length
        S = codes_1 @ codes_2_matmuled  # Shape: (B, N_out)
        HD = (K - S) / 2.0  # Hamming distance from dot product of {-1,1} codes
        theta_approx = (math.pi / K) * HD  # Angle approximation (DeepCAM Eq. 3)
        cos_est = torch.cos(theta_approx)  # Cosine similarity (DeepCAM Eq. 4)
        return cos_est

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"hash_length={self.hash_length}, query_bits={self.query_bits}, "
            f"k_top={self.k_top}, type=learned_projection_STE"
        )
            
    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Hash-based Top-k routed matrix multiply with proper handling of k_layer <= out_features.

        Args:
            x: (B, in_features) - input tensor
            weights: (N_out, in_features) - weight matrix

        Returns:
            y: (B, N_out) - output tensor
        """
        B, in_features = x.shape
        N_out, _ = weights.shape
        
        # Use appropriate epsilon for numerical stability
        eps = torch.finfo(x.dtype).eps if x.dtype.is_floating_point else 1e-7

        # 0. Compute L2 norms and unit vectors
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)    # (B, 1)
        x_unit = x / x_norm                                     # (B, in_features)

        w_norm = weights.norm(dim=-1, keepdim=True).clamp_min(eps)  # (N_out, 1)
        w_unit = weights / w_norm                                   # (N_out, in_features)

        # 1. Full K-bit hash codes for x and W
        code_x_full = self._compute_codes_internal(x_unit)     # (B, K)
        code_w_full = self._compute_codes_internal(w_unit)     # (N_out, K)

        # 2. Extract first q bits for prefix routing
        q = self.query_bits
        code_x_pref = code_x_full[:, :q]                       # (B, q)
        code_w_pref = code_w_full[:, :q]                       # (N_out, q)

        # 3. Compute prefix similarity: (B, q) @ (q, N_out) → (B, N_out)
        sim_pref = torch.einsum("bq,nq->bn", code_x_pref, code_w_pref)

        # 4. Determine actual k for this layer (cannot exceed N_out)
        k_layer = min(self.k_top, N_out)

        # 5. Top-k on the prefix similarity
        #    top_idx: (B, k_layer).  If N_out < self.k_top, this picks N_out.
        top_val, top_idx = torch.topk(sim_pref, k_layer, dim=-1)

        # 6. Prepare final output y = zeros((B, N_out))
        y = x.new_zeros((B, N_out))

        # 7. For each batch row b, gather k_layer candidates and compute full-K cosine
        K = self.hash_length
        for b in range(B):
            idx_b = top_idx[b]                             # (k_layer,)

            # 7a. Gather the k_layer full-K codes of W for these indices
            #     code_w_full is (N_out, K).  Selecting idx_b → (k_layer, K)
            codes_w_k = code_w_full[idx_b, :]              # (k_layer, K)
            codes_w_k_t = codes_w_k.transpose(0, 1)        # (K, k_layer)

            # 7b. Get x's full-K code for this sample
            code_x_b = code_x_full[b].unsqueeze(0)         # (1, K)

            # 7c. Use existing _estimate_cosine_internal to get (1, k_layer)
            #     That internally does: S = code_x_b @ codes_w_k_t → (1, k_layer), etc.
            cos_line = self._estimate_cosine_internal(code_x_b, codes_w_k_t)  # (1, k_layer)
            cos_line = cos_line.squeeze(0)                # (k_layer,)

            # 7d. Scale by norms: y_cand[i] = ||x[b]|| * ||W[idx_b[i]]|| * cos_line[i]
            x_n = x_norm[b].item()                        # scalar
            w_n_line = w_norm[idx_b].squeeze(-1)          # (k_layer,)
            y_cand = (x_n * w_n_line) * cos_line          # (k_layer,)

            # 7e. Scatter these k_layer values back into y[b]
            #     y[b, idx_b[i]] = y_cand[i]
            y[b].scatter_(dim=0, index=idx_b, src=y_cand)

        return y