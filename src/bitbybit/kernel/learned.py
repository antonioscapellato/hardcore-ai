"""
Learned Projection Kernel implementation for Locality-Sensitive Hashing (LSH).
This module implements a learned projection-based LSH kernel that maps high-dimensional
vectors to binary hash codes using trainable projections.
"""

import torch
import torch.nn as nn
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from ._base import _HashKernel

class LearnedProjKernel(_HashKernel):
    """
    A kernel that uses learned projections for Locality-Sensitive Hashing.
    
    This implementation uses trainable projections to map input vectors to binary
    hash codes. The projection matrix is initialized randomly and can be optimized
    during training to better preserve similarity relationships.
    """
    
    def __init__(
        self, in_features: int, out_features: int, hash_length: int, **kwargs
    ) -> None:
        """
        Initialize the LearnedProjKernel.
        
        Args:
            in_features (int): Dimension of input vectors
            out_features (int): Number of output classes/features
            hash_length (int): Length of the binary hash codes
            **kwargs: Additional keyword arguments
        """
        logger.info(f"Initializing LearnedProjKernel with in_features={in_features}, "
                   f"out_features={out_features}, hash_length={hash_length}")
        super().__init__(in_features, out_features, hash_length)
        
        # Initialize learnable projection matrix
        initial_proj_mat = torch.randn(hash_length, self.in_features)
        self._learnable_projection_matrix = nn.Parameter(initial_proj_mat)
        logger.debug(f"Initialized projection matrix with shape: {initial_proj_mat.shape}")

    @property
    def projection_matrix(self) -> torch.Tensor:
        """Get the learnable projection matrix."""
        return self._learnable_projection_matrix

    def _compute_codes_internal(self, unit_vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute binary hash codes for input unit vectors using learned projection.
        
        Args:
            unit_vectors (torch.Tensor): Input tensor of shape (batch_size, in_features)
                                       containing unit vectors.
        
        Returns:
            torch.Tensor: Binary hash codes of shape (batch_size, hash_length)
                         with values in {0, 1}.
        """
        logger.debug(f"Computing hash codes for input shape: {unit_vectors.shape}")
        
        # Perform projection with learnable matrix
        projected = torch.matmul(unit_vectors, self.projection_matrix.t())
        logger.debug(f"Projected vectors shape: {projected.shape}")
        
        # Apply sign function to get binary codes
        codes = (projected > 0).float()
        logger.debug(f"Generated binary codes shape: {codes.shape}")
        
        return codes

    def _estimate_cosine_internal(
        self, codes_1: torch.Tensor, codes_2_matmuled: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate the cosine similarity between vectors based on their hash codes.
        
        Args:
            codes_1 (torch.Tensor): Binary hash codes of shape (batch_size, hash_length).
            codes_2_matmuled (torch.Tensor): Binary hash codes of shape 
                                            (batch_size, out_features, hash_length).
        
        Returns:
            torch.Tensor: Estimated cosine values of shape (batch_size, out_features).
        """
        logger.debug(f"Estimating cosine similarity between codes shapes: "
                    f"{codes_1.shape} and {codes_2_matmuled.shape}")
        
        # Compute hamming distance
        hamming_dist = (codes_1.unsqueeze(1) != codes_2_matmuled).float().sum(dim=2)
        logger.debug(f"Computed hamming distances shape: {hamming_dist.shape}")
        
        # Convert to angle and then to cosine
        theta = (math.pi / self.hash_length) * hamming_dist
        cosine_theta = torch.cos(theta)
        logger.debug(f"Final cosine estimates shape: {cosine_theta.shape}")
        
        return cosine_theta

    def extra_repr(self) -> str:
        """Return a string representation of the kernel's configuration."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"hash_length={self.hash_length}, type=learned_projection"
        )