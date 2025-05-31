"""
Random Projection Kernel implementation for Locality-Sensitive Hashing (LSH).
This module implements a random projection-based LSH kernel that maps high-dimensional
vectors to binary hash codes using random projections.
"""

import torch
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from ._base import _HashKernel

class RandomProjKernel(_HashKernel):
    """
    A kernel that uses random projections for Locality-Sensitive Hashing.
    
    This implementation uses random Gaussian projections to map input vectors to binary
    hash codes. The projection matrix is randomly initialized and remains fixed during
    the lifetime of the kernel (unlike LearnedProjKernel which learns the projections).
    
    The kernel estimates cosine similarity between vectors based on the Hamming distance
    of their hash codes, following the principle that similar vectors in the original
    space will have similar hash codes.
    """
    
    def __init__(
        self, in_features: int, out_features: int, hash_length: int, **kwargs
    ) -> None:
        """
        Initialize the RandomProjKernel.
        
        Args:
            in_features (int): Dimension of input vectors
            out_features (int): Number of output classes/features
            hash_length (int): Length of the binary hash codes
            **kwargs: Additional keyword arguments
        """
        logger.info(f"Initializing RandomProjKernel with in_features={in_features}, "
                   f"out_features={out_features}, hash_length={hash_length}")
        super().__init__(in_features, out_features, hash_length)
        
        # Initialize random projection matrix with standard normal distribution
        initial_proj_mat = torch.randn(hash_length, self.in_features)
        logger.debug(f"Generated random projection matrix with shape: {initial_proj_mat.shape}")
        
        # Register as buffer since this matrix is not learnable
        self.register_buffer("_random_projection_matrix", initial_proj_mat)
        logger.info("Registered random projection matrix as non-learnable buffer")

    @property
    def projection_matrix(self) -> torch.Tensor:
        """Get the random projection matrix."""
        return self._random_projection_matrix

    def _compute_codes_internal(self, unit_vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute binary hash codes for input unit vectors using random projection.
        
        The method projects input vectors using a random matrix and then binarizes
        the result using the sign function. This creates a binary code where each bit
        represents whether the projection was positive or negative.
        
        Args:
            unit_vectors (torch.Tensor): Input tensor of shape (batch_size, in_features)
                                       containing unit vectors.
        
        Returns:
            torch.Tensor: Binary hash codes of shape (batch_size, hash_length)
                         with values in {0, 1}.
        """
        logger.debug(f"Computing hash codes for input shape: {unit_vectors.shape}")
        
        # Project vectors using random matrix
        projected = torch.matmul(unit_vectors, self.projection_matrix.t())
        logger.debug(f"Projected vectors shape: {projected.shape}")
        
        # Convert to binary codes using sign function
        codes = (projected > 0).float()
        logger.debug(f"Generated binary codes shape: {codes.shape}")
        
        return codes

    def _estimate_cosine_internal(
        self, codes_1: torch.Tensor, codes_2_matmuled: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate the cosine similarity between vectors based on their hash codes.
        
        This method uses the Hamming distance between hash codes to estimate the
        angle between the original vectors, and then computes the cosine of that angle.
        The estimation follows the principle that similar vectors will have similar
        hash codes, resulting in smaller Hamming distances.
        
        Args:
            codes_1 (torch.Tensor): Binary hash codes of shape (batch_size, hash_length).
            codes_2_matmuled (torch.Tensor): Binary hash codes of shape 
                                            (batch_size, out_features, hash_length).
        
        Returns:
            torch.Tensor: Estimated cosine values of shape (batch_size, out_features).
        """
        logger.debug(f"Estimating cosine similarity between codes shapes: "
                    f"{codes_1.shape} and {codes_2_matmuled.shape}")
        
        # Calculate Hamming distance between codes
        hamming_dist = (codes_1.unsqueeze(1) != codes_2_matmuled).float().sum(dim=2)
        logger.debug(f"Computed hamming distances shape: {hamming_dist.shape}")
        
        # Convert Hamming distance to angle
        theta = (math.pi / self.hash_length) * hamming_dist
        logger.debug(f"Computed angles shape: {theta.shape}")
        
        # Convert angle to cosine similarity
        cosine_theta = torch.cos(theta)
        logger.debug(f"Final cosine estimates shape: {cosine_theta.shape}")
        
        return cosine_theta

    def extra_repr(self) -> str:
        """Return a string representation of the kernel's configuration."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"hash_length={self.hash_length}, type=random_projection"
        )