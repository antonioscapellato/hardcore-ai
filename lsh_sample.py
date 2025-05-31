import numpy as np

# --------------------------------------------------------
# Step 1: Generate random input vectors (like neural inputs and weights)
# --------------------------------------------------------

dim = 128  # Dimensionality of the vectors (e.g., features in a layer)
x = np.random.randn(dim)  # Input vector
w = np.random.randn(dim)  # Weight vector

# --------------------------------------------------------
# Step 2: Create random hyperplanes for LSH hashing
# --------------------------------------------------------

# LSH for cosine similarity uses random hyperplanes to split the space
# Each hyperplane is a random vector. A vector's sign with respect to this
# hyperplane determines a bit in the hash code.
num_bits = 32  # Length of the hash (more bits = better accuracy, more compute)
random_hyperplanes = np.random.randn(num_bits, dim)

# --------------------------------------------------------
# Step 3: Define LSH hash function using random projections
# --------------------------------------------------------

def lsh_hash(vec, hyperplanes):
    """
    Projects vector onto random hyperplanes and outputs a binary hash code.
    Each bit in the hash is 1 if the dot product with the hyperplane >= 0,
    and 0 otherwise.
    
    This approximates angular distance or cosine similarity.
    """
    return (np.dot(hyperplanes, vec) >= 0).astype(int)  # Binary vector

# Hash both input vector and weight vector
h_x = lsh_hash(x, random_hyperplanes)
h_w = lsh_hash(w, random_hyperplanes)

# --------------------------------------------------------
# Step 4: Compute Hamming similarity between hashes
# --------------------------------------------------------

def hamming_similarity(h1, h2):
    """
    Returns the normalized Hamming similarity, i.e., proportion of matching bits.
    This approximates cosine similarity in the original space.
    """
    return np.sum(h1 == h2) / len(h1)

# Compute approximate similarity using LSH hashes
approx_sim = hamming_similarity(h_x, h_w)

# Compute true cosine similarity for comparison
true_sim = np.dot(x, w) / (np.linalg.norm(x) * np.linalg.norm(w))

# --------------------------------------------------------
# Output Results
# --------------------------------------------------------

print(f"Approximate similarity (LSH): {approx_sim:.3f}")
print(f"True cosine similarity:       {true_sim:.3f}")

# --------------------------------------------------------
# 🔍 Theory Summary:
# --------------------------------------------------------

# - LSH replaces exact dot products with fast bitwise operations.
# - The hash codes encode vector directions using signs of dot products.
# - Similar vectors (with small angles between them) will hash to similar codes.
# - This allows approximate, energy-efficient similarity computation.
# - Instead of computing full dot products (128 multiplications + adds),
#   we only compare binary strings of length 32 here (very fast, low-power).
#
# This forms the basis of LSH-based approximate inference.
