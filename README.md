# Hardcore AI 02
### Hashing Towards Energy-Efficient AI Inference (Software Track)

"Hashing Towards Energy-Efficient AI Inference" hackathon! We are building software for a conceptual energy-efficient AI accelerator that leverages Locality Sensitive Hashing (LSH) to approximate dot-products in neural networks. This 24-hour challenge focuses on minimizing data movement and utilizing analog in-memory computing (AIMC) with 1-bit ADCs to reduce energy consumption and latency in AI inference.

Your mission is to implement LSH-based kernels, optimize them through training, and achieve top accuracy on the CIFAR-10 and CIFAR-100 benchmarks using ResNet20. This README provides a comprehensive guide to help you succeed.

Table of Contents

Challenge Overview
Tasks
Kernel Engineering
RandomProjKernel
LearnedProjKernel


Training Strategy


Evaluation and Optimization
Submission
Time Management (24-Hour Hackathon)
Prerequisites
References
Leaderboard and Scoring
Conclusion


Challenge Overview
The "Hashing Towards Energy-Efficient AI Inference" hackathon challenges participants to develop software that leverages Locality Sensitive Hashing (LSH) to approximate dot-products in neural networks. This approach aims to reduce energy consumption and latency in AI inference by minimizing data movement and utilizing analog in-memory computing (AIMC) with 1-bit ADCs.
Key goals:

Implement LSH-based kernels to approximate multiply-accumulate operations.
Devise data-driven methods to optimize hash functions and quantization parameters.
Evaluate your solution on benchmark deep-learning workloads (CIFAR-10 and CIFAR-100 with ResNet20).


Tasks
Kernel Engineering
You will implement two subclasses inside the bitbybit.kernel module:
RandomProjKernel

Purpose: Approximate dot-products using fixed random projections (inspired by DeepCAM).
Key Steps:
Normalize weight matrix rows and input vectors to unit length.
Generate a random projection matrix ( P ) (shape ([k, \text{input_dim}])).
Compute hash codes for weights (( h_W = \text{sign}(W @ P^T) )) and inputs (( h_x = \text{sign}(P @ x) )).
Approximate the dot-product using Hamming distance: ( s = \frac{h_W @ h_x}{k} ), then ( y = \cos\left(\frac{\pi}{2} \cdot (1 - s)\right) ).



Pseudocode:
# Normalize weights and input
W_norm = W / ||W||_2
x_norm = x / ||x||_2

# Random projection matrix P
P = randn(k, input_dim) / sqrt(input_dim)

# Hash codes
h_W = sign(W_norm @ P.T)
h_x = sign(P @ x_norm)

# Approximate dot-product
s = (h_W @ h_x) / k
y = cos(pi/2 * (1 - s))

LearnedProjKernel

Purpose: Optimize the projection matrix end-to-end for better accuracy.
Key Steps:
Initialize ( P ) as a learnable parameter.
Use a straight-through estimator (STE) to handle gradients through the non-differentiable sign() function.
Compute hash codes and approximate dot-products similarly to RandomProjKernel.



Pseudocode:
# Learnable projection matrix P
P = nn.Parameter(randn(k, input_dim) / sqrt(input_dim))

# Forward pass
h_W = sign(W_norm @ P.T)
h_x = sign(P @ x_norm)
s = (h_W @ h_x) / k
y = cos(pi/2 * (1 - s))

# Backward pass with STE: treat sign() as identity


Training Strategy

Model Loading: Use get_backbone(model_name) to load pre-trained ResNet20 models for CIFAR-10 and CIFAR-100.
Patching: Apply bb.patch_model(model) to replace standard linear layers with your hash kernels.
Fine-Tuning:
For RandomProjKernel: Fine-tune the model weights with a small learning rate (e.g., 0.001).
For LearnedProjKernel: Jointly optimize the projection matrix ( P ) and weights using STE for gradients through sign().


Hyperparameters:
Optimizer: Adam.
Learning rates: 0.001 for weights, 0.0001 for ( P ) (if applicable).
Epochs: 10-20.
Hash bits (( k )): Experiment with 32, 64, 128.



Training Script Example:
from bitbybit.utils.models import get_backbone
import bitbybit as bb

model = get_backbone("cifar10_resnet20")
hashed_model = bb.patch_model(model)
# Implement training loop with STE for LearnedProjKernel


Evaluation and Optimization

Accuracy: Evaluate your models on the CIFAR-10 and CIFAR-100 test sets.
Tuning: Adjust the number of hash bits (( k )) and training epochs to balance accuracy and computational efficiency.
Validation: Use the provided skeleton tests for local validation before submitting to the leaderboard.


Submission

Packaging: Save your trained models to submission_checkpoints/<model_name>.pth using:torch.save(hashed_model.state_dict(), OUTPUT_DIR / f"{model_name}.pth")


Submission Command:python publish.py --team-name <team-name> --key <pre-shared-key>


Frequency: You can submit multiple times during the 24-hour window. Each submission overwrites your previous score.


Time Management (24-Hour Hackathon)
Given the tight 24-hour schedule, here's a suggested timeline to maximize productivity:

Hours 1-6: Implement and test RandomProjKernel.
Hours 6-12: Develop LearnedProjKernel and integrate it into the training script.
Hours 12-18: Train models, experiment with different values of ( k ), learning rates, and epochs.
Hours 18-24: Finalize the best-performing models, validate locally, and submit multiple times to refine your leaderboard score.

Tips:

Take short breaks every 2-3 hours to maintain focus and avoid burnout.
Prioritize getting a working solution first, then optimize for accuracy.
Use leaderboard feedback to guide your final adjustments.


Prerequisites

Python 3.8+
PyTorch 1.9+
bitbybit package (provided)

Installation:
pip install torch torchvision
# Assuming bitbybit is a local package or installed separately


References

Charikar, M. (2002). Similarity Estimation Techniques from Rounding Algorithms
Nguyen, T. (2023). DeepCAM: A Fully CAM-based Inference Accelerator
Chen, Y. (2019). SLIDE: In Defense of Smart Algorithms over Hardware Acceleration
Askary, H. (2023). Intuitive Explanation of Straight-Through Estimators


Leaderboard and Scoring

Leaderboard: Live scores are updated immediately after each submission.
Scoring: The scoring mechanism is defined in bitbybit.utils.score.calculate_submission_score.
Prize: The top-ranked team at the end of the 24-hour window wins the hackathon.


Conclusion
This hackathon is an exciting opportunity to innovate at the intersection of AI efficiency and hardware-aware computing. By leveraging LSH and analog in-memory computing, you can contribute to making AI more sustainable and scalable. Good luck, and may the best team win! 🏆

