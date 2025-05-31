# Hardcore AI 02
### Hashing Towards Energy-Efficient AI Inference (Software Track)

### Energy-Efficient AI Inference with LSH

This project focuses on developing an energy-efficient AI inference system by replacing traditional dot-products in deep neural networks with approximate geometric dot-products using **Locality Sensitive Hashing (LSH)**. The approach leverages **analog in-memory computing (AIMC)** to minimize data movement and reduce energy consumption, while maintaining high accuracy through LSH-based approximations.

---

## Motivation

Deep neural networks (DNNs), particularly large language models (LLMs), are growing rapidly in size and complexity. However, the primary cost in deployed systems is no longer arithmetic throughput but the **energy and latency** associated with transferring parameters and activations between processing elements and external memory. Traditional digital approaches are becoming unsustainable for future models.

**Analog in-memory computing (AIMC)** addresses this by performing multiply-accumulate (MAC) operations directly within memory arrays, significantly reducing data movement and improving energy efficiency. However, AIMC requires quantization of analog dot-products, typically using power-hungry analog-to-digital converters (ADCs). This project removes high-resolution ADCs from the critical path by using **1-bit ADCs** and reconstructing outputs via **Hamming distance calculations** against pre-encoded weight patterns, a technique based on LSH.

---

## Objectives

The main goals of this project are:
- Design and implement **LSH-based kernels** (`RandomProjKernel` and `LearnedProjKernel`) to approximate dot-products in neural networks.
- Develop data-driven methods to **learn or adapt hash functions** and optimize for accuracy vs. efficiency trade-offs.
- Evaluate the approach on benchmark deep-learning workloads (CIFAR-10 and CIFAR-100 with ResNet20) and compete on a public leaderboard.

---

## Implementation Details

### RandomProjKernel
The `RandomProjKernel` uses fixed random projections to hash weights and inputs, approximating dot-products via Hamming distance calculations.

- **Normalization**: Both the weight matrix rows \( W \) and the input vector \( x \) are normalized to unit length.
- **Projection Matrix**: A random matrix \( P \) (shape \([k, \text{input_dim}]\)) is generated from a standard normal distribution.
- **Hash Codes**:
  - Weights: \( h_W = \text{sign}(W @ P^T) \) (pre-computed).
  - Inputs: \( h_x = \text{sign}(P @ x) \) (computed during inference).
- **Dot-Product Approximation**:
  - Compute \( s = \frac{h_W @ h_x}{k} \).
  - Approximate \( W @ x \) as \( y = \cos\left(\frac{\pi}{2} \cdot (1 - s)\right) \).

### LearnedProjKernel
The `LearnedProjKernel` optimizes the projection matrix end-to-end, improving accuracy over random projections.

- **Learnable Parameter**: The projection matrix \( P \) is initialized randomly and trained as a model parameter.
- **Hash Codes**: Similar to `RandomProjKernel`, using \( h_W = \text{sign}(W @ P^T) \) and \( h_x = \text{sign}(P @ x) \).
- **Straight-Through Estimator (STE)**: Used to handle the non-differentiable `sign()` function by passing gradients through as if it were the identity function.
- **Dot-Product Approximation**: Same as in `RandomProjKernel`.

### bitbybit Package Usage
The `bitbybit` package provides core abstractions and utilities:
- **`HashKernel`**: Interface for handling unit-vector normalization and hash-based similarity.
- **`RandomProjKernel`** and **`LearnedProjKernel`**: Implementations of the LSH-based kernels.
- **`patch_model(model)`**: Utility to replace standard linear layers with hash-based kernels.

---

## Training Instructions

1. **Load Pre-trained Models**:
   - Use `get_backbone(model_name)` to load pre-trained ResNet20 models for CIFAR-10 and CIFAR-100.
   
2. **Patch Models**:
   - Use `bb.patch_model(model)` to replace standard linear layers with the implemented hash kernels.

3. **Fine-Tuning**:
   - For `RandomProjKernel`, fine-tune the weights \( W \) with a small learning rate (e.g., 0.001).
   - For `LearnedProjKernel`, jointly optimize \( P \) and \( W \), using STE for gradients through the `sign()` function.

4. **Training Hyperparameters**:
   - Use Adam optimizer with learning rates of 0.001 for weights and 0.0001 for the projection matrix (if applicable).
   - Train for 10-20 epochs, monitoring validation accuracy.
   - Experiment with different values of \( k \) (e.g., 32, 64, 128) to balance accuracy and efficiency.

5. **Save Checkpoints**:
   - Save the trained models to `submission_checkpoints/<model_name>.pth` using:
     ```python
     torch.save(hashed_model.state_dict(), OUTPUT_DIR / f"{model_name}.pth")
     ```

---

## Benchmark Datasets and Models

The project uses the following benchmarks:
- **Datasets**: CIFAR-10 and CIFAR-100.
- **Models**: Pre-trained ResNet20 models for each dataset.

---

## Evaluation

- **Accuracy**: The primary evaluation metric is the test set accuracy on CIFAR-10 and CIFAR-100.
- **Local Validation**: Use the provided skeleton tests to validate your kernels locally before submission.
- **Hyperparameter Tuning**: Adjust \( k \) and training epochs to optimize the accuracy-efficiency trade-off.

---

## Submission Process

1. **Save Trained Models**:
   - Ensure your trained models are saved in `submission_checkpoints/<model_name>.pth`.

2. **Submit Using `publish.py`**:
   - Use the following command to submit your models to the evaluation server:
     ```bash
     python publish.py --team-name <team-name> --key <pre-shared-key>
     ```
   - You can submit multiple times during the 24-hour window; each submission overwrites the previous best score.

---

## Leaderboard and Scoring

- **Leaderboard**: Live scores are displayed after each submission.
- **Scoring**: The score is calculated using `bitbybit.utils.score.calculate_submission_score`.
- **Final Ranking**: The top-ranked team at the end of the 24-hour period wins.

---

## References

- Charikar, M. (2002). [Similarity Estimation Techniques from Rounding Algorithms](https://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/CharikarEstim.pdf).
- Nguyen, T., et al. (2023). [DeepCAM: A Fully CAM-based Inference Accelerator with Variable Hash Lengths for Energy-efficient Deep Neural Networks](https://arxiv.org/abs/2302.04712).
- Chen, Y., et al. (2019). [SLIDE: In Defense of Smart Algorithms over Hardware Acceleration for Large-Scale Deep Learning Systems](https://arxiv.org/abs/1903.03129).
- Askary, H. (2023). [Intuitive Explanation of Straight-Through Estimators with PyTorch Implementation](https://hassanaskary.medium.com/intuitive-explanation-of-straight-through-estimators-with-pytorch-implementation-71d99d25d9d0).

---

**Built with 🔥 by SEMRON AI**


Conclusion
This hackathon is an exciting opportunity to innovate at the intersection of AI efficiency and hardware-aware computing. By leveraging LSH and analog in-memory computing, you can contribute to making AI more sustainable and scalable. Good luck, and may the best team win! 🏆

