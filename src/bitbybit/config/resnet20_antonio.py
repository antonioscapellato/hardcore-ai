resnet20_full_patch_config = {
    # Common parameters for all layers
    "common_params": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },

    # --- Top-level Conv ---
    "conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 4096
    },

    # --- Layer 1 (16 channels in, 16 channels out for convs) ---
    "layer1.0.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer1.0.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer1.1.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer1.1.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer1.2.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer1.2.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },

    # --- Layer 2 (16/32 channels in, 32 channels out for convs) ---
    "layer2.0.conv1": { # Strided conv (16 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer2.0.conv2": { # (32 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer2.0.downsample.0": { # Downsample conv (16 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer2.1.conv1": { # (32 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer2.1.conv2": { # (32 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer2.2.conv1": { # (32 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer2.2.conv2": { # (32 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },

    # --- Layer 3 (32/64 channels in, 64 channels out for convs) ---
    "layer3.0.conv1": { # Strided conv (32 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 4096
    },
    "layer3.0.conv2": { # (64 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 4096
    },
    "layer3.0.downsample.0": { # Downsample conv (32 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 4096
    },
    "layer3.1.conv1": { # (64 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 4096
    },
    "layer3.1.conv2": { # (64 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 4096
    },
    "layer3.2.conv1": { # (64 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 4096
    },
    "layer3.2.conv2": { # (64 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 4096
    },

    # --- Fully Connected Layer (64 in_features, 10 out_features) ---
    "fc": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 4096
    }
}

# Optimized Configuration for CIFAR-10
submission_config_cifar10 = {
    # Common parameters for all layers (baseline, overridden by specific layers)
    "common_params": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024  # Default for simpler dataset
    },

    # --- Top-level Conv ---
    "conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024  # Early layer, low complexity
    },

    # --- Layer 1 (16 channels in, 16 channels out for convs) ---
    "layer1.0.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer1.0.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer1.1.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer1.1.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer1.2.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer1.2.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },

    # --- Layer 2 (16/32 channels in, 32 channels out for convs) ---
    "layer2.0.conv1": { # Strided conv (16 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024  # Moderate complexity
    },
    "layer2.0.conv2": { # (32 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer2.0.downsample.0": { # Downsample conv (16 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer2.1.conv1": { # (32 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer2.1.conv2": { # (32 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer2.2.conv1": { # (32 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer2.2.conv2": { # (32 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },

    # --- Layer 3 (32/64 channels in, 64 channels out for convs) ---
    "layer3.0.conv1": { # Strided conv (32 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048  # Critical transition layer
    },
    "layer3.0.conv2": { # (64 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer3.0.downsample.0": { # Downsample conv (32 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048  # Critical for downsampling
    },
    "layer3.1.conv1": { # (64 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer3.1.conv2": { # (64 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer3.2.conv1": { # (64 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer3.2.conv2": { # (64 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },

    # --- Fully Connected Layer (64 in_features, 10 out_features) ---
    "fc": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048  # Critical for final classification
    }
}

# Optimized Configuration for CIFAR-100
submission_config_cifar100 = {
    # Common parameters for all layers (baseline, overridden by specific layers)
    "common_params": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 4096  # Default for complex dataset
    },

    # --- Top-level Conv ---
    "conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048  # Early layer, moderate complexity
    },

    # --- Layer 1 (16 channels in, 16 channels out for convs) ---
    "layer1.0.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer1.0.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer1.1.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer1.1.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer1.2.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer1.2.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },

    # --- Layer 2 (16/32 channels in, 32 channels out for convs) ---
    "layer2.0.conv1": { # Strided conv (16 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer2.0.conv2": { # (32 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer2.0.downsample.0": { # Downsample conv (16 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer2.1.conv1": { # (32 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer2.1.conv2": { # (32 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer2.2.conv1": { # (32 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer2.2.conv2": { # (32 in, 32 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },

    # --- Layer 3 (32/64 channels in, 64 channels out for convs) ---
    "layer3.0.conv1": {  # Strided conv (32 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048  # Critical transition layer
    },
    "layer3.0.conv2": {  # (64 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024
    },
    "layer3.0.downsample.0": {  # Downsample conv (32 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048  # Critical for downsampling
    },
    "layer3.1.conv1": {  # (64 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer3.1.conv2": {  # (64 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer3.2.conv1": {  # (64 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },
    "layer3.2.conv2": {  # (64 in, 64 out)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048
    },

    # --- Fully Connected Layer (64 in_features, 100 out_features) ---
    "fc": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 4096  # Critical for final classification
    }
}

