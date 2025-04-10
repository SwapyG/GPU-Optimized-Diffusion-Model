# GPU-Optimized Diffusion Model

This project implements a diffusion model for image generation with GPU-level optimizations using custom CUDA kernels and TensorRT integration for significantly improved performance.

## Overview

Diffusion models work by gradually denoising random Gaussian noise until a coherent image emerges. This project provides a full implementation of the diffusion algorithm with three optimization levels:

1. **Base PyTorch Implementation** - A standard implementation using PyTorch operations
2. **CUDA-Optimized Version** - Enhanced with custom CUDA kernels for critical operations
3. **TensorRT-Accelerated Version** - Further optimized using NVIDIA's TensorRT for maximum performance

The implementation demonstrates how to apply GPU optimization techniques to deep learning models, with diffusion models as a practical and challenging example.

## Features

- Complete diffusion model pipeline (training and sampling)
- Simple U-Net architecture for the denoising network
- Support for MNIST, Fashion-MNIST, and CIFAR-10 datasets
- Custom CUDA kernels for optimized reverse diffusion steps
- TensorRT integration for model acceleration
- Comprehensive benchmarking capabilities
- Command-line interface for all operations

## Performance

Based on our benchmarks, you can expect the following relative performance improvements:

| Implementation | Relative Speed | Memory Usage | Notes |
|----------------|---------------|--------------|-------|
| Original PyTorch | 1x (baseline) | Standard | Good for training and experimentation |
| CUDA-optimized | ~2-3x faster | Slightly reduced | Recommended for production sampling |
| TensorRT | ~3-5x faster | Significantly reduced | Best for high-throughput applications |

*Note: Actual performance gains vary based on hardware, image size, and batch size.*

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.7 or later
- PyTorch 2.0 or later
- TensorRT 8.0 or later

### Dependencies

```bash
# Create a conda environment
conda create -n diffusion-opt python=3.10
conda activate diffusion-opt

# Install PyTorch with CUDA
conda install pytorch torchvision cudatoolkit=11.7 -c pytorch

# Install other dependencies
pip install matplotlib numpy tqdm tensorrt torch-tensorrt
```

### Building the CUDA Extension

You need to build the custom CUDA kernels before using the optimized implementation:

```bash
# Build and install the CUDA extension
python setup.py install
```

## Usage

### Training a Diffusion Model

```bash
# Train on MNIST (default)
python main.py --mode train --n_epochs 10

# Train on Fashion-MNIST
python main.py --mode train --dataset fashion_mnist --n_epochs 15 --img_size 28

# Train on CIFAR-10 (RGB images)
python main.py --mode train --dataset cifar10 --n_epochs 20 --img_size 32 --channels 3
```

### Generating Samples

```bash
# Using the original implementation
python main.py --mode sample --implementation original --n_samples 4

# Using the CUDA-optimized implementation
python main.py --mode sample --implementation cuda --n_samples 9

# Using the TensorRT implementation
python main.py --mode sample --implementation tensorrt --n_samples 16
```

### Benchmarking

```bash
# Run comprehensive benchmarks
python main.py --mode benchmark --n_samples 8 --img_size 32 --channels 3
```

## Project Structure

- `basic_diffusion.py` - Original PyTorch implementation with U-Net model
- `optimized_diffusion.py` - CUDA-optimized implementation using custom kernels
- `tensorrt_integration.py` - TensorRT model conversion and acceleration
- `diffusion_kernels.cu` - Custom CUDA kernels for key operations
- `setup.py` - CUDA extension builder
- `main.py` - Command-line interface for all operations

## Technical Details

### Diffusion Process

The diffusion process consists of two phases:

1. **Forward Process (q)**: Gradually adds Gaussian noise to an image over T timesteps
2. **Reverse Process (p)**: Learns to denoise images by predicting the added noise

The training objective is to train a neural network to predict the noise that was added at each step.

### U-Net Architecture

The project uses a simplified U-Net architecture with:

- Time embedding to condition the model on the diffusion timestep
- Downsampling path with convolutional layers
- Bottleneck with time embedding injection
- Upsampling path with transposed convolutions
- Skip connections (implicit in our implementation)

### GPU Optimizations

#### Custom CUDA Kernels

The custom CUDA implementation optimizes the reverse diffusion step:

```cuda
// Optimized reverse diffusion step kernel
__global__ void reverse_diffusion_step_kernel(
    float* x, const float* noise_pred, float alpha_t, float beta_t,
    float alpha_cumprod_t, const float* noise, bool add_noise,
    int total_elements)
{
    // ... kernel implementation ...
}
```

#### TensorRT Optimizations

TensorRT applies numerous optimizations including:

- FP16 precision for faster computation with minimal accuracy loss
- Kernel fusion to reduce memory access overhead
- Layer and operation fusion for improved efficiency
- Memory and workspace optimizations
- Graph optimizations for the specific GPU architecture

## Examples

### Generated MNIST Samples

After training, the model can generate sample digits from noise:

```
# Generate 16 MNIST samples
python main.py --mode sample --implementation tensorrt --n_samples 16 --model_path mnist_model.pt
```

### Benchmark Results

Sample benchmark output:

```
===== BENCHMARK RESULTS =====
Original PyTorch implementation: 12.35s
CUDA-optimized implementation: 4.82s (Speedup: 2.56x)
TensorRT implementation: 2.78s (Speedup vs original: 4.44x)
TensorRT vs CUDA-optimized speedup: 1.73x
```

## Extending the Project

### Adding New Datasets

Support for additional datasets can be added in `main.py`:

```python
# Example for adding a new dataset
elif args.dataset == "new_dataset":
    dataset_class = datasets.NewDataset
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    args.channels = 3
```

### Modifying the U-Net Architecture

The U-Net architecture can be modified in `basic_diffusion.py` by changing the `SimpleUNet` class.

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors:
- Reduce the batch size with `--batch_size`
- Reduce image size with `--img_size`
- Use a smaller model by modifying the U-Net architecture

### TensorRT Compilation Issues

If TensorRT compilation fails:
- Ensure you have the correct TensorRT version installed
- Try using `fp32` precision instead of `fp16`
- Check that your model structure is compatible with TensorRT

## License

MIT

## Citation

If you use this code in your research, please cite:

```
@software{gpu_optimized_diffusion,
  author = {Swapnil Gaikwad},
  title = {GPU-Optimized Diffusion Model},
  year = {2025},
}
```

## Acknowledgments

- This project is inspired by the Stable Diffusion architecture
- Optimization techniques are based on NVIDIA's best practices for deep learning
- Thanks to the PyTorch and CUDA communities for their excellent documentation and examples
