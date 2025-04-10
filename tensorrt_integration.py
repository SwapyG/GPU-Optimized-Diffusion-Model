import torch
import torch_tensorrt
import time
import numpy as np
from basic_diffusion import SimpleUNet


class TensorRTDiffusionModel:
    """Implementation of diffusion model sampling using TensorRT for faster inference."""

    def __init__(self, model_path, n_steps=1000, beta_min=1e-4, beta_max=0.02, device="cuda",
                 img_size=28, channels=1, batch_size=4):
        self.device = device
        self.n_steps = n_steps
        self.img_size = img_size
        self.channels = channels

        # Load the PyTorch model
        self.orig_model = SimpleUNet(in_channels=channels, out_channels=channels, time_dim=256)
        self.orig_model.load_state_dict(torch.load(model_path))
        self.orig_model.eval().to(device)

        # Define noise schedule (linear beta schedule)
        self.betas = torch.linspace(beta_min, beta_max, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Compile with TensorRT
        print("Compiling model with TensorRT...")

        # Create example inputs for tracing
        example_x = torch.randn(batch_size, channels, img_size, img_size, device=device)
        example_t = torch.zeros(batch_size, dtype=torch.long, device=device)

        # TensorRT settings
        trt_settings = {
            "enabled": True,
            "max_batch_size": batch_size,
            "workspace_size": 1 << 30,  # 1GB
            "min_block_size": 1,
            "precision": "fp16",  # Use FP16 for faster inference
            "strict_type_constraints": False,
            "preserve_parameters": True
        }

        # Trace and compile the model
        self.trt_model = torch_tensorrt.compile(
            self.orig_model,
            inputs=[example_x, example_t],
            enabled_precisions={"fp16", "fp32"},  # Allow mixed precision
            **trt_settings
        )

        print("TensorRT compilation complete!")

    def sample(self, n_samples, img_size=None, channels=None):
        """Generate new images using the diffusion model with TensorRT acceleration."""
        if img_size is None:
            img_size = self.img_size
        if channels is None:
            channels = self.channels

        # Start with random noise
        x = torch.randn(n_samples, channels, img_size, img_size).to(self.device)

        # Start timer for performance measurement
        start_time = time.time()

        # Gradually denoise the image
        for t in range(self.n_steps - 1, -1, -1):
            # Print progress every 100 steps
            if t % 100 == 0:
                print(f"Sampling step {self.n_steps - t}/{self.n_steps}, "
                      f"time elapsed: {time.time() - start_time:.2f}s")

            # Batch of timesteps
            t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)

            # Predict noise using TensorRT model
            with torch.no_grad():
                noise_pred = self.trt_model(x, t_batch)

            # Compute parameters for denoising step
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_cumprod_t) * noise_pred) + sigma_t * noise
            x = (1 / alpha_t.sqrt()) * (
                    x - (beta_t / (1 - alpha_cumprod_t).sqrt()) * noise_pred
            ) + beta_t.sqrt() * noise

        total_time = time.time() - start_time
        print(f"TensorRT sampling completed in {total_time:.2f}s, {total_time / n_samples:.2f}s per image")

        return x


def benchmark_all_methods(model_path, n_samples=4, img_size=28, channels=1):
    """Compare performance between original, CUDA-optimized, and TensorRT implementations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = SimpleUNet(in_channels=channels, out_channels=channels, time_dim=256)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Initialize all implementations
    from basic_diffusion import DiffusionModel
    from optimized_diffusion import OptimizedDiffusionModel

    print("Setting up original diffusion model...")
    original_model = DiffusionModel(model, n_steps=1000, device=device)

    print("Setting up CUDA-optimized diffusion model...")
    optimized_model = OptimizedDiffusionModel(model, n_steps=1000, device=device)

    print("Setting up TensorRT diffusion model...")
    tensorrt_model = TensorRTDiffusionModel(model_path, n_steps=1000, device=device,
                                            img_size=img_size, channels=channels, batch_size=n_samples)

    # Run warmup for all models
    print("Running warmup...")
    with torch.no_grad():
        _ = original_model.sample(1, img_size, channels)
        _ = optimized_model.sample(1, img_size, channels)
        _ = tensorrt_model.sample(1, img_size, channels)

    # Benchmark original implementation
    print("\nBenchmarking original implementation...")
    torch.cuda.synchronize()
    start_time = time.time()
    _ = original_model.sample(n_samples, img_size, channels)
    torch.cuda.synchronize()
    original_time = time.time() - start_time

    # Benchmark CUDA-optimized implementation
    print("\nBenchmarking CUDA-optimized implementation...")
    torch.cuda.synchronize()
    start_time = time.time()
    _ = optimized_model.sample(n_samples, img_size, channels)
    torch.cuda.synchronize()
    optimized_time = time.time() - start_time

    # Benchmark TensorRT implementation
    print("\nBenchmarking TensorRT implementation...")
    torch.cuda.synchronize()
    start_time = time.time()
    _ = tensorrt_model.sample(n_samples, img_size, channels)
    torch.cuda.synchronize()
    tensorrt_time = time.time() - start_time

    # Calculate speedups
    cuda_speedup = original_time / optimized_time
    trt_speedup = original_time / tensorrt_time
    trt_vs_cuda_speedup = optimized_time / tensorrt_time

    # Print results
    print("\n===== BENCHMARK RESULTS =====")
    print(f"Original PyTorch implementation: {original_time:.2f}s")
    print(f"CUDA-optimized implementation: {optimized_time:.2f}s (Speedup: {cuda_speedup:.2f}x)")
    print(f"TensorRT implementation: {tensorrt_time:.2f}s (Speedup vs original: {trt_speedup:.2f}x)")
    print(f"TensorRT vs CUDA-optimized speedup: {trt_vs_cuda_speedup:.2f}x")

    return {
        'original_time': original_time,
        'optimized_time': optimized_time,
        'tensorrt_time': tensorrt_time,
        'cuda_speedup': cuda_speedup,
        'trt_speedup': trt_speedup,
        'trt_vs_cuda_speedup': trt_vs_cuda_speedup
    }