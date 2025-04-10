import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# Import the custom CUDA extension (after building it)
# Run this first: python setup.py install
import diffusion_cuda


class OptimizedDiffusionModel:
    """Optimized implementation of the diffusion process using custom CUDA kernels."""

    def __init__(self, model, n_steps=1000, beta_min=1e-4, beta_max=0.02, device="cuda"):
        self.model = model.to(device)
        self.n_steps = n_steps
        self.device = device

        # Define noise schedule (linear beta schedule)
        self.betas = torch.linspace(beta_min, beta_max, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Pre-compute parameters to avoid redundant calculations
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def add_noise(self, x_0, t):
        """Add noise to the input image according to the diffusion schedule at time t."""
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)

        # q(x_t | x_0) = sqrt(alphas_cumprod) * x_0 + sqrt(1 - alphas_cumprod) * noise
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def training_loss(self, x_0, t):
        """Compute the training loss for the model."""
        x_noisy, noise_added = self.add_noise(x_0, t)

        # Predict the noise that was added
        noise_pred = self.model(x_noisy, t)

        # Loss is the mean squared error between the added noise and predicted noise
        return F.mse_loss(noise_pred, noise_added)

    def sample(self, n_samples, img_size, channels=1):
        """Generate new images using the trained diffusion model with CUDA optimization."""
        self.model.eval()

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

            # Predict noise
            with torch.no_grad():
                noise_pred = self.model(x, t_batch)

            # Prepare parameters for denoising step
            alpha_t = self.alphas[t].item()
            alpha_cumprod_t = self.alphas_cumprod[t].item()
            beta_t = self.betas[t].item()

            # For the final step (t=0), we don't add noise
            add_noise = t > 0

            if add_noise:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # Use custom CUDA kernel for the reverse diffusion step
            x = diffusion_cuda.reverse_diffusion_step(
                x,
                noise_pred,
                alpha_t,
                beta_t,
                alpha_cumprod_t,
                noise,
                add_noise
            )

        total_time = time.time() - start_time
        print(f"Sampling completed in {total_time:.2f}s, {total_time / n_samples:.2f}s per image")

        self.model.train()
        return x


# Benchmark function to compare original vs optimized implementation
def benchmark_diffusion(original_model, optimized_model, n_samples=4, img_size=28, channels=1, device="cuda"):
    """Compare performance between original and optimized diffusion models."""
    print("Benchmarking original implementation...")
    torch.cuda.synchronize()
    start_time = time.time()
    _ = original_model.sample(n_samples, img_size, channels)
    torch.cuda.synchronize()
    original_time = time.time() - start_time

    print("Benchmarking optimized implementation...")
    torch.cuda.synchronize()
    start_time = time.time()
    _ = optimized_model.sample(n_samples, img_size, channels)
    torch.cuda.synchronize()
    optimized_time = time.time() - start_time

    speedup = original_time / optimized_time
    print(f"Original implementation: {original_time:.2f}s")
    print(f"Optimized implementation: {optimized_time:.2f}s")
    print(f"Speedup: {speedup:.2f}x")

    return original_time, optimized_time, speedup