// diffusion_kernels.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for the reverse diffusion step (denoising)
// This combines several operations that normally would require multiple GPU kernel launches
template <typename scalar_t>
__global__ void reverse_diffusion_step_kernel(
    scalar_t* x,                   // Current noisy image x_t
    scalar_t* noise_pred,          // Predicted noise
    scalar_t* output,              // Output denoised image x_{t-1}
    scalar_t alpha_t,              // alpha_t parameter
    scalar_t beta_t,               // beta_t parameter
    scalar_t one_minus_alpha_cumprod_t, // 1 - alpha_cumprod_t
    scalar_t* noise,               // Random noise to add (can be 0 for final step)
    int size,                      // Total number of elements
    bool add_noise                 // Whether to add noise
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Compute x_{t-1} in a single fused operation
        // x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_cumprod_t) * noise_pred) + sigma_t * noise
        const scalar_t noise_scale = beta_t / sqrt(one_minus_alpha_cumprod_t);
        const scalar_t alpha_scale = 1.0 / sqrt(alpha_t);

        scalar_t result = alpha_scale * (x[idx] - noise_scale * noise_pred[idx]);

        // Add random noise if needed (not the final step)
        if (add_noise) {
            result += sqrt(beta_t) * noise[idx];
        }

        output[idx] = result;
    }
}

// C++ wrapper function that will be called from Python
torch::Tensor reverse_diffusion_step_cuda(
    torch::Tensor x,
    torch::Tensor noise_pred,
    float alpha_t,
    float beta_t,
    float alpha_cumprod_t,
    torch::Tensor noise,
    bool add_noise
) {
    const int batch_size = x.size(0);
    const int channels = x.size(1);
    const int height = x.size(2);
    const int width = x.size(3);
    const int size = batch_size * channels * height * width;

    // Create output tensor
    auto output = torch::empty_like(x);

    // Calculate parameters
    const float one_minus_alpha_cumprod_t = 1.0f - alpha_cumprod_t;

    // Determine thread configuration
    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(x.type(), "reverse_diffusion_step_cuda", ([&] {
        reverse_diffusion_step_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            noise_pred.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<scalar_t>(alpha_t),
            static_cast<scalar_t>(beta_t),
            static_cast<scalar_t>(one_minus_alpha_cumprod_t),
            noise.data_ptr<scalar_t>(),
            size,
            add_noise
        );
    }));

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reverse_diffusion_step", &reverse_diffusion_step_cuda, "Reverse diffusion step (CUDA)");
}