import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import our modules
from basic_diffusion import SimpleUNet, DiffusionModel, train_diffusion_model
from optimized_diffusion import OptimizedDiffusionModel
from tensorrt_integration import TensorRTDiffusionModel, benchmark_all_methods


def parse_args():
    parser = argparse.ArgumentParser(description="GPU-Optimized Diffusion Model")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample", "benchmark"],
                        help="Mode to run: train, sample, or benchmark")
    parser.add_argument("--model_path", type=str, default="diffusion_model.pt",
                        help="Path to load/save the model")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training or sampling")
    parser.add_argument("--img_size", type=int, default=28,
                        help="Image size (assumed square)")
    parser.add_argument("--channels", type=int, default=1,
                        help="Number of image channels (1 for grayscale, 3 for RGB)")
    parser.add_argument("--n_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--n_steps", type=int, default=1000,
                        help="Number of diffusion steps")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Number of samples to generate")
    parser.add_argument("--implementation", type=str, default="original",
                        choices=["original", "cuda", "tensorrt"],
                        help="Implementation to use for sampling")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist", "cifar10"],
                        help="Dataset to use for training")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up dataset based on the selected option
    if args.dataset == "mnist":
        dataset_class = datasets.MNIST
        mean, std = (0.5,), (0.5,)
    elif args.dataset == "fashion_mnist":
        dataset_class = datasets.FashionMNIST
        mean, std = (0.5,), (0.5,)
    elif args.dataset == "cifar10":
        dataset_class = datasets.CIFAR10
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        args.channels = 3

    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Create output directory for samples
    os.makedirs("samples", exist_ok=True)

    # Mode-specific operations
    if args.mode == "train":
        # Data loading for training
        dataset = dataset_class(root='./data', train=True, download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        # Model setup
        model = SimpleUNet(in_channels=args.channels, out_channels=args.channels, time_dim=256)
        diffusion = DiffusionModel(model, n_steps=args.n_steps, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Train the model
        print(f"Training diffusion model for {args.n_epochs} epochs...")
        train_diffusion_model(diffusion, dataloader, optimizer, args.n_epochs, device)

        # Save model
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")

        # Generate and save some samples
        print("Generating samples...")
        samples = diffusion.sample(args.n_samples, args.img_size, args.channels)
        save_samples(samples, args.channels, "samples/training_samples.png")

    elif args.mode == "sample":
        # Load model
        model = SimpleUNet(in_channels=args.channels, out_channels=args.channels, time_dim=256)
        model.load_state_dict(torch.load(args.model_path))
        model.to(device)

        # Select implementation
        if args.implementation == "original":
            diffusion = DiffusionModel(model, n_steps=args.n_steps, device=device)
            print("Using original PyTorch implementation")
        elif args.implementation == "cuda":
            diffusion = OptimizedDiffusionModel(model, n_steps=args.n_steps, device=device)
            print("Using CUDA-optimized implementation")
        elif args.implementation == "tensorrt":
            diffusion = TensorRTDiffusionModel(args.model_path, n_steps=args.n_steps, device=device,
                                               img_size=args.img_size, channels=args.channels,
                                               batch_size=args.n_samples)
            print("Using TensorRT implementation")

        # Generate samples
        print(f"Generating {args.n_samples} samples...")
        samples = diffusion.sample(args.n_samples, args.img_size, args.channels)

        # Save samples
        output_file = f"samples/{args.implementation}_samples.png"
        save_samples(samples, args.channels, output_file)
        print(f"Samples saved to {output_file}")

    elif args.mode == "benchmark":
        print("Running benchmark of all implementations...")
        benchmark_results = benchmark_all_methods(
            args.model_path,
            n_samples=args.n_samples,
            img_size=args.img_size,
            channels=args.channels
        )

        # Save benchmark results
        with open("benchmark_results.txt", "w") as f:
            f.write("===== DIFFUSION MODEL BENCHMARK RESULTS =====\n")
            f.write(f"Image size: {args.img_size}x{args.img_size}, Channels: {args.channels}\n")
            f.write(f"Number of samples: {args.n_samples}, Diffusion steps: {args.n_steps}\n\n")

            f.write(f"Original PyTorch: {benchmark_results['original_time']:.4f}s\n")
            f.write(
                f"CUDA-optimized: {benchmark_results['optimized_time']:.4f}s (Speedup: {benchmark_results['cuda_speedup']:.2f}x)\n")
            f.write(
                f"TensorRT: {benchmark_results['tensorrt_time']:.4f}s (Speedup: {benchmark_results['trt_speedup']:.2f}x)\n")

        print(f"Benchmark results saved to benchmark_results.txt")


def save_samples(samples, channels, filename):
    """Save generated samples as an image grid."""
    # Denormalize
    samples = (samples + 1) / 2

    # Convert to numpy for plotting
    samples = samples.cpu().detach().numpy()

    # Create a grid
    grid_size = int(np.sqrt(samples.shape[0]))
    if grid_size * grid_size != samples.shape[0]:
        grid_size = samples.shape[0]  # Just use a single row if not a perfect square
        grid_h, grid_w = 1, grid_size
    else:
        grid_h, grid_w = grid_size, grid_size

    plt.figure(figsize=(grid_w * 2, grid_h * 2))

    for i in range(samples.shape[0]):
        plt.subplot(grid_h, grid_w, i + 1)
        if channels == 1:
            plt.imshow(samples[i, 0], cmap='gray')
        else:
            plt.imshow(np.transpose(samples[i], (1, 2, 0)))
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    main()