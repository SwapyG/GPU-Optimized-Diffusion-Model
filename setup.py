from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="diffusion_cuda",
    ext_modules=[
        CUDAExtension(
            name="diffusion_cuda",
            sources=["diffusion_kernels.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"]
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)