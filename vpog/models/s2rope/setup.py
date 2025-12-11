from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

all_cuda_archs = cuda.get_gencode_flags().replace("compute=", "arch=").split()

setup(
    name="s2rope",
    # packages=["models", "models.s2rope"],
    # package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            name="s2rope_ext",
            sources=[
                "s2rope.cpp",
                "s2rope_kernels.cu",
            ],
            extra_compile_args=dict(
                nvcc=["-O3", "--ptxas-options=-v", "--use_fast_math"] + all_cuda_archs,
                cxx=["-O3"],
            ),
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
