from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='TerActivateFunc',
    ext_modules=[
        CUDAExtension('TerActivateFunc_cuda', [
            'TerActivateFunc_cuda.cpp',
            'TerActivateFunc_cuda_kernel.cu'
            ]),
    ],

    cmdclass={
        'build_ext': BuildExtension
    })
