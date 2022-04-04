#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

namespace {
template <typename scalar_t>
__global__ void TerActivateFunc_cuda_backward_kernel(
    const int nthreads,
    const scalar_t* __restrict__ input_data,
    scalar_t* __restrict__ gradInput_data) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        if (*(input_data + n) > 1. || *(input_data + n) < -1.) {
            *(gradInput_data + n) = 0.;
        }
    }
}
} // namespace

namespace {
template <typename scalar_t>
__global__ void TerActivateFunc_cuda_forward_kernel(
    const int nthreads,
    const scalar_t* __restrict__ input_data,
    scalar_t* __restrict__ output_data) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        if (*(input_data + n) < -0.5) {
            *(output_data + n) = -1.;
        } else if (*(input_data + n) < 0.5) {
            *(output_data + n) = 0.;
        } else {
            *(output_data + n) = 1.;
        }
    }
}
} // namespace

void TerActivateFunc_cuda_backward(
    at::Tensor input,
    at::Tensor gradInput) 
{
    const int nthreads = input.numel();
    const int CUDA_NUM_THREADS = 1024;
    const int nblocks = (nthreads + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "SiBinActivateFunc_cuda_backward", ([&] {
        TerActivateFunc_cuda_backward_kernel<scalar_t><<<nblocks, CUDA_NUM_THREADS>>>(
            nthreads,
            input.data<scalar_t>(),
            gradInput.data<scalar_t>());
    }));
}

void TerActivateFunc_cuda_forward(
    at::Tensor input,
    at::Tensor output) 
{
    const int nthreads = input.numel();
    const int CUDA_NUM_THREADS = 1024;
    const int nblocks = (nthreads + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "SiBinActivateFunc_cuda_forward", ([&] {
        TerActivateFunc_cuda_forward_kernel<scalar_t><<<nblocks, CUDA_NUM_THREADS>>>(
            nthreads,
            input.data<scalar_t>(),
            output.data<scalar_t>());
    }));
}
