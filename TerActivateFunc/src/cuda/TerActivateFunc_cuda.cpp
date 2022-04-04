#include <torch/torch.h>

#include <vector>

// CUDA forward declarations
void TerActivateFunc_cuda_backward(
    at::Tensor input,
    at::Tensor gradInput
);

// CUDA forward declarations
void TerActivateFunc_cuda_forward(
    at::Tensor input,
    at::Tensor output
);

// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor TerActivateFunc_forward(
    at::Tensor input,
    at::Tensor output) 
{
    CHECK_INPUT(input);

    TerActivateFunc_cuda_forward(input, output);

    return output;
}

int TerActivateFunc_backward(
    at::Tensor input,
    at::Tensor gradInput,
    const float threshold) 
{
    CHECK_INPUT(input);
    CHECK_INPUT(gradInput);

    TerActivateFunc_cuda_backward(input, gradInput);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &TerActivateFunc_forward, "TerActivateFunc forward (CUDA)");
  m.def("backward", &TerActivateFunc_backward, "TerActivateFunc backward (CUDA)");
}
