#include "xops.h"
#include "dispatch_utils.h"
#include <torch/extension.h>
torch::Tensor weak_ref_tensor(torch::Tensor& tensor) {
    // Ensure tensor is on CUDA
    if (!tensor.is_cuda()) {
        throw std::runtime_error("Tensor must be on CUDA device");
    }

    // Get the raw data pointer
    void* data_ptr = tensor.data_ptr();

    // Get tensor sizes and strides
    std::vector<int64_t> sizes = tensor.sizes().vec();
    std::vector<int64_t> strides = tensor.strides().vec();

    // Get tensor options (dtype, device)
    auto options = tensor.options();

    // Create a new tensor from the raw data pointer
    auto new_tensor = torch::from_blob(data_ptr, sizes, strides, options);

    return new_tensor;
}

// On Kunlun, vllm's own `_C.abi3.so` is never built (no `vllm._C` module), so
// nothing registers `torch.ops._C.weak_ref_tensor`. CUDA graph capture in vLLM
// calls it, so we must register it here. A catch-all kernel (function passed to
// def) is used instead of a CUDA-keyed impl so it also matches XPU tensors that
// masquerade as CUDA under XPytorch.
TORCH_LIBRARY(_C, m) {
    m.def("weak_ref_tensor(Tensor input) -> Tensor", &weak_ref_tensor);
}

PYBIND11_MODULE(_kunlun, m) {
    m.def("weak_ref_tensor", &weak_ref_tensor);
}
