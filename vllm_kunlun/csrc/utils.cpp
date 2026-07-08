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

// NOTE: vllm's own _C.abi3.so already defines `_C::weak_ref_tensor`
// (csrc/torch_bindings.cpp). Do NOT register it here or torch aborts with a
// duplicate-operator error. We only expose it via the pybind module below for
// direct `vllm_kunlun._kunlun.weak_ref_tensor` access; torch.ops._C.weak_ref_tensor
// is provided by vllm.

PYBIND11_MODULE(_kunlun, m) {
    m.def("weak_ref_tensor", &weak_ref_tensor);
}
