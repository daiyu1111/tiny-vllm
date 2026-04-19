#include <torch/extension.h>
#include <string>

torch::Tensor int8_weight_only_gemm_cutlass_cuda(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    c10::optional<torch::Tensor> bias);

std::string int8_weight_only_gemm_cutlass_path_cuda(
    torch::Tensor x,
    torch::Tensor qweight);

torch::Tensor int8_weight_only_gemm_cutlass(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    c10::optional<torch::Tensor> bias) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(qweight.is_cuda(), "qweight must be a CUDA tensor");
  TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(qweight.dim() == 2, "qweight must be 2D");
  TORCH_CHECK(scales.dim() == 1, "scales must be 1D");
  TORCH_CHECK(qweight.scalar_type() == at::kChar, "qweight must be int8");
  TORCH_CHECK(scales.scalar_type() == at::kFloat, "scales must be float32");
  TORCH_CHECK(
      x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
      "x must be float16 or bfloat16");
  TORCH_CHECK(x.size(1) == qweight.size(1), "x and qweight shapes do not align");
  TORCH_CHECK(scales.size(0) == qweight.size(0), "scales must match qweight rows");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias->dim() == 1, "bias must be 1D");
    TORCH_CHECK(bias->size(0) == qweight.size(0), "bias must match qweight rows");
    TORCH_CHECK(
        bias->scalar_type() == x.scalar_type(),
        "bias must match x dtype for CUTLASS backend");
  }
  return int8_weight_only_gemm_cutlass_cuda(x, qweight, scales, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &int8_weight_only_gemm_cutlass, "INT8 weight-only GEMM via CUTLASS");
  m.def("select_path", &int8_weight_only_gemm_cutlass_path_cuda, "Return the CUTLASS backend path");
}
