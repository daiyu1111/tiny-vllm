#include <torch/extension.h>
#include <string>

torch::Tensor w8a8_gemm_cuda(
    torch::Tensor x_q,
    torch::Tensor a_scales,
    torch::Tensor qweight,
    torch::Tensor w_scales,
    c10::optional<torch::Tensor> bias,
    bool use_bfloat16);

std::string w8a8_gemm_path_cuda(
    torch::Tensor x_q,
    torch::Tensor qweight);

torch::Tensor w8a8_gemm(
    torch::Tensor x_q,
    torch::Tensor a_scales,
    torch::Tensor qweight,
    torch::Tensor w_scales,
    c10::optional<torch::Tensor> bias,
    bool use_bfloat16) {
  TORCH_CHECK(x_q.is_cuda(), "x_q must be a CUDA tensor");
  TORCH_CHECK(a_scales.is_cuda(), "a_scales must be a CUDA tensor");
  TORCH_CHECK(qweight.is_cuda(), "qweight must be a CUDA tensor");
  TORCH_CHECK(w_scales.is_cuda(), "w_scales must be a CUDA tensor");
  TORCH_CHECK(x_q.dim() == 2, "x_q must be 2D");
  TORCH_CHECK(qweight.dim() == 2, "qweight must be 2D");
  TORCH_CHECK(a_scales.dim() == 1, "a_scales must be 1D");
  TORCH_CHECK(w_scales.dim() == 1, "w_scales must be 1D");
  TORCH_CHECK(x_q.scalar_type() == at::kChar, "x_q must be int8");
  TORCH_CHECK(qweight.scalar_type() == at::kChar, "qweight must be int8");
  TORCH_CHECK(a_scales.scalar_type() == at::kFloat, "a_scales must be float32");
  TORCH_CHECK(w_scales.scalar_type() == at::kFloat, "w_scales must be float32");
  TORCH_CHECK(x_q.size(1) == qweight.size(1), "x_q and qweight shapes do not align");
  TORCH_CHECK(a_scales.size(0) == x_q.size(0), "a_scales must match x_q rows");
  TORCH_CHECK(w_scales.size(0) == qweight.size(0), "w_scales must match qweight rows");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias->dim() == 1, "bias must be 1D");
    TORCH_CHECK(bias->size(0) == qweight.size(0), "bias must match qweight rows");
    if (use_bfloat16) {
      TORCH_CHECK(bias->scalar_type() == at::kBFloat16, "bias must be bfloat16 when use_bfloat16=True");
    } else {
      TORCH_CHECK(bias->scalar_type() == at::kHalf, "bias must be float16 when use_bfloat16=False");
    }
  }
  return w8a8_gemm_cuda(x_q, a_scales, qweight, w_scales, bias, use_bfloat16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &w8a8_gemm, "W8A8 GEMM (CUDA)");
  m.def("select_path", &w8a8_gemm_path_cuda, "Return the W8A8 kernel path for the given tensors");
}
