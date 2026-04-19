#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/bfloat16.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>
#include <torch/extension.h>

#include <string>

namespace {

template <typename scalar_t>
struct cutlass_scalar_type;

template <>
struct cutlass_scalar_type<at::Half> {
  using type = cutlass::half_t;
};

template <>
struct cutlass_scalar_type<at::BFloat16> {
  using type = cutlass::bfloat16_t;
};

template <typename scalar_t>
using cutlass_scalar_t = typename cutlass_scalar_type<scalar_t>::type;

template <typename scalar_t>
__device__ __forceinline__ cutlass_scalar_t<scalar_t> cutlass_zero();

template <>
__device__ __forceinline__ cutlass::half_t cutlass_zero<at::Half>() {
  return cutlass::half_t(0.0f);
}

template <>
__device__ __forceinline__ cutlass::bfloat16_t cutlass_zero<at::BFloat16>() {
  return cutlass::bfloat16_t(0.0f);
}

template <typename scalar_t>
__device__ __forceinline__ cutlass_scalar_t<scalar_t> cutlass_from_float(float value);

template <>
__device__ __forceinline__ cutlass::half_t cutlass_from_float<at::Half>(float value) {
  return cutlass::half_t(value);
}

template <>
__device__ __forceinline__ cutlass::bfloat16_t cutlass_from_float<at::BFloat16>(float value) {
  return cutlass::bfloat16_t(value);
}

template <typename scalar_t>
__global__ void dequantize_transpose_kernel(
    const int8_t* __restrict__ qweight,
    const float* __restrict__ scales,
    cutlass_scalar_t<scalar_t>* __restrict__ weight_t,
    int n,
    int k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * k;
  if (idx >= total) {
    return;
  }
  int row = idx / k;
  int col = idx % k;
  float value = static_cast<float>(qweight[idx]) * scales[row];
  weight_t[col * n + row] = cutlass_from_float<scalar_t>(value);
}

template <typename scalar_t>
__global__ void add_bias_kernel(
    scalar_t* __restrict__ y,
    const scalar_t* __restrict__ bias,
    int m,
    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = m * n;
  if (idx >= total) {
    return;
  }
  int col = idx % n;
  y[idx] = y[idx] + bias[col];
}

template <typename scalar_t>
torch::Tensor launch_cutlass_gemm(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    c10::optional<torch::Tensor> bias) {
  using Element = cutlass_scalar_t<scalar_t>;
  using Gemm = cutlass::gemm::device::Gemm<
      Element,
      cutlass::layout::RowMajor,
      Element,
      cutlass::layout::RowMajor,
      Element,
      cutlass::layout::RowMajor,
      float>;

  auto y = torch::zeros({x.size(0), qweight.size(0)}, x.options());
  auto weight_t = torch::empty({qweight.size(1), qweight.size(0)}, x.options());

  const int n = static_cast<int>(qweight.size(0));
  const int k = static_cast<int>(qweight.size(1));
  const int m = static_cast<int>(x.size(0));
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  constexpr int threads = 256;
  int total_weight = n * k;
  int blocks = (total_weight + threads - 1) / threads;
  dequantize_transpose_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
      qweight.data_ptr<int8_t>(),
      scales.data_ptr<float>(),
      reinterpret_cast<Element*>(weight_t.data_ptr<scalar_t>()),
      n,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  Gemm gemm_op;
  cutlass::Status status = gemm_op({
      {m, n, k},
      {reinterpret_cast<Element const*>(x.data_ptr<scalar_t>()), k},
      {reinterpret_cast<Element const*>(weight_t.data_ptr<scalar_t>()), n},
      {reinterpret_cast<Element const*>(y.data_ptr<scalar_t>()), n},
      {reinterpret_cast<Element*>(y.data_ptr<scalar_t>()), n},
      {1.0f, 0.0f},
  }, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM launch failed");

  if (bias.has_value()) {
    int total_y = m * n;
    int bias_blocks = (total_y + threads - 1) / threads;
    add_bias_kernel<scalar_t><<<bias_blocks, threads, 0, stream>>>(
        y.data_ptr<scalar_t>(),
        bias->data_ptr<scalar_t>(),
        m,
        n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return y;
}

}  // namespace

std::string int8_weight_only_gemm_cutlass_path_cuda(
    torch::Tensor x,
    torch::Tensor qweight) {
  (void)qweight;
  return x.scalar_type() == at::kHalf ? "cutlass_fp16_gemm" : "cutlass_bf16_gemm";
}

torch::Tensor int8_weight_only_gemm_cutlass_cuda(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    c10::optional<torch::Tensor> bias) {
  if (x.scalar_type() == at::kHalf) {
    return launch_cutlass_gemm<at::Half>(x, qweight, scales, bias);
  }
  return launch_cutlass_gemm<at::BFloat16>(x, qweight, scales, bias);
}
