#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

namespace {

using namespace nvcuda;

constexpr int FB_BLOCK_M = 64;
constexpr int FB_BLOCK_N = 64;
constexpr int FB_BLOCK_K = 32;
constexpr int FB_THREADS_X = 16;
constexpr int FB_THREADS_Y = 16;
constexpr int FB_THREAD_TILE_M = FB_BLOCK_M / FB_THREADS_Y;  // 4
constexpr int FB_THREAD_TILE_N = FB_BLOCK_N / FB_THREADS_X;  // 4
constexpr int FB_X_SKEW = 8;
constexpr int FB_W_SKEW = 16;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WMMA_BLOCK_ROW_WARPS = 2;
constexpr int WMMA_BLOCK_COL_WARPS = 4;
constexpr int WMMA_WARPS_PER_BLOCK = WMMA_BLOCK_ROW_WARPS * WMMA_BLOCK_COL_WARPS;
constexpr int WMMA_THREADS_PER_BLOCK = WMMA_WARPS_PER_BLOCK * 32;
constexpr int WMMA_BLOCK_M = WMMA_BLOCK_ROW_WARPS * WMMA_M;  // 32
constexpr int WMMA_BLOCK_N = WMMA_BLOCK_COL_WARPS * WMMA_N;  // 64
constexpr int WMMA_BLOCK_K = 32;
constexpr int WMMA_A_SKEW = 8;
constexpr int WMMA_BQ_SKEW = 16;
constexpr int WMMA_B_SKEW = 8;
constexpr int WMMA_C_SKEW = 8;
constexpr int WMMA_A_STRIDE = WMMA_BLOCK_K + WMMA_A_SKEW;
constexpr int WMMA_BQ_STRIDE = WMMA_BLOCK_K + WMMA_BQ_SKEW;
constexpr int WMMA_B_STRIDE = WMMA_BLOCK_N + WMMA_B_SKEW;
constexpr int WMMA_C_STRIDE = WMMA_BLOCK_N + WMMA_C_SKEW;
constexpr int CP_ASYNC_BYTES = 16;
constexpr int CP_ASYNC_A_HALFS = CP_ASYNC_BYTES / static_cast<int>(sizeof(half));

static_assert(WMMA_THREADS_PER_BLOCK == 256, "Expected 256 threads per WMMA block");

template <typename scalar_t>
__device__ __forceinline__ scalar_t cast_from_float(float value);

template <>
__device__ __forceinline__ at::Half cast_from_float<at::Half>(float value) {
  return static_cast<at::Half>(value);
}

template <>
__device__ __forceinline__ at::BFloat16 cast_from_float<at::BFloat16>(float value) {
  return static_cast<at::BFloat16>(value);
}

template <typename scalar_t>
__device__ __forceinline__ float load_input(const scalar_t* ptr, int idx);

template <>
__device__ __forceinline__ float load_input<at::Half>(const at::Half* ptr, int idx) {
  return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
}

template <>
__device__ __forceinline__ float load_input<at::BFloat16>(const at::BFloat16* ptr, int idx) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
#else
  return static_cast<float>(ptr[idx]);
#endif
}

template <typename scalar_t>
__device__ __forceinline__ float load_bias(const scalar_t* ptr, int idx);

template <>
__device__ __forceinline__ float load_bias<at::Half>(const at::Half* ptr, int idx) {
  return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
}

template <>
__device__ __forceinline__ float load_bias<at::BFloat16>(const at::BFloat16* ptr, int idx) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
#else
  return static_cast<float>(ptr[idx]);
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
  const unsigned smem_int = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_int), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}
#endif

template <typename scalar_t>
__global__ void int8_weight_only_gemm_fallback_kernel(
    const scalar_t* __restrict__ x,
    const int8_t* __restrict__ qweight,
    const float* __restrict__ scales,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ y,
    int m,
    int n,
    int k) {
  __shared__ float x_tile[FB_BLOCK_M][FB_BLOCK_K + FB_X_SKEW];
  __shared__ int8_t w_tile[FB_BLOCK_N][FB_BLOCK_K + FB_W_SKEW];
  __shared__ float scale_tile[FB_BLOCK_N];
  __shared__ float bias_tile[FB_BLOCK_N];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int threads_per_block = blockDim.x * blockDim.y;

  const int block_row = blockIdx.y * FB_BLOCK_M;
  const int block_col = blockIdx.x * FB_BLOCK_N;

  const int row_base = block_row + ty;
  const int col_base = block_col + tx;

  float acc[FB_THREAD_TILE_M][FB_THREAD_TILE_N];
#pragma unroll
  for (int i = 0; i < FB_THREAD_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < FB_THREAD_TILE_N; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  if (ty == 0) {
#pragma unroll
    for (int j = 0; j < FB_THREAD_TILE_N; ++j) {
      const int global_col = col_base + j * FB_THREADS_X;
      const int tile_col = tx + j * FB_THREADS_X;
      if (global_col < n) {
        scale_tile[tile_col] = scales[global_col];
        bias_tile[tile_col] = bias != nullptr ? load_bias<scalar_t>(bias, global_col) : 0.0f;
      } else {
        scale_tile[tile_col] = 0.0f;
        bias_tile[tile_col] = 0.0f;
      }
    }
  }
  __syncthreads();

  for (int k0 = 0; k0 < k; k0 += FB_BLOCK_K) {
    for (int idx = tid; idx < FB_BLOCK_M * FB_BLOCK_K; idx += threads_per_block) {
      const int tile_row = idx / FB_BLOCK_K;
      const int tile_k = idx % FB_BLOCK_K;
      const int global_row = block_row + tile_row;
      const int global_k = k0 + tile_k;
      if (global_row < m && global_k < k) {
        x_tile[tile_row][tile_k] = load_input<scalar_t>(x, global_row * k + global_k);
      } else {
        x_tile[tile_row][tile_k] = 0.0f;
      }
    }

    for (int idx = tid; idx < FB_BLOCK_N * FB_BLOCK_K; idx += threads_per_block) {
      const int tile_col = idx / FB_BLOCK_K;
      const int tile_k = idx % FB_BLOCK_K;
      const int global_col = block_col + tile_col;
      const int global_k = k0 + tile_k;
      if (global_col < n && global_k < k) {
        w_tile[tile_col][tile_k] = qweight[global_col * k + global_k];
      } else {
        w_tile[tile_col][tile_k] = 0;
      }
    }

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < FB_BLOCK_K; ++kk) {
      float a_frag[FB_THREAD_TILE_M];
      float b_frag[FB_THREAD_TILE_N];
#pragma unroll
      for (int i = 0; i < FB_THREAD_TILE_M; ++i) {
        a_frag[i] = x_tile[ty + i * FB_THREADS_Y][kk];
      }
#pragma unroll
      for (int j = 0; j < FB_THREAD_TILE_N; ++j) {
        const int tile_col = tx + j * FB_THREADS_X;
        b_frag[j] = static_cast<float>(w_tile[tile_col][kk]) * scale_tile[tile_col];
      }
#pragma unroll
      for (int i = 0; i < FB_THREAD_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < FB_THREAD_TILE_N; ++j) {
          acc[i][j] += a_frag[i] * b_frag[j];
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < FB_THREAD_TILE_M; ++i) {
    const int global_row = row_base + i * FB_THREADS_Y;
    if (global_row >= m) {
      continue;
    }
#pragma unroll
    for (int j = 0; j < FB_THREAD_TILE_N; ++j) {
      const int tile_col = tx + j * FB_THREADS_X;
      const int global_col = block_col + tile_col;
      if (global_col < n) {
        float value = acc[i][j] + bias_tile[tile_col];
        y[global_row * n + global_col] = cast_from_float<scalar_t>(value);
      }
    }
  }
}

__device__ __forceinline__ void load_wmma_stage_async(
    const at::Half* __restrict__ x,
    const int8_t* __restrict__ qweight,
    const float* __restrict__ scales,
    half* __restrict__ a_stage,
    int8_t* __restrict__ bq_stage,
    half* __restrict__ b_stage,
    int block_row,
    int block_col,
    int k0,
    int m,
    int n,
    int k,
    int tid,
    int threads_per_block) {
  const half* x_half = reinterpret_cast<const half*>(x);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int a_segments_per_row = WMMA_BLOCK_K / CP_ASYNC_A_HALFS;
  const int a_segments = WMMA_BLOCK_M * a_segments_per_row;
  for (int idx = tid; idx < a_segments; idx += threads_per_block) {
    const int tile_row = idx / a_segments_per_row;
    const int seg = idx % a_segments_per_row;
    const int tile_k = seg * CP_ASYNC_A_HALFS;
    const int global_row = block_row + tile_row;
    const int global_k = k0 + tile_k;
    half* smem_ptr = a_stage + tile_row * WMMA_A_STRIDE + tile_k;
    const half* gmem_ptr = x_half + global_row * k + global_k;
    if (global_row < m && global_k + CP_ASYNC_A_HALFS - 1 < k) {
      cp_async_16(smem_ptr, gmem_ptr);
    } else {
#pragma unroll
      for (int t = 0; t < CP_ASYNC_A_HALFS; ++t) {
        half value = __float2half(0.0f);
        if (global_row < m && global_k + t < k) {
          value = x_half[global_row * k + global_k + t];
        }
        smem_ptr[t] = value;
      }
    }
  }

  const int b_segments_per_row = WMMA_BLOCK_K / CP_ASYNC_BYTES;
  const int b_segments = WMMA_BLOCK_N * b_segments_per_row;
  for (int idx = tid; idx < b_segments; idx += threads_per_block) {
    const int tile_col = idx / b_segments_per_row;
    const int seg = idx % b_segments_per_row;
    const int tile_k = seg * CP_ASYNC_BYTES;
    const int global_col = block_col + tile_col;
    const int global_k = k0 + tile_k;
    int8_t* smem_ptr = bq_stage + tile_col * WMMA_BQ_STRIDE + tile_k;
    const int8_t* gmem_ptr = qweight + global_col * k + global_k;
    if (global_col < n && global_k + CP_ASYNC_BYTES - 1 < k) {
      cp_async_16(smem_ptr, gmem_ptr);
    } else {
#pragma unroll
      for (int t = 0; t < CP_ASYNC_BYTES; ++t) {
        int8_t value = 0;
        if (global_col < n && global_k + t < k) {
          value = qweight[global_col * k + global_k + t];
        }
        smem_ptr[t] = value;
      }
    }
  }
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();
#else
  const int a_half_elems = WMMA_BLOCK_M * WMMA_BLOCK_K;
  for (int idx = tid; idx < a_half_elems; idx += threads_per_block) {
    const int tile_row = idx / WMMA_BLOCK_K;
    const int tile_k = idx % WMMA_BLOCK_K;
    const int global_row = block_row + tile_row;
    const int global_k = k0 + tile_k;
    if (global_row < m && global_k < k) {
      a_stage[tile_row * WMMA_A_STRIDE + tile_k] = x_half[global_row * k + global_k];
    } else {
      a_stage[tile_row * WMMA_A_STRIDE + tile_k] = __float2half(0.0f);
    }
  }
  const int b_int8_elems = WMMA_BLOCK_N * WMMA_BLOCK_K;
  for (int idx = tid; idx < b_int8_elems; idx += threads_per_block) {
    const int tile_col = idx / WMMA_BLOCK_K;
    const int tile_k = idx % WMMA_BLOCK_K;
    const int global_col = block_col + tile_col;
    const int global_k = k0 + tile_k;
    if (global_col < n && global_k < k) {
      bq_stage[tile_col * WMMA_BQ_STRIDE + tile_k] = qweight[global_col * k + global_k];
    } else {
      bq_stage[tile_col * WMMA_BQ_STRIDE + tile_k] = 0;
    }
  }
  __syncthreads();
#endif

  const int b_vec_elems = (WMMA_BLOCK_N * WMMA_BLOCK_K) / 16;
  for (int idx = tid; idx < b_vec_elems; idx += threads_per_block) {
    const int tile_col = idx / (WMMA_BLOCK_K / 16);
    const int tile_k_vec = (idx % (WMMA_BLOCK_K / 16)) * 16;
    const int global_col = block_col + tile_col;
    half scale_h = __float2half(0.0f);
    if (global_col < n) {
      scale_h = __float2half(scales[global_col]);
    }
    const int8_t* vals = bq_stage + tile_col * WMMA_BQ_STRIDE + tile_k_vec;
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      b_stage[(tile_k_vec + t) * WMMA_B_STRIDE + tile_col] = __hmul(__int2half_rn(static_cast<int>(vals[t])), scale_h);
    }
  }
  __syncthreads();
}

__global__ void int8_weight_only_gemm_wmma_kernel(
    const at::Half* __restrict__ x,
    const int8_t* __restrict__ qweight,
    const float* __restrict__ scales,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ y,
    int m,
    int n,
    int k) {
  __shared__ half a_tiles[2][WMMA_BLOCK_M * WMMA_A_STRIDE];
  __shared__ int8_t bq_tiles[2][WMMA_BLOCK_N * WMMA_BQ_STRIDE];
  __shared__ half b_tiles[2][WMMA_BLOCK_K * WMMA_B_STRIDE];
  __shared__ float bias_tile[WMMA_BLOCK_N];
  __shared__ float c_tile[WMMA_BLOCK_M * WMMA_C_STRIDE];

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;

  const int block_row = blockIdx.y * WMMA_BLOCK_M;
  const int block_col = blockIdx.x * WMMA_BLOCK_N;

  if (tid < WMMA_BLOCK_N) {
    const int global_col = block_col + tid;
    bias_tile[tid] = (bias != nullptr && global_col < n) ? load_bias<at::Half>(bias, global_col) : 0.0f;
  }

  load_wmma_stage_async(
      x,
      qweight,
      scales,
      a_tiles[0],
      bq_tiles[0],
      b_tiles[0],
      block_row,
      block_col,
      0,
      m,
      n,
      k,
      tid,
      WMMA_THREADS_PER_BLOCK);

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  const int warp_row = warp_id / WMMA_BLOCK_COL_WARPS;
  const int warp_col = warp_id % WMMA_BLOCK_COL_WARPS;

  int stage = 0;
  for (int k0 = 0; k0 < k; k0 += WMMA_BLOCK_K) {
    const int next_k0 = k0 + WMMA_BLOCK_K;
    const int next_stage = stage ^ 1;
    if (next_k0 < k) {
      load_wmma_stage_async(
          x,
          qweight,
          scales,
          a_tiles[next_stage],
          bq_tiles[next_stage],
          b_tiles[next_stage],
          block_row,
          block_col,
          next_k0,
          m,
          n,
          k,
          tid,
          WMMA_THREADS_PER_BLOCK);
    }

    half* a_stage = a_tiles[stage];
    half* b_stage = b_tiles[stage];
#pragma unroll
    for (int kk = 0; kk < WMMA_BLOCK_K; kk += WMMA_K) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
      wmma::load_matrix_sync(a_frag, a_stage + warp_row * WMMA_M * WMMA_A_STRIDE + kk, WMMA_A_STRIDE);
      wmma::load_matrix_sync(b_frag, b_stage + kk * WMMA_B_STRIDE + warp_col * WMMA_N, WMMA_B_STRIDE);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __syncthreads();
    stage = next_stage;
  }

  float* warp_c_ptr = c_tile + warp_row * WMMA_M * WMMA_C_STRIDE + warp_col * WMMA_N;
  wmma::store_matrix_sync(warp_c_ptr, c_frag, WMMA_C_STRIDE, wmma::mem_row_major);
  __syncthreads();

  for (int idx = tid; idx < WMMA_BLOCK_M * WMMA_BLOCK_N; idx += WMMA_THREADS_PER_BLOCK) {
    const int tile_row = idx / WMMA_BLOCK_N;
    const int tile_col = idx % WMMA_BLOCK_N;
    const int global_row = block_row + tile_row;
    const int global_col = block_col + tile_col;
    if (global_row < m && global_col < n) {
      const float value = c_tile[tile_row * WMMA_C_STRIDE + tile_col] + bias_tile[tile_col];
      y[global_row * n + global_col] = cast_from_float<at::Half>(value);
    }
  }
}

bool can_use_wmma(torch::Tensor x, torch::Tensor qweight) {
  if (x.scalar_type() != at::kHalf) {
    return false;
  }
  if (x.size(1) < WMMA_K) {
    return false;
  }
  if (x.size(1) % 2 != 0) {
    return false;
  }
  if (qweight.size(1) % 16 != 0) {
    return false;
  }
  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  return device_prop->major >= 8;
}

}  // namespace

torch::Tensor int8_weight_only_gemm_cuda(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    c10::optional<torch::Tensor> bias) {
  auto y = torch::empty({x.size(0), qweight.size(0)}, x.options());
  const int m = static_cast<int>(x.size(0));
  const int n = static_cast<int>(qweight.size(0));
  const int k = static_cast<int>(x.size(1));
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  if (can_use_wmma(x, qweight)) {
    dim3 block(WMMA_THREADS_PER_BLOCK);
    dim3 grid((n + WMMA_BLOCK_N - 1) / WMMA_BLOCK_N, (m + WMMA_BLOCK_M - 1) / WMMA_BLOCK_M);
    const auto* x_ptr = x.data_ptr<at::Half>();
    const auto* qweight_ptr = qweight.data_ptr<int8_t>();
    const auto* scales_ptr = scales.data_ptr<float>();
    const auto* bias_ptr = bias.has_value() ? bias->data_ptr<at::Half>() : nullptr;
    auto* y_ptr = y.data_ptr<at::Half>();
    int8_weight_only_gemm_wmma_kernel<<<grid, block, 0, stream>>>(
        x_ptr,
        qweight_ptr,
        scales_ptr,
        bias_ptr,
        y_ptr,
        m,
        n,
        k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
  }

  dim3 block(FB_THREADS_X, FB_THREADS_Y);
  dim3 grid((n + FB_BLOCK_N - 1) / FB_BLOCK_N, (m + FB_BLOCK_M - 1) / FB_BLOCK_M);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "int8_weight_only_gemm_cuda_fallback", [&] {
    const auto* x_ptr = x.data_ptr<scalar_t>();
    const auto* qweight_ptr = qweight.data_ptr<int8_t>();
    const auto* scales_ptr = scales.data_ptr<float>();
    const auto* bias_ptr = bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr;
    auto* y_ptr = y.data_ptr<scalar_t>();
    int8_weight_only_gemm_fallback_kernel<scalar_t><<<grid, block, 0, stream>>>(
        x_ptr,
        qweight_ptr,
        scales_ptr,
        bias_ptr,
        y_ptr,
        m,
        n,
        k);
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
