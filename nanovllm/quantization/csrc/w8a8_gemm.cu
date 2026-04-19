#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <string>
#include <torch/extension.h>

namespace {

using namespace nvcuda;

constexpr int FB_BLOCK_M = 64;
constexpr int FB_BLOCK_N = 64;
constexpr int FB_BLOCK_K = 32;
constexpr int FB_THREADS_X = 16;
constexpr int FB_THREADS_Y = 16;
constexpr int FB_THREAD_TILE_M = FB_BLOCK_M / FB_THREADS_Y;
constexpr int FB_THREAD_TILE_N = FB_BLOCK_N / FB_THREADS_X;
constexpr int FB_X_SKEW = 8;
constexpr int FB_W_SKEW = 16;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WMMA_BLOCK_ROW_WARPS = 2;
constexpr int WMMA_BLOCK_COL_WARPS = 4;
constexpr int WMMA_WARPS_PER_BLOCK = WMMA_BLOCK_ROW_WARPS * WMMA_BLOCK_COL_WARPS;
constexpr int WMMA_THREADS_PER_BLOCK = WMMA_WARPS_PER_BLOCK * 32;
constexpr int WMMA_BLOCK_M = WMMA_BLOCK_ROW_WARPS * WMMA_M;
constexpr int WMMA_BLOCK_N = WMMA_BLOCK_COL_WARPS * WMMA_N;
constexpr int WMMA_BLOCK_K = 32;
constexpr int WMMA_A_SKEW = 16;
constexpr int WMMA_B_SKEW = 16;
constexpr int WMMA_C_SKEW = 8;
constexpr int WMMA_A_STRIDE = WMMA_BLOCK_K + WMMA_A_SKEW;
constexpr int WMMA_B_STRIDE = WMMA_BLOCK_K + WMMA_B_SKEW;
constexpr int WMMA_C_STRIDE = WMMA_BLOCK_N + WMMA_C_SKEW;
constexpr int CP_ASYNC_BYTES = 16;

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
__global__ void w8a8_gemm_fallback_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ a_scales,
    const int8_t* __restrict__ qweight,
    const float* __restrict__ w_scales,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ y,
    int m,
    int n,
    int k) {
  __shared__ int8_t x_tile[FB_BLOCK_M][FB_BLOCK_K + FB_X_SKEW];
  __shared__ int8_t w_tile[FB_BLOCK_N][FB_BLOCK_K + FB_W_SKEW];
  __shared__ float a_scale_tile[FB_BLOCK_M];
  __shared__ float w_scale_tile[FB_BLOCK_N];
  __shared__ float bias_tile[FB_BLOCK_N];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int threads_per_block = blockDim.x * blockDim.y;
  const int block_row = blockIdx.y * FB_BLOCK_M;
  const int block_col = blockIdx.x * FB_BLOCK_N;
  const int row_base = block_row + ty;
  const int col_base = block_col + tx;

  int acc[FB_THREAD_TILE_M][FB_THREAD_TILE_N];
#pragma unroll
  for (int i = 0; i < FB_THREAD_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < FB_THREAD_TILE_N; ++j) {
      acc[i][j] = 0;
    }
  }

  if (tx == 0) {
#pragma unroll
    for (int i = 0; i < FB_THREAD_TILE_M; ++i) {
      const int global_row = row_base + i * FB_THREADS_Y;
      const int tile_row = ty + i * FB_THREADS_Y;
      a_scale_tile[tile_row] = global_row < m ? a_scales[global_row] : 0.0f;
    }
  }
  if (ty == 0) {
#pragma unroll
    for (int j = 0; j < FB_THREAD_TILE_N; ++j) {
      const int global_col = col_base + j * FB_THREADS_X;
      const int tile_col = tx + j * FB_THREADS_X;
      if (global_col < n) {
        w_scale_tile[tile_col] = w_scales[global_col];
        bias_tile[tile_col] = bias != nullptr ? load_bias<scalar_t>(bias, global_col) : 0.0f;
      } else {
        w_scale_tile[tile_col] = 0.0f;
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
      x_tile[tile_row][tile_k] = (global_row < m && global_k < k) ? x_q[global_row * k + global_k] : 0;
    }
    for (int idx = tid; idx < FB_BLOCK_N * FB_BLOCK_K; idx += threads_per_block) {
      const int tile_col = idx / FB_BLOCK_K;
      const int tile_k = idx % FB_BLOCK_K;
      const int global_col = block_col + tile_col;
      const int global_k = k0 + tile_k;
      w_tile[tile_col][tile_k] = (global_col < n && global_k < k) ? qweight[global_col * k + global_k] : 0;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < FB_BLOCK_K; ++kk) {
      int a_frag[FB_THREAD_TILE_M];
      int b_frag[FB_THREAD_TILE_N];
#pragma unroll
      for (int i = 0; i < FB_THREAD_TILE_M; ++i) {
        a_frag[i] = static_cast<int>(x_tile[ty + i * FB_THREADS_Y][kk]);
      }
#pragma unroll
      for (int j = 0; j < FB_THREAD_TILE_N; ++j) {
        b_frag[j] = static_cast<int>(w_tile[tx + j * FB_THREADS_X][kk]);
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
    const int tile_row = ty + i * FB_THREADS_Y;
    if (global_row >= m) {
      continue;
    }
#pragma unroll
    for (int j = 0; j < FB_THREAD_TILE_N; ++j) {
      const int tile_col = tx + j * FB_THREADS_X;
      const int global_col = block_col + tile_col;
      if (global_col < n) {
        float value = static_cast<float>(acc[i][j]) * a_scale_tile[tile_row] * w_scale_tile[tile_col] + bias_tile[tile_col];
        y[global_row * n + global_col] = cast_from_float<scalar_t>(value);
      }
    }
  }
}

__device__ __forceinline__ void load_w8a8_stage_async(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ qweight,
    int8_t* __restrict__ a_stage,
    int8_t* __restrict__ b_stage,
    int block_row,
    int block_col,
    int k0,
    int m,
    int n,
    int k,
    int tid,
    int threads_per_block) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int a_segments_per_row = WMMA_BLOCK_K / CP_ASYNC_BYTES;
  const int a_segments = WMMA_BLOCK_M * a_segments_per_row;
  for (int idx = tid; idx < a_segments; idx += threads_per_block) {
    const int tile_row = idx / a_segments_per_row;
    const int seg = idx % a_segments_per_row;
    const int tile_k = seg * CP_ASYNC_BYTES;
    const int global_row = block_row + tile_row;
    const int global_k = k0 + tile_k;
    int8_t* smem_ptr = a_stage + tile_row * WMMA_A_STRIDE + tile_k;
    const int8_t* gmem_ptr = x_q + global_row * k + global_k;
    if (global_row < m && global_k + CP_ASYNC_BYTES - 1 < k) {
      cp_async_16(smem_ptr, gmem_ptr);
    } else {
#pragma unroll
      for (int t = 0; t < CP_ASYNC_BYTES; ++t) {
        smem_ptr[t] = (global_row < m && global_k + t < k) ? x_q[global_row * k + global_k + t] : 0;
      }
    }
  }

  const int b_segments_per_col = WMMA_BLOCK_K / CP_ASYNC_BYTES;
  const int b_segments = WMMA_BLOCK_N * b_segments_per_col;
  for (int idx = tid; idx < b_segments; idx += threads_per_block) {
    const int tile_col = idx / b_segments_per_col;
    const int seg = idx % b_segments_per_col;
    const int tile_k = seg * CP_ASYNC_BYTES;
    const int global_col = block_col + tile_col;
    const int global_k = k0 + tile_k;
    int8_t* smem_ptr = b_stage + tile_col * WMMA_B_STRIDE + tile_k;
    const int8_t* gmem_ptr = qweight + global_col * k + global_k;
    if (global_col < n && global_k + CP_ASYNC_BYTES - 1 < k) {
      cp_async_16(smem_ptr, gmem_ptr);
    } else {
#pragma unroll
      for (int t = 0; t < CP_ASYNC_BYTES; ++t) {
        smem_ptr[t] = (global_col < n && global_k + t < k) ? qweight[global_col * k + global_k + t] : 0;
      }
    }
  }
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();
#else
  const int a_elems = WMMA_BLOCK_M * WMMA_BLOCK_K;
  for (int idx = tid; idx < a_elems; idx += threads_per_block) {
    const int tile_row = idx / WMMA_BLOCK_K;
    const int tile_k = idx % WMMA_BLOCK_K;
    const int global_row = block_row + tile_row;
    const int global_k = k0 + tile_k;
    a_stage[tile_row * WMMA_A_STRIDE + tile_k] = (global_row < m && global_k < k) ? x_q[global_row * k + global_k] : 0;
  }
  const int b_elems = WMMA_BLOCK_N * WMMA_BLOCK_K;
  for (int idx = tid; idx < b_elems; idx += threads_per_block) {
    const int tile_col = idx / WMMA_BLOCK_K;
    const int tile_k = idx % WMMA_BLOCK_K;
    const int global_col = block_col + tile_col;
    const int global_k = k0 + tile_k;
    b_stage[tile_col * WMMA_B_STRIDE + tile_k] = (global_col < n && global_k < k) ? qweight[global_col * k + global_k] : 0;
  }
  __syncthreads();
#endif
}

template <typename scalar_t>
__global__ void w8a8_gemm_wmma_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ a_scales,
    const int8_t* __restrict__ qweight,
    const float* __restrict__ w_scales,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ y,
    int m,
    int n,
    int k) {
  __shared__ int8_t a_tiles[2][WMMA_BLOCK_M * WMMA_A_STRIDE];
  __shared__ int8_t b_tiles[2][WMMA_BLOCK_N * WMMA_B_STRIDE];
  __shared__ float a_scale_tile[WMMA_BLOCK_M];
  __shared__ float w_scale_tile[WMMA_BLOCK_N];
  __shared__ float bias_tile[WMMA_BLOCK_N];
  __shared__ int c_tile[WMMA_BLOCK_M * WMMA_C_STRIDE];

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int block_row = blockIdx.y * WMMA_BLOCK_M;
  const int block_col = blockIdx.x * WMMA_BLOCK_N;

  if (tid < WMMA_BLOCK_M) {
    const int global_row = block_row + tid;
    a_scale_tile[tid] = global_row < m ? a_scales[global_row] : 0.0f;
  }
  if (tid < WMMA_BLOCK_N) {
    const int global_col = block_col + tid;
    if (global_col < n) {
      w_scale_tile[tid] = w_scales[global_col];
      bias_tile[tid] = bias != nullptr ? load_bias<scalar_t>(bias, global_col) : 0.0f;
    } else {
      w_scale_tile[tid] = 0.0f;
      bias_tile[tid] = 0.0f;
    }
  }

  load_w8a8_stage_async(
      x_q,
      qweight,
      a_tiles[0],
      b_tiles[0],
      block_row,
      block_col,
      0,
      m,
      n,
      k,
      tid,
      WMMA_THREADS_PER_BLOCK);

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;
  wmma::fill_fragment(c_frag, 0);

  const int warp_row = warp_id / WMMA_BLOCK_COL_WARPS;
  const int warp_col = warp_id % WMMA_BLOCK_COL_WARPS;

  int stage = 0;
  for (int k0 = 0; k0 < k; k0 += WMMA_BLOCK_K) {
    const int next_k0 = k0 + WMMA_BLOCK_K;
    const int next_stage = stage ^ 1;
    if (next_k0 < k) {
      load_w8a8_stage_async(
          x_q,
          qweight,
          a_tiles[next_stage],
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

    const int8_t* a_stage = a_tiles[stage];
    const int8_t* b_stage = b_tiles[stage];
#pragma unroll
    for (int kk = 0; kk < WMMA_BLOCK_K; kk += WMMA_K) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::col_major> b_frag;
      wmma::load_matrix_sync(a_frag, a_stage + warp_row * WMMA_M * WMMA_A_STRIDE + kk, WMMA_A_STRIDE);
      wmma::load_matrix_sync(b_frag, b_stage + warp_col * WMMA_N * WMMA_B_STRIDE + kk, WMMA_B_STRIDE);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __syncthreads();
    stage = next_stage;
  }

  int* warp_c_ptr = c_tile + warp_row * WMMA_M * WMMA_C_STRIDE + warp_col * WMMA_N;
  wmma::store_matrix_sync(warp_c_ptr, c_frag, WMMA_C_STRIDE, wmma::mem_row_major);
  __syncthreads();

  for (int idx = tid; idx < WMMA_BLOCK_M * WMMA_BLOCK_N; idx += WMMA_THREADS_PER_BLOCK) {
    const int tile_row = idx / WMMA_BLOCK_N;
    const int tile_col = idx % WMMA_BLOCK_N;
    const int global_row = block_row + tile_row;
    const int global_col = block_col + tile_col;
    if (global_row < m && global_col < n) {
      const float value =
          static_cast<float>(c_tile[tile_row * WMMA_C_STRIDE + tile_col]) * a_scale_tile[tile_row] * w_scale_tile[tile_col]
          + bias_tile[tile_col];
      y[global_row * n + global_col] = cast_from_float<scalar_t>(value);
    }
  }
}

bool can_use_wmma(torch::Tensor x_q, torch::Tensor qweight, bool use_bfloat16) {
  if (x_q.scalar_type() != at::kChar || qweight.scalar_type() != at::kChar) {
    return false;
  }
  if (x_q.size(1) < WMMA_K || x_q.size(1) % WMMA_K != 0) {
    return false;
  }
  if (qweight.size(1) % WMMA_K != 0) {
    return false;
  }
  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  if (use_bfloat16) {
    return device_prop->major >= 8;
  }
  return device_prop->major >= 8;
}

std::string select_w8a8_gemm_path(torch::Tensor x_q, torch::Tensor qweight) {
  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  if (device_prop->major >= 8 && x_q.size(1) % WMMA_K == 0 && qweight.size(1) % WMMA_K == 0) {
    return "wmma_int8_cp_async";
  }
  return "fallback_cuda";
}

}  // namespace

std::string w8a8_gemm_path_cuda(torch::Tensor x_q, torch::Tensor qweight) {
  return select_w8a8_gemm_path(x_q, qweight);
}

torch::Tensor w8a8_gemm_cuda(
    torch::Tensor x_q,
    torch::Tensor a_scales,
    torch::Tensor qweight,
    torch::Tensor w_scales,
    c10::optional<torch::Tensor> bias,
    bool use_bfloat16) {
  auto out_dtype = use_bfloat16 ? at::kBFloat16 : at::kHalf;
  auto y = torch::empty({x_q.size(0), qweight.size(0)}, x_q.options().dtype(out_dtype));
  const int m = static_cast<int>(x_q.size(0));
  const int n = static_cast<int>(qweight.size(0));
  const int k = static_cast<int>(qweight.size(1));
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  if (can_use_wmma(x_q, qweight, use_bfloat16)) {
    dim3 block(WMMA_THREADS_PER_BLOCK);
    dim3 grid((n + WMMA_BLOCK_N - 1) / WMMA_BLOCK_N, (m + WMMA_BLOCK_M - 1) / WMMA_BLOCK_M);
    const auto* x_ptr = x_q.data_ptr<int8_t>();
    const auto* a_scales_ptr = a_scales.data_ptr<float>();
    const auto* qweight_ptr = qweight.data_ptr<int8_t>();
    const auto* w_scales_ptr = w_scales.data_ptr<float>();
    if (use_bfloat16) {
      const auto* bias_ptr = bias.has_value() ? bias->data_ptr<at::BFloat16>() : nullptr;
      auto* y_ptr = y.data_ptr<at::BFloat16>();
      w8a8_gemm_wmma_kernel<at::BFloat16><<<grid, block, 0, stream>>>(
          x_ptr,
          a_scales_ptr,
          qweight_ptr,
          w_scales_ptr,
          bias_ptr,
          y_ptr,
          m,
          n,
          k);
    } else {
      const auto* bias_ptr = bias.has_value() ? bias->data_ptr<at::Half>() : nullptr;
      auto* y_ptr = y.data_ptr<at::Half>();
      w8a8_gemm_wmma_kernel<at::Half><<<grid, block, 0, stream>>>(
          x_ptr,
          a_scales_ptr,
          qweight_ptr,
          w_scales_ptr,
          bias_ptr,
          y_ptr,
          m,
          n,
          k);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
  }

  dim3 block(FB_THREADS_X, FB_THREADS_Y);
  dim3 grid((n + FB_BLOCK_N - 1) / FB_BLOCK_N, (m + FB_BLOCK_M - 1) / FB_BLOCK_M);
  if (use_bfloat16) {
    const auto* bias_ptr = bias.has_value() ? bias->data_ptr<at::BFloat16>() : nullptr;
    w8a8_gemm_fallback_kernel<at::BFloat16><<<grid, block, 0, stream>>>(
        x_q.data_ptr<int8_t>(),
        a_scales.data_ptr<float>(),
        qweight.data_ptr<int8_t>(),
        w_scales.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<at::BFloat16>(),
        m,
        n,
        k);
  } else {
    const auto* bias_ptr = bias.has_value() ? bias->data_ptr<at::Half>() : nullptr;
    w8a8_gemm_fallback_kernel<at::Half><<<grid, block, 0, stream>>>(
        x_q.data_ptr<int8_t>(),
        a_scales.data_ptr<float>(),
        qweight.data_ptr<int8_t>(),
        w_scales.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<at::Half>(),
        m,
        n,
        k);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
