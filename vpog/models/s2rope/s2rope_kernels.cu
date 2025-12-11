// src/models/s2rope/s2rope_kernels.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

__global__ void s2rope_kernel(
    float* __restrict__ tokens,   // [B,N,H,D]
    const float* __restrict__ phase_x,   // [B,N,Px]
    const float* __restrict__ phase_y,   // [B,N,Py]
    const float* __restrict__ phase_sph, // [B,N,F,Ps]
    int B, int N, int H, int D,
    int Px, int Py, int F, int Ps,
    float fwd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * H;
    if (idx >= total) return;

    int h = idx % H;
    int tmp = idx / H;
    int n = tmp % N;
    int b = tmp / N;

    int baseTok = ((b * N + n) * H + h) * D;

    int offset = 0;

    // X block
    if (Px > 0 && phase_x != nullptr) {
        int basePhase = (b * N + n) * Px;
        for (int p = 0; p < Px; ++p) {
            float angle = fwd * phase_x[basePhase + p];
            float c = __cosf(angle);
            float s = __sinf(angle);

            int idx_u = baseTok + offset + 2 * p;
            int idx_v = idx_u + 1;

            float u = tokens[idx_u];
            float v = tokens[idx_v];

            tokens[idx_u] = u * c - v * s;
            tokens[idx_v] = v * c + u * s;
        }
        offset += 2 * Px;
    }

    // Y block
    if (Py > 0 && phase_y != nullptr) {
        int basePhase = (b * N + n) * Py;
        for (int p = 0; p < Py; ++p) {
            float angle = fwd * phase_y[basePhase + p];
            float c = __cosf(angle);
            float s = __sinf(angle);

            int idx_u = baseTok + offset + 2 * p;
            int idx_v = idx_u + 1;

            float u = tokens[idx_u];
            float v = tokens[idx_v];

            tokens[idx_u] = u * c - v * s;
            tokens[idx_v] = v * c + u * s;
        }
        offset += 2 * Py;
    }

    // S² block
    if (F > 0 && Ps > 0 && phase_sph != nullptr) {
        int strideFaces = F * Ps;
        int basePhase = (b * N + n) * strideFaces;

        for (int f = 0; f < F; ++f) {
            int faceBase = basePhase + f * Ps;
            for (int p = 0; p < Ps; ++p) {
                float angle = fwd * phase_sph[faceBase + p];
                float c = __cosf(angle);
                float s = __sinf(angle);

                int idx_u = baseTok + offset + 2 * p;
                int idx_v = idx_u + 1;

                float u = tokens[idx_u];
                float v = tokens[idx_v];

                tokens[idx_u] = u * c - v * s;
                tokens[idx_v] = v * c + u * s;
            }
            offset += 2 * Ps;
        }
    }
}

} // namespace

void s2rope_cuda(
    torch::Tensor tokens,
    const torch::Tensor phase_x,
    const torch::Tensor phase_y,
    const torch::Tensor phase_sph,
    const float fwd
) {
    int B = tokens.size(0);
    int N = tokens.size(1);
    int H = tokens.size(2);
    int D = tokens.size(3);

    int Px = (phase_x.defined() && phase_x.numel() > 0) ? phase_x.size(2) : 0;
    int Py = (phase_y.defined() && phase_y.numel() > 0) ? phase_y.size(2) : 0;
    int F  = (phase_sph.defined() && phase_sph.numel() > 0) ? phase_sph.size(2) : 0;
    int Ps = (phase_sph.defined() && phase_sph.numel() > 0) ? phase_sph.size(3) : 0;

    const float* px_ptr  = (Px > 0 ? phase_x.data_ptr<float>()   : nullptr);
    const float* py_ptr  = (Py > 0 ? phase_y.data_ptr<float>()   : nullptr);
    const float* sph_ptr = (F>0 && Ps>0 ? phase_sph.data_ptr<float>() : nullptr);

    int total = B * N * H;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    s2rope_kernel<<<blocks, threads>>>(
        tokens.data_ptr<float>(),
        px_ptr,
        py_ptr,
        sph_ptr,
        B, N, H, D,
        Px, Py, F, Ps,
        fwd
    );
    cudaDeviceSynchronize();
}
