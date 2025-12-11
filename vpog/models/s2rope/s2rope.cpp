// src/models/s2rope/s2rope.cpp

#include <torch/extension.h>
#include <cmath>

// Declaration of CUDA function implemented in s2rope_kernels.cu
void s2rope_cuda(
    torch::Tensor tokens,
    const torch::Tensor phase_x,
    const torch::Tensor phase_y,
    const torch::Tensor phase_sph,
    const float fwd
);

// CPU implementation
void s2rope_cpu(
    torch::Tensor tokens,       // [B,N,H,D]
    const torch::Tensor phase_x,
    const torch::Tensor phase_y,
    const torch::Tensor phase_sph,
    const float fwd
) {
    TORCH_CHECK(tokens.dim() == 4, "tokens must have shape [B,N,H,D]");

    const int B = tokens.size(0);
    const int N = tokens.size(1);
    const int H = tokens.size(2);
    const int D = tokens.size(3);

    TORCH_CHECK(tokens.scalar_type() == torch::kFloat,
                "tokens must be float32");

    auto tok = tokens.accessor<float, 4>();

    int Px = 0, Py = 0, F = 0, Ps = 0;

    if (phase_x.defined() && phase_x.numel() > 0) {
        TORCH_CHECK(phase_x.dim() == 3, "phase_x must have shape [B,N,Px]");
        TORCH_CHECK(phase_x.size(0) == B && phase_x.size(1) == N,
                    "phase_x B,N must match tokens");
        TORCH_CHECK(phase_x.scalar_type() == torch::kFloat,
                    "phase_x must be float32");
        Px = phase_x.size(2);
    }

    if (phase_y.defined() && phase_y.numel() > 0) {
        TORCH_CHECK(phase_y.dim() == 3, "phase_y must have shape [B,N,Py]");
        TORCH_CHECK(phase_y.size(0) == B && phase_y.size(1) == N,
                    "phase_y B,N must match tokens");
        TORCH_CHECK(phase_y.scalar_type() == torch::kFloat,
                    "phase_y must be float32");
        Py = phase_y.size(2);
    }

    if (phase_sph.defined() && phase_sph.numel() > 0) {
        TORCH_CHECK(phase_sph.dim() == 4, "phase_sph must have shape [B,N,F,Ps]");
        TORCH_CHECK(phase_sph.size(0) == B && phase_sph.size(1) == N,
                    "phase_sph B,N must match tokens");
        TORCH_CHECK(phase_sph.scalar_type() == torch::kFloat,
                    "phase_sph must be float32");
        F  = phase_sph.size(2);
        Ps = phase_sph.size(3);
    }

    const int D_expected = 2 * (Px + Py + F * Ps);
    TORCH_CHECK(D == D_expected,
                "tokens.D (", D, ") does not match 2*(Px+Py+F*Ps) = ", D_expected);

    for (int b = 0; b < B; ++b) {
        for (int n = 0; n < N; ++n) {
            int offset = 0;

            // x-block
            if (Px > 0 && phase_x.numel() > 0) {
                auto phase_x_acc = phase_x.accessor<float,3>();
                for (int h = 0; h < H; ++h) {
                    for (int p = 0; p < Px; ++p) {
                        const float angle = fwd * phase_x_acc[b][n][p];
                        const float c = std::cos(angle);
                        const float s = std::sin(angle);

                        const int idx_u = offset + 2 * p;
                        const int idx_v = idx_u + 1;

                        float u = tok[b][n][h][idx_u];
                        float v = tok[b][n][h][idx_v];

                        tok[b][n][h][idx_u] = u * c - v * s;
                        tok[b][n][h][idx_v] = v * c + u * s;
                    }
                }
                offset += 2 * Px;
            }

            // y-block
            if (Py > 0 && phase_y.numel() > 0) {
                auto phase_y_acc = phase_y.accessor<float,3>();
                for (int h = 0; h < H; ++h) {
                    for (int p = 0; p < Py; ++p) {
                        const float angle = fwd * phase_y_acc[b][n][p];
                        const float c = std::cos(angle);
                        const float s = std::sin(angle);

                        const int idx_u = offset + 2 * p;
                        const int idx_v = idx_u + 1;

                        float u = tok[b][n][h][idx_u];
                        float v = tok[b][n][h][idx_v];

                        tok[b][n][h][idx_u] = u * c - v * s;
                        tok[b][n][h][idx_v] = v * c + u * s;
                    }
                }
                offset += 2 * Py;
            }

            // S² faces block
            if (F > 0 && Ps > 0 && phase_sph.numel() > 0) {
                auto phase_sph_acc = phase_sph.accessor<float,4>();
                for (int f = 0; f < F; ++f) {
                    for (int h = 0; h < H; ++h) {
                        for (int p = 0; p < Ps; ++p) {
                            const float angle = fwd * phase_sph_acc[b][n][f][p];
                            const float c = std::cos(angle);
                            const float s = std::sin(angle);

                            const int idx_u = offset + 2 * p;
                            const int idx_v = idx_u + 1;

                            float u = tok[b][n][h][idx_u];
                            float v = tok[b][n][h][idx_v];

                            tok[b][n][h][idx_u] = u * c - v * s;
                            tok[b][n][h][idx_v] = v * c + u * s;
                        }
                    }
                    offset += 2 * Ps;
                }
            }
        }
    }
}

void s2rope(
    torch::Tensor tokens,
    const torch::Tensor phase_x,
    const torch::Tensor phase_y,
    const torch::Tensor phase_sph,
    const float fwd
) {
    TORCH_CHECK(
    (!phase_x.defined()) || (phase_x.is_cuda() == tokens.is_cuda()),
    "tokens and phase_x must be on same device (or phase_x undefined)"
    );


    if (tokens.is_cuda()) {
        s2rope_cuda(tokens, phase_x, phase_y, phase_sph, fwd);
    } else {
        s2rope_cpu(tokens, phase_x, phase_y, phase_sph, fwd);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("s2rope", &s2rope, "S2RoPE (in-place)");
}
