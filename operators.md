---
title: "GPU Operator Pseudocode"
layout: single
permalink: /operators/
author_profile: false
sidebar:
  nav: "docs"
toc: true
toc_sticky: true
toc_label: "On this page"
---

I have already implemented GELU and HardSwish in [this branch](https://github.com/Mmadan128/SOFIE/tree/gpu/alpaka) with parser, ROperator class, CPU emitter, and Alpaka GPU kernel.

Here is a brief overview of what im thinking about the implemenetation of the following operators::
1. GELU and HardSwish are elementwise, so one index per thread works directly.
2. Mish is also elementwise, so I keep the same mapping.
3. GroupNorm is different because outputs need group stats first, so I split work by batch and group.
4. SwiGLU is elementwise on two inputs, so one index per thread works here too.

For Mish, GroupNorm and SwiGLU I wrote pseudocode below showing how I am thinking about the implementation before writing its code.

---

## Mish

Mish is elementwise, so I split by index and run all elements together.

For normal values I use a fast middle formula with one exp call. For very large values I use simple edge cases so the math stays stable.

When x is very positive, Mish is close to x. When x is very negative, Mish is close to 0.

Stable form I use in the middle range:
x * tanh(softplus(x)) = x * (e^(2x) + 2e^x) / (e^(2x) + 2e^x + 2)

**CPU:**
```text
#pragma omp parallel for schedule(static)
for (size_t i = 0; i < N; ++i) {
    x = X[i]
    if (x > 20) {
        Y[i] = x
    } else if (x < -10) {
        Y[i] = 0
    } else {
        e = exp(x)
        n = e * (e + 2)
        Y[i] = x * n / (n + 2)
    }
}
```

**GPU kernel:**
```cpp
struct MishKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        float const* input, float* output, size_t const N
    ) const {
        for (auto i : alpaka::uniformElements(acc, N)) {
            float x = input[i];
            if (x >  20.0f) { output[i] = x;    continue; }
            if (x < -10.0f) { output[i] = 0.0f; continue; }
            float e = alpaka::math::expf(acc, x);
            float n = e * (e + 2.0f);
            output[i] = x * n / (n + 2.0f);
        }
    }
};
```

---

## GroupNormalization

GroupNorm is not pure elementwise because each element depends on group mean and variance. I split work by (batch, group) pairs, then each pair computes stats and normalizes its own slice.


**CPU:**
```text
#pragma omp parallel for schedule(static)
for (size_t task = 0; task < N * G; ++task) {
    n = task / G
    g = task % G
    cPerG = C / G
    gStart = g * cPerG

    mean = 0
    M2 = 0
    count = 0
    for c in [gStart, gStart + cPerG), l in [0, L):
        x = X[n, c, l]
        d = x - mean
        mean += d / ++count
        M2 += d * (x - mean)

    invStd = 1 / sqrt(M2 / count + eps)
    for c in [gStart, gStart + cPerG), l in [0, L):
        Y[n, c, l] = (X[n, c, l] - mean) * invStd * gamma[c] + beta[c]
}
```

**GPU kernel:**
```cpp
// one block handles one (n, g) pair
struct GroupNormKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        float const* input, float* output,
        float const* gamma, float const* beta,
        int N, int C, int L, int G, float eps
    ) const {
        auto* smem   = alpaka::getDynSharedMem<float>(acc);
        int blockId  = alpaka::getIdx<alpaka::Grid,  alpaka::Blocks>(acc)[0u];
        int tid      = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
        int nThreads = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];

        int n = blockId / G,  g = blockId % G;
        int cPerG = C / G,  gSize = cPerG * L,  gStart = g * cPerG;

        // each thread runs Welford on its chunk
        float t_mean = 0, t_M2 = 0, t_count = 0;
        for (int idx = tid; idx < gSize; idx += nThreads) {
            float x = input[n*C*L + (gStart + idx/L)*L + idx%L];
            float d = x - t_mean;
            t_mean += d / ++t_count;
            t_M2   += d * (x - t_mean);
        }
        smem[tid]            = t_mean * t_count;
        smem[nThreads + tid] = t_M2;
        alpaka::syncBlockThreads(acc);

        // tree reduction
        for (int s = nThreads / 2; s > 0; s >>= 1) {
            if (tid < s) {
                smem[tid]            += smem[tid + s];
                smem[nThreads + tid] += smem[nThreads + tid + s];
            }
            alpaka::syncBlockThreads(acc);
        }

        float mu     = smem[0] / gSize;
        float invStd = alpaka::math::rsqrt(acc, smem[nThreads] / gSize + eps);

        for (int idx = tid; idx < gSize; idx += nThreads) {
            int c = gStart + idx / L,  flat = n*C*L + c*L + idx%L;
            output[flat] = gamma[c] * (input[flat] - mu) * invStd + beta[c];
        }
    }
};
```

---

## SwiGLU

SwiGLU is elementwise across gate and up. Each output index is independent, so I run all indices at once.

**CPU:**
```text
#pragma omp parallel for simd schedule(static)
for (size_t i = 0; i < M; ++i) {
    g = gate[i]
    sig = 1.0 / (1.0 + exp(-g))
    Y[i] = up[i] * (g * sig)
}
```

**GPU kernel:**
```cpp
struct SwiGLUKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        float const* gate, float const* up,
        float* output, size_t const M
    ) const {
        for (auto i : alpaka::uniformElements(acc, M)) {
            float g   = gate[i];
            float sig = 1.0f / (1.0f + alpaka::math::expf(acc, -g));
            output[i] = (g * sig) * up[i];
        }
    }
};
```