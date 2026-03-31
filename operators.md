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

For normal values, I use a compact Mish form that needs only one exp call. For very large positive values, I switch to a fast path to avoid float32 overflow. For very negative values, it naturally underflows to 0.0f, For very negative values, the output naturally goes to 0.0f, so no extra branching is needed.


Stable form I use in the middle range:
x * tanh(softplus(x)) = x * (e^(2x) + 2e^x) / (e^(2x) + 2e^x + 2)

**CPU:**
```text
#pragma omp parallel for schedule(static)
for (size_t i = 0; i < N; ++i) {
    float x = X[i];
    if (x > 20.0f) {
        Y[i] = x;
    } else {
        float e = exp(x);
        float n = e * (e + 2.0f);
        Y[i] = x * n / (n + 2.0f);
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
            
            if (x > 20.0f) { 
                output[i] = x; 
            } else {
                float e = alpaka::math::exp(acc, x);
                float n = e * (e + 2.0f);
                output[i] = x * n / (n + 2.0f);
            }
        }
    }
};
```

---

## GroupNormalization

On the CPU, a single thread can handle the sequential Welford algorithm. On the GPU, to ensure mathematical precision across parallel threads, I use a standard two pass block reduction .

**CPU:**
```text
#pragma omp parallel for schedule(static)
for (size_t task = 0; task < N * G; ++task) {
    int n = task / G;
    int g = task % G;
    int cPerG = C / G;
    int gStart = g * cPerG;

    float mean = 0;
    float M2 = 0;
    int count = 0;
    
    for (int c = gStart; c < gStart + cPerG; ++c) {
        for (int l = 0; l < L; ++l) {
            float x = X[n*C*L + c*L + l];
            float d = x - mean;
            mean += d / ++count;
            M2 += d * (x - mean);
        }
    }

    float invStd = 1.0f / sqrt(M2 / count + eps);
    
    for (int c = gStart; c < gStart + cPerG; ++c) {
        for (int l = 0; l < L; ++l) {
            int flat = n*C*L + c*L + l;
            Y[flat] = (X[flat] - mean) * invStd * gamma[c] + beta[c];
        }
    }
}
```

**GPU kernel:**
```cpp
// One block handles one (n, g) pair
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

        //pass1: compute mean
        float t_sum = 0;
        for (int idx = tid; idx < gSize; idx += nThreads) {
            t_sum += input[n*C*L + (gStart + idx/L)*L + idx%L];
        }
        smem[tid] = t_sum;
        alpaka::syncBlockThreads(acc);

        for (int s = nThreads / 2; s > 0; s >>= 1) {
            if (tid < s) smem[tid] += smem[tid + s];
            alpaka::syncBlockThreads(acc);
        }
        float mu = smem[0] / gSize;
        alpaka::syncBlockThreads(acc);

        // pass2:compute variance 
        float t_var = 0;
        for (int idx = tid; idx < gSize; idx += nThreads) {
            float d = input[n*C*L + (gStart + idx/L)*L + idx%L] - mu;
            t_var += d * d;
        }
        smem[tid] = t_var;
        alpaka::syncBlockThreads(acc);

        for (int s = nThreads / 2; s > 0; s >>= 1) {
            if (tid < s) smem[tid] += smem[tid + s];
            alpaka::syncBlockThreads(acc);
        }
        
        float invStd = alpaka::math::rsqrt(acc, smem[0] / gSize + eps);

        // normalize
        for (int idx = tid; idx < gSize; idx += nThreads) {
            int c = gStart + idx / L;
            int flat = n*C*L + c*L + idx%L;
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