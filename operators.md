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

GELU and HardSwish are fully implemented in [this branch](https://github.com/Mmadan128/SOFIE/tree/gpu/alpaka) with parser, ROperator class, CPU emitter and Alpaka GPU kernel. Code is already there so no point repeating it here.

For Mish, GroupNorm and SwiGLU I wrote pseudocode below showing how I am thinking about the implementation before coding it up.

---

## Mish

Each output only depends on its own input, no sharing between threads, so I just launch N workers and each one handles a single index independently. The only issue is `log(1 + e**x))` overflows for large x in float32, so I handle it in three ranges.

**CPU:**
```text
parallel for i in [0, N):
    x = X[i]
    if   x >  20: s = x + log(1 + exp(-x))   // avoid overflow
    elif x < -20: s = exp(x)                   // underflow region
    else:         s = log(1 + exp(x))
    Y[i] = x * tanh(s)
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
            float sp;
            if (x > 20.0f)
                sp = x + alpaka::math::log(acc, 1.0f + alpaka::math::exp(acc, -x));
            else if (x < -20.0f)
                sp = alpaka::math::exp(acc, x);
            else
                sp = alpaka::math::log(acc, 1.0f + alpaka::math::exp(acc, x));
            output[i] = x * alpaka::math::tanh(acc, sp);
        }
    }
};
```

---

## GroupNormalization

Unlike Mish/SwiGLU you need the mean and variance of the whole group before writing any output, so element level parallelism does not work here. I made the work unit one `(batch, group)` pair instead. Each task owns its slice fully so there is no overlap between workers.

CPU runs each `(n, g)` pair as an independent task. On GPU one block maps to one `(n, g)` pair and does the reduction internally using warp shuffles and shared memory.

**CPU:**
```text
parallel for task in [0, N*G):
    n = task / G
    g = task % G

    // pass 1: mean and variance
    mean = 0
    for c in [g*cPerG, (g+1)*cPerG), l in [0, L):
        mean += X[n, c, l]
    mean /= cPerG * L

    var = 0
    for c in [g*cPerG, (g+1)*cPerG), l in [0, L):
        var += (X[n, c, l] - mean)^2
    var /= cPerG * L
    invStd = 1 / sqrt(var + eps)

    // pass 2: normalize
    for c in [g*cPerG, (g+1)*cPerG), l in [0, L):
        Y[n, c, l] = (X[n, c, l] - mean) * invStd * gamma[c] + beta[c]
```

**GPU kernel:**
```cpp
// one block handles one (n, g) pair
// each thread accumulates locally, then warp shuffle to get block wide mean and var
// reduction code not written yet
struct GroupNormKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        float const* input, float* output,
        float const* gamma, float const* beta,
        int N, int C, int L, int G, float eps
    ) const {
        int blockId = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
        int n = blockId / G;
        int g = blockId % G;
        int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];

        int cPerG  = C / G;
        int gSize  = cPerG * L;
        int gStart = g * cPerG;

        float count = 0, mean = 0, M2 = 0;
        for (int idx = tid; idx < gSize; idx += 256) {
            int c = gStart + idx / L;
            int l = idx % L;
            float x = input[n * C * L + c * L + l];
            count += 1;
            float d = x - mean; mean += d / count;
            M2 += d * (x - mean);
        }

        // reduction will happen here

        for (int idx = tid; idx < gSize; idx += 256) {
            int c = gStart + idx / L;
            int l = idx % L;
            int flat = n * C * L + c * L + l;
            output[flat] = gamma[c] * (input[flat] - mu) * invStd + beta[c];
        }
    }
};
```

---

## SwiGLU

Input is split into gate and up, the op is elementwise across both so each index is independent, same as Mish. I kept it as one fused kernel so the sigmoid of gate never gets written to memory and feeds straight into the multiply with up.

**CPU:**
```text
parallel for i in [0, M):
    gate = A[i]
    up   = B[i]
    sig  = 1 / (1 + exp(-gate))
    Y[i] = up * (gate * sig)   // swish(gate) * up
```

**GPU kernel:**
```cpp
// sigmoid fused inline without temp buffer
struct SwiGLUKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        float const* gate, float const* up,
        float* output, size_t const M
    ) const {
        for (auto i : alpaka::uniformElements(acc, M)) {
            float g = gate[i];
            float u = up[i];
            float sig = 1.0f / (1.0f + alpaka::math::exp(acc, -g));
            output[i] = (g * sig) * u;
        }
    }
};
```