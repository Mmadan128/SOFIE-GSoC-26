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

Same idea as GELU. Each output only depends on its own input so every element can be computed independently. The one thing I had to think about is that `log(1 + exp(x))` overflows for large x in float32, so I handle it in three ranges.

**CPU:**
```text
parallel for i in [0, N):
    x = X[i]
    if   x >  20: s = x + log(1 + exp(-x))
    elif x < -20: s = exp(x)
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

This one is trickier because you need the mean and variance of the whole group before you can write any output. So the work unit here is one `(batch, group)` pair, not one element. Each task handles its own slice from start to finish with no overlap.

On CPU each task runs independently. On GPU one block maps to one `(n, g)` pair and does the reduction internally using warp shuffles and shared memory.

**CPU:**
```text
parallel for task in [0, N*G):
    n = task / G
    g = task % G

    mean = sum of X over group slice / group_size
    var  = sum of (X - mean)^2 over group slice / group_size
    invStd = 1 / sqrt(var + eps)

    for each element in group slice:
        Y[n, c, l] = (X[n, c, l] - mean) * invStd * gamma[c] + beta[c]
```

**GPU kernel:**
```cpp
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

        // threads split the group slice between them and accumulate locally
        float count = 0, mean = 0, M2 = 0;
        for (int idx = tid; idx < gSize; idx += 256) {
            int c = gStart + idx / L;
            int l = idx % L;
            float x = input[n * C * L + c * L + l];
            count += 1;
            float d = x - mean; mean += d / count;
            M2 += d * (x - mean);
        }

        // warp shuffle + shared mem merge to get block wide mean and variance
        // ... reduction step ...

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

Input gets split into two halves, gate and up, and the op is elementwise across them. Pretty similar to Mish in structure. I keep it as a single fused kernel so the sigmoid of gate never gets written out to memory and just feeds straight into the multiply with up.

**CPU:**
```text
parallel for i in [0, N):
    a = A[i]
    b = B[i]
    sig = 1 / (1 + exp(-b))
    Y[i] = a * (b * sig)
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
            float g = gate[i];
            float u = up[i];
            float sig = 1.0f / (1.0f + alpaka::math::exp(acc, -g));
            output[i] = (g * sig) * u;
        }
    }
};
```