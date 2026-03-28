---
title: "GPU Operator Pseudocode"
layout: single
permalink: /operators/
author_profile: false
sidebar:
  nav: "docs"
toc: true
toc_sticky: true
toc_label: "Operators"
---

This page documents the GPU parallelism strategy for the four operators I plan to implement during GSoC 2026. Each section covers: the core algorithm, a diagram of how work is distributed across GPU threads, and alpaka-flavoured pseudocode.

All kernels target the [alpaka](https://github.com/alpaka-group/alpaka) abstraction layer — the same source compiles to CUDA (NVIDIA) and HIP (AMD) by changing one compiler flag.

---

## 1. GELU

### Algorithm

GELU (Gaussian Error Linear Unit) is used in models like Particle Transformer and GPT. It applies an element-wise nonlinearity:

$$\text{GELU}(x) = x \cdot \frac{1}{2}\left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\,(x + 0.044715\,x^3)\right)\right)$$

### Parallelism Strategy

**Pattern: Embarrassingly parallel.** Every output element depends only on its corresponding input element — no thread needs to communicate with any other.

<figure>
<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;display:block;margin:1.2em auto;font-family:monospace;">
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0,8 3,0 6" fill="#7ec8e3"/>
    </marker>
  </defs>
  <!-- Input row -->
  <text x="20" y="38" fill="#aaa" font-size="12">input[N]</text>
  <rect x="20"  y="48" width="48" height="28" rx="4" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/>
  <text x="44"  y="67" fill="#7ec8e3" font-size="11" text-anchor="middle">x₀</text>
  <rect x="76"  y="48" width="48" height="28" rx="4" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/>
  <text x="100" y="67" fill="#7ec8e3" font-size="11" text-anchor="middle">x₁</text>
  <rect x="132" y="48" width="48" height="28" rx="4" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/>
  <text x="156" y="67" fill="#7ec8e3" font-size="11" text-anchor="middle">x₂</text>
  <rect x="188" y="48" width="48" height="28" rx="4" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/>
  <text x="212" y="67" fill="#7ec8e3" font-size="11" text-anchor="middle">x₃</text>
  <text x="246" y="67" fill="#555" font-size="14">···</text>
  <!-- Arrows down -->
  <line x1="44"  y1="76" x2="44"  y2="108" stroke="#7ec8e3" stroke-width="1.2" marker-end="url(#arr)"/>
  <line x1="100" y1="76" x2="100" y2="108" stroke="#7ec8e3" stroke-width="1.2" marker-end="url(#arr)"/>
  <line x1="156" y1="76" x2="156" y2="108" stroke="#7ec8e3" stroke-width="1.2" marker-end="url(#arr)"/>
  <line x1="212" y1="76" x2="212" y2="108" stroke="#7ec8e3" stroke-width="1.2" marker-end="url(#arr)"/>
  <!-- Thread boxes -->
  <text x="20" y="104" fill="#aaa" font-size="11">threads</text>
  <rect x="20"  y="110" width="48" height="28" rx="4" fill="#2a1f3d" stroke="#c084fc" stroke-width="1.2"/>
  <text x="44"  y="129" fill="#c084fc" font-size="10" text-anchor="middle">T₀</text>
  <rect x="76"  y="110" width="48" height="28" rx="4" fill="#2a1f3d" stroke="#c084fc" stroke-width="1.2"/>
  <text x="100" y="129" fill="#c084fc" font-size="10" text-anchor="middle">T₁</text>
  <rect x="132" y="110" width="48" height="28" rx="4" fill="#2a1f3d" stroke="#c084fc" stroke-width="1.2"/>
  <text x="156" y="129" fill="#c084fc" font-size="10" text-anchor="middle">T₂</text>
  <rect x="188" y="110" width="48" height="28" rx="4" fill="#2a1f3d" stroke="#c084fc" stroke-width="1.2"/>
  <text x="212" y="129" fill="#c084fc" font-size="10" text-anchor="middle">T₃</text>
  <text x="246" y="129" fill="#555" font-size="14">···</text>
  <!-- Arrows down to output -->
  <line x1="44"  y1="138" x2="44"  y2="162" stroke="#7ec8e3" stroke-width="1.2" marker-end="url(#arr)"/>
  <line x1="100" y1="138" x2="100" y2="162" stroke="#7ec8e3" stroke-width="1.2" marker-end="url(#arr)"/>
  <line x1="156" y1="138" x2="156" y2="162" stroke="#7ec8e3" stroke-width="1.2" marker-end="url(#arr)"/>
  <line x1="212" y1="138" x2="212" y2="162" stroke="#7ec8e3" stroke-width="1.2" marker-end="url(#arr)"/>
  <!-- Output row -->
  <text x="20" y="158" fill="#aaa" font-size="11">output[N]</text>
  <rect x="20"  y="164" width="48" height="28" rx="4" fill="#1a3a2a" stroke="#4ade80" stroke-width="1.2"/>
  <text x="44"  y="183" fill="#4ade80" font-size="11" text-anchor="middle">y₀</text>
  <rect x="76"  y="164" width="48" height="28" rx="4" fill="#1a3a2a" stroke="#4ade80" stroke-width="1.2"/>
  <text x="100" y="183" fill="#4ade80" font-size="11" text-anchor="middle">y₁</text>
  <rect x="132" y="164" width="48" height="28" rx="4" fill="#1a3a2a" stroke="#4ade80" stroke-width="1.2"/>
  <text x="156" y="183" fill="#4ade80" font-size="11" text-anchor="middle">y₂</text>
  <rect x="188" y="164" width="48" height="28" rx="4" fill="#1a3a2a" stroke="#4ade80" stroke-width="1.2"/>
  <text x="212" y="183" fill="#4ade80" font-size="11" text-anchor="middle">y₃</text>
  <text x="246" y="183" fill="#555" font-size="14">···</text>
  <!-- Label -->
  <text x="360" y="90" fill="#888" font-size="11">1 thread → 1 element</text>
  <text x="360" y="108" fill="#888" font-size="11">no shared memory</text>
  <text x="360" y="126" fill="#888" font-size="11">no synchronisation</text>
  <text x="360" y="144" fill="#888" font-size="11">grid = ⌈N / 256⌉ blocks</text>
</svg>
<figcaption>Each thread reads one input element, computes GELU independently, and writes one output. Grid size scales with N.</figcaption>
</figure>

### Pseudocode

```cpp
struct GELUKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        float const* input,   // [N] flattened
        float*       output,  // [N] flattened
        size_t const N
    ) const {
        // Each thread gets a unique global index
        size_t const i =
            alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];

        if (i >= N) return;   // guard: tail threads do nothing

        float const x = input[i];

        // Tanh approximation (matches PyTorch default GELU)
        constexpr float kAlpha = 0.7978845608f;  // sqrt(2/π)
        constexpr float kBeta  = 0.044715f;

        float const inner = kAlpha * (x + kBeta * x * x * x);
        output[i] = x * 0.5f * (1.0f + alpaka::math::tanh(acc, inner));
        // ↑ hardware tanh: fast intrinsic on both CUDA and HIP
    }
};

// Launch: one thread per element, 256 threads per block
void launch_gelu(float const* d_in, float* d_out, size_t N) {
    constexpr size_t kBlock = 256;
    auto workDiv = WorkDiv1D{ (N + kBlock - 1) / kBlock, 1, kBlock };
    alpaka::exec<TAcc>(queue, workDiv, GELUKernel{}, d_in, d_out, N);
}
```

> **Why this works on GPU:** GELU is memory-bandwidth-bound, not compute-bound. The key is *coalesced access* — consecutive threads read and write consecutive memory addresses, so the hardware combines them into one wide transaction instead of N separate ones. No shared memory or synchronisation needed.

---

## 2. ReduceSum / ReduceMean

### Algorithm

Reduces a tensor over a given axis by summing (or averaging) all elements along that axis. Input `[batch, C, L]` reduced over `L` → output `[batch, C, 1]`.

### Parallelism Strategy

**Pattern: Parallel tree reduction.** A sequential sum over L elements takes O(L) steps. A parallel tree reduction takes O(log L) steps using all threads in a block. The key insight is using *reversed-stride addressing* to avoid shared memory bank conflicts.

<figure>
<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;display:block;margin:1.2em auto;font-family:monospace;">
  <defs>
    <marker id="arr2" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0,8 3,0 6" fill="#f59e0b"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="20" y="22" fill="#aaa" font-size="12">One block reduces one row of L=8 elements</text>

  <!-- Row 0: input in shared memory -->
  <text x="20" y="50" fill="#aaa" font-size="11">sdata[ ]  (shared memory)</text>
  <rect x="20"  y="56" width="44" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/><text x="42"  y="74" fill="#7ec8e3" font-size="11" text-anchor="middle">2</text>
  <rect x="70"  y="56" width="44" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/><text x="92"  y="74" fill="#7ec8e3" font-size="11" text-anchor="middle">5</text>
  <rect x="120" y="56" width="44" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/><text x="142" y="74" fill="#7ec8e3" font-size="11" text-anchor="middle">1</text>
  <rect x="170" y="56" width="44" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/><text x="192" y="74" fill="#7ec8e3" font-size="11" text-anchor="middle">3</text>
  <rect x="220" y="56" width="44" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/><text x="242" y="74" fill="#7ec8e3" font-size="11" text-anchor="middle">4</text>
  <rect x="270" y="56" width="44" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/><text x="292" y="74" fill="#7ec8e3" font-size="11" text-anchor="middle">2</text>
  <rect x="320" y="56" width="44" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/><text x="342" y="74" fill="#7ec8e3" font-size="11" text-anchor="middle">6</text>
  <rect x="370" y="56" width="44" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/><text x="392" y="74" fill="#7ec8e3" font-size="11" text-anchor="middle">1</text>
  <text x="460" y="74" fill="#888" font-size="11">stride = 4</text>

  <!-- Stride 4 arrows: 0←4, 1←5, 2←6, 3←7 -->
  <path d="M242,56 Q242,42 42,56" stroke="#f59e0b" stroke-width="1.4" fill="none" marker-end="url(#arr2)"/>
  <path d="M292,56 Q292,40 92,56" stroke="#f59e0b" stroke-width="1.4" fill="none" marker-end="url(#arr2)"/>
  <path d="M342,56 Q342,38 142,56" stroke="#f59e0b" stroke-width="1.4" fill="none" marker-end="url(#arr2)"/>
  <path d="M392,56 Q392,36 192,56" stroke="#f59e0b" stroke-width="1.4" fill="none" marker-end="url(#arr2)"/>

  <!-- Row 1 after stride 4 -->
  <text x="20" y="116" fill="#aaa" font-size="11">after stride=4  (syncthreads)</text>
  <rect x="20"  y="122" width="44" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/><text x="42"  y="140" fill="#7ec8e3" font-size="11" text-anchor="middle">6</text>
  <rect x="70"  y="122" width="44" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/><text x="92"  y="140" fill="#7ec8e3" font-size="11" text-anchor="middle">7</text>
  <rect x="120" y="122" width="44" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/><text x="142" y="140" fill="#7ec8e3" font-size="11" text-anchor="middle">7</text>
  <rect x="170" y="122" width="44" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/><text x="192" y="140" fill="#7ec8e3" font-size="11" text-anchor="middle">4</text>
  <rect x="220" y="122" width="44" height="26" rx="3" fill="#292929" stroke="#444" stroke-width="1"/><text x="242" y="140" fill="#555" font-size="11" text-anchor="middle">·</text>
  <rect x="270" y="122" width="44" height="26" rx="3" fill="#292929" stroke="#444" stroke-width="1"/><text x="292" y="140" fill="#555" font-size="11" text-anchor="middle">·</text>
  <rect x="320" y="122" width="44" height="26" rx="3" fill="#292929" stroke="#444" stroke-width="1"/><text x="342" y="140" fill="#555" font-size="11" text-anchor="middle">·</text>
  <rect x="370" y="122" width="44" height="26" rx="3" fill="#292929" stroke="#444" stroke-width="1"/><text x="392" y="140" fill="#555" font-size="11" text-anchor="middle">·</text>
  <text x="460" y="140" fill="#888" font-size="11">stride = 2</text>

  <!-- Stride 2 arrows -->
  <path d="M142,122 Q142,110 42,122" stroke="#f59e0b" stroke-width="1.4" fill="none" marker-end="url(#arr2)"/>
  <path d="M192,122 Q192,108 92,122" stroke="#f59e0b" stroke-width="1.4" fill="none" marker-end="url(#arr2)"/>

  <!-- Row 2 after stride 2 -->
  <text x="20" y="182" fill="#aaa" font-size="11">after stride=2  (syncthreads)</text>
  <rect x="20"  y="188" width="44" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/><text x="42"  y="206" fill="#7ec8e3" font-size="11" text-anchor="middle">13</text>
  <rect x="70"  y="188" width="44" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/><text x="92"  y="206" fill="#7ec8e3" font-size="11" text-anchor="middle">11</text>
  <rect x="120" y="188" width="44" height="26" rx="3" fill="#292929" stroke="#444" stroke-width="1"/><text x="142" y="206" fill="#555" font-size="11" text-anchor="middle">·</text>
  <rect x="170" y="188" width="44" height="26" rx="3" fill="#292929" stroke="#444" stroke-width="1"/><text x="192" y="206" fill="#555" font-size="11" text-anchor="middle">·</text>
  <text x="460" y="206" fill="#888" font-size="11">stride = 1</text>

  <!-- Stride 1 arrow -->
  <path d="M92,188 Q92,178 42,188" stroke="#f59e0b" stroke-width="1.4" fill="none" marker-end="url(#arr2)"/>

  <!-- Final result -->
  <text x="20" y="248" fill="#aaa" font-size="11">sdata[0] = 24  ✓  (thread 0 writes output)</text>
  <rect x="20" y="230" width="44" height="26" rx="3" fill="#1a3a2a" stroke="#4ade80" stroke-width="1.8"/><text x="42" y="248" fill="#4ade80" font-size="12" font-weight="bold" text-anchor="middle">24</text>
</svg>
<figcaption>Tree reduction over 8 elements in 3 steps (log₂8). Arrows show which thread adds into which slot. Greyed slots are idle after each step. Final sum lands in sdata[0].</figcaption>
</figure>

### Pseudocode

```cpp
struct ReduceSumKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        float const* input,       // [total_rows, L]
        float*       output,      // [total_rows]
        size_t const L,
        size_t const total_rows,
        bool   const compute_mean
    ) const {
        constexpr size_t kBlockSize = 256;
        // Shared memory: one float per thread in the block
        auto& sdata = alpaka::declareSharedVar<float[kBlockSize]>(acc);

        size_t const tid      = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
        size_t const row      = alpaka::getIdx<alpaka::Grid,  alpaka::Blocks >(acc)[0u];
        size_t const blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];

        if (row >= total_rows) return;

        // Phase 1: grid-stride loop — handles L > blockDim without extra kernels
        // tid=0 covers elements 0, blockDim, 2*blockDim, …
        // tid=1 covers elements 1, blockDim+1, …  (all threads stay busy)
        float partial = 0.0f;
        for (size_t i = tid; i < L; i += blockDim)
            partial += input[row * L + i];
        sdata[tid] = partial;
        alpaka::syncBlockThreads(acc);   // all partials must land before reduction

        // Phase 2: bank-conflict-free tree reduction
        // stride = blockDim/2, blockDim/4, …, 1
        // Active threads always access consecutive sdata slots → no bank conflicts
        for (size_t stride = blockDim / 2; stride > 0; stride >>= 1) {
            if (tid < stride)
                sdata[tid] += sdata[tid + stride];
            alpaka::syncBlockThreads(acc);
        }

        // Phase 3: one thread writes the result
        if (tid == 0) {
            float result = sdata[0];
            if (compute_mean) result /= static_cast<float>(L);
            output[row] = result;
        }
    }
};
```

> **Why reversed-stride?** The naive approach doubles the stride each step (1, 2, 4, …). This causes *shared memory bank conflicts* — multiple threads land on the same memory bank and serialize. Reversing (starting at blockDim/2 and halving) keeps active threads at consecutive indices, so all accesses hit different banks simultaneously.

---

## 3. LayerNorm (Welford Algorithm)

### Algorithm

Layer Normalisation normalises each token vector independently:

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} \cdot \gamma + \beta$$

where μ and σ² are computed over the last dimension D (e.g. d_model).

### Parallelism Strategy

**Pattern: Welford single-pass + two-tier warp/block reduction.** The naïve approach reads the row twice — once for mean, once for variance. Welford's algorithm computes both in a single pass, halving memory traffic. Warp-level shuffles (`shfl_down`) then merge results within each warp using registers only, before a final shared-memory merge across warps — avoiding atomic operations entirely.

<figure>
<svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;display:block;margin:1.2em auto;font-family:monospace;">
  <defs>
    <marker id="arr3" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
      <polygon points="0 0,7 2.5,0 5" fill="#c084fc"/>
    </marker>
    <marker id="arr4" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
      <polygon points="0 0,7 2.5,0 5" fill="#f59e0b"/>
    </marker>
  </defs>

  <!-- Block label -->
  <text x="20" y="20" fill="#aaa" font-size="12">One block = one token row (D elements split across threads)</text>

  <!-- Phase 1 -->
  <rect x="14" y="30" width="610" height="52" rx="5" fill="none" stroke="#333" stroke-dasharray="4"/>
  <text x="22" y="44" fill="#888" font-size="10">PHASE 1 — each thread runs Welford update over its slice (grid-stride)</text>
  <rect x="22"  y="50" width="70" height="26" rx="3" fill="#2a1f3d" stroke="#c084fc" stroke-width="1.2"/>
  <text x="57"  y="68" fill="#c084fc" font-size="10" text-anchor="middle">T0 (μ,M2)</text>
  <rect x="100" y="50" width="70" height="26" rx="3" fill="#2a1f3d" stroke="#c084fc" stroke-width="1.2"/>
  <text x="135" y="68" fill="#c084fc" font-size="10" text-anchor="middle">T1 (μ,M2)</text>
  <rect x="178" y="50" width="70" height="26" rx="3" fill="#2a1f3d" stroke="#c084fc" stroke-width="1.2"/>
  <text x="213" y="68" fill="#c084fc" font-size="10" text-anchor="middle">T2 (μ,M2)</text>
  <text x="256" y="68" fill="#555" font-size="13">···</text>
  <rect x="280" y="50" width="80" height="26" rx="3" fill="#2a1f3d" stroke="#c084fc" stroke-width="1.2"/>
  <text x="320" y="68" fill="#c084fc" font-size="10" text-anchor="middle">T31 (μ,M2)</text>
  <text x="370" y="65" fill="#888" font-size="10">← warp 0 (32 threads)</text>
  <text x="370" y="78" fill="#555" font-size="10">warp 1, 2, … also run</text>

  <!-- Phase 2a -->
  <rect x="14" y="92" width="610" height="52" rx="5" fill="none" stroke="#333" stroke-dasharray="4"/>
  <text x="22" y="106" fill="#888" font-size="10">PHASE 2a — warp-level Welford merge via shfl_down (register shuffles, no shared mem)</text>
  <!-- Merge arrows within warp -->
  <rect x="22"  y="112" width="70" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/>
  <text x="57"  y="130" fill="#7ec8e3" font-size="10" text-anchor="middle">warp0 (μ,σ²)</text>
  <path d="M92,125 L100,125" stroke="#c084fc" stroke-width="1.4" marker-end="url(#arr3)"/>
  <path d="M170,125 L178,125" stroke="#c084fc" stroke-width="1.4" marker-end="url(#arr3)"/>
  <path d="M248,125 L256,125" stroke="#c084fc" stroke-width="1.4" marker-end="url(#arr3)"/>
  <text x="100" y="130" fill="#555" font-size="10">→ lane 0 holds merged state</text>
  <text x="370" y="126" fill="#888" font-size="10">5 shuffle steps: offset 16,8,4,2,1</text>
  <text x="370" y="139" fill="#888" font-size="10">pure register ops, zero latency</text>

  <!-- Phase 2b -->
  <rect x="14" y="154" width="610" height="52" rx="5" fill="none" stroke="#333" stroke-dasharray="4"/>
  <text x="22" y="168" fill="#888" font-size="10">PHASE 2b — lane 0 of each warp deposits into shared mem; first warp merges</text>
  <rect x="22"  y="174" width="50" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/>
  <text x="47"  y="192" fill="#7ec8e3" font-size="10" text-anchor="middle">w0</text>
  <rect x="80"  y="174" width="50" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/>
  <text x="105" y="192" fill="#7ec8e3" font-size="10" text-anchor="middle">w1</text>
  <rect x="138" y="174" width="50" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/>
  <text x="163" y="192" fill="#7ec8e3" font-size="10" text-anchor="middle">w2</text>
  <text x="195" y="192" fill="#555" font-size="12">···</text>
  <rect x="214" y="174" width="50" height="26" rx="3" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/>
  <text x="239" y="192" fill="#7ec8e3" font-size="10" text-anchor="middle">w7</text>
  <text x="280" y="185" fill="#888" font-size="10">← 8 warp results in shared mem</text>
  <text x="280" y="198" fill="#888" font-size="10">first warp reads these 8 → shfl_down again → μ, σ²</text>

  <!-- Phase 3 -->
  <rect x="14" y="216" width="610" height="52" rx="5" fill="none" stroke="#333" stroke-dasharray="4"/>
  <text x="22" y="230" fill="#888" font-size="10">PHASE 3 — normalise + affine transform (all threads, grid-stride)</text>
  <text x="22" y="248" fill="#4ade80" font-size="11">y[i] = γ[i] · (x[i] − μ) · rsqrt(σ² + ε)  +  β[i]</text>
  <text x="370" y="244" fill="#888" font-size="10">one global read of μ, σ² (shared mem)</text>
  <text x="370" y="257" fill="#888" font-size="10">rsqrt: hardware fast-path</text>
</svg>
<figcaption>Three-phase LayerNorm. Welford accumulation (Phase 1) replaces two sequential passes. Warp shuffles (Phase 2a) merge within a warp with zero shared memory. Only 8 floats touch shared memory in Phase 2b.</figcaption>
</figure>

### Pseudocode

```cpp
// Warp-level Welford merge using register shuffles (no shared memory)
template <typename TAcc>
ALPAKA_FN_ACC void warp_welford_reduce(
    TAcc const& acc, float& count, float& mean, float& M2)
{
    // 5 steps: offset 16 → 8 → 4 → 2 → 1
    // shfl_down: thread i reads the register value of thread i+offset
    for (int offset = 16; offset > 0; offset >>= 1) {
        float cb = alpaka::warp::shfl_down(acc, count, offset);
        float mb = alpaka::warp::shfl_down(acc, mean,  offset);
        float m2b= alpaka::warp::shfl_down(acc, M2,    offset);

        // Welford parallel merge formula
        float total = count + cb;
        float delta = mb - mean;
        mean  = (count * mean + cb * mb) / (total + 1e-30f);
        M2   += m2b + delta * delta * count * cb / (total + 1e-30f);
        count = total;
    }
    // Lane 0 now holds the warp-wide (count, mean, M2)
}

struct LayerNormKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        float const* input, float* output,
        float const* gamma, float const* beta,
        size_t const D, size_t const num_rows, float const eps
    ) const {
        constexpr size_t kBlock   = 256;
        constexpr size_t kWarps   = kBlock / 32;  // = 8

        auto& s_count = alpaka::declareSharedVar<float[kWarps]>(acc);
        auto& s_mean  = alpaka::declareSharedVar<float[kWarps]>(acc);
        auto& s_M2    = alpaka::declareSharedVar<float[kWarps]>(acc);

        size_t const tid    = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
        size_t const row    = alpaka::getIdx<alpaka::Grid,  alpaka::Blocks >(acc)[0u];
        size_t const lane   = tid % 32;
        size_t const warp   = tid / 32;

        if (row >= num_rows) return;
        float const* xrow = input  + row * D;
        float*       yrow = output + row * D;

        // Phase 1: Welford online update — single pass computes mean AND variance
        // update rule: count++; δ = x−μ; μ += δ/count; M2 += δ*(x−μ_new)
        // Numerically stable: avoids catastrophic cancellation in Σx²−nμ²
        float count = 0, mean = 0, M2 = 0;
        for (size_t i = tid; i < D; i += kBlock) {
            float x = xrow[i];  count += 1;
            float d1 = x - mean;    mean += d1 / count;
            M2 += d1 * (x - mean);  // uses updated mean → stable
        }

        // Phase 2a: warp-level merge (register shuffles — fastest possible)
        warp_welford_reduce(acc, count, mean, M2);
        if (lane == 0) { s_count[warp]=count; s_mean[warp]=mean; s_M2[warp]=M2; }
        alpaka::syncBlockThreads(acc);

        // Phase 2b: first warp merges the 8 warp-results
        if (warp == 0) {
            count = (lane < kWarps) ? s_count[lane] : 0;
            mean  = (lane < kWarps) ? s_mean [lane] : 0;
            M2    = (lane < kWarps) ? s_M2   [lane] : 0;
            warp_welford_reduce(acc, count, mean, M2);
        }

        auto& s_mu  = alpaka::declareSharedVar<float>(acc);
        auto& s_inv = alpaka::declareSharedVar<float>(acc);
        if (tid == 0) {
            s_mu  = mean;
            s_inv = alpaka::math::rsqrt(acc, M2 / float(D) + eps);
        }
        alpaka::syncBlockThreads(acc);

        // Phase 3: normalise + affine transform
        float const mu = s_mu, inv = s_inv;
        for (size_t i = tid; i < D; i += kBlock)
            yrow[i] = gamma[i] * (xrow[i] - mu) * inv + beta[i];
    }
};
```

> **Why Welford?** The textbook two-pass approach reads the row twice (once for μ, once for σ²) — doubling memory bandwidth. Welford computes both in one pass. It is also numerically stable: the formula `Σx² − nμ²` catastrophically cancels when values are large and close together. Welford avoids this entirely.

---

## 4. Conv1D via Im2col + GEMM

### Algorithm

Convolution over a 1-D sequence with kernel size K, stride s, dilation d, padding p:

$$\text{output}[n, c_{out}, l] = \sum_{c_{in}}\sum_{k} W[c_{out}, c_{in}, k] \cdot x[n, c_{in},\; l \cdot s + k \cdot d - p]$$

### Parallelism Strategy

**Pattern: Im2col restructuring → reuse existing tiled GEMM.** Rather than writing a new tiled kernel that handles stride, dilation, padding, and channel loops simultaneously, Im2col *unrolls* the kernel dimension into matrix columns — converting the convolution into a dense matrix multiply that reuses SOFIE's already-optimised GPU GEMM.

<figure>
<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;display:block;margin:1.2em auto;font-family:monospace;">
  <defs>
    <marker id="arr5" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0,8 3,0 6" fill="#4ade80"/>
    </marker>
  </defs>

  <!-- Input tensor -->
  <text x="20" y="18" fill="#aaa" font-size="11">input  [N, Cin, L]</text>
  <rect x="20" y="24" width="110" height="60" rx="4" fill="#1e3a4a" stroke="#7ec8e3" stroke-width="1.2"/>
  <text x="75" y="52" fill="#7ec8e3" font-size="10" text-anchor="middle">N × Cin × L</text>
  <text x="75" y="68" fill="#555" font-size="9" text-anchor="middle">e.g. 32 × 16 × 128</text>

  <!-- Im2col arrow -->
  <line x1="136" y1="54" x2="176" y2="54" stroke="#4ade80" stroke-width="1.5" marker-end="url(#arr5)"/>
  <text x="138" y="48" fill="#4ade80" font-size="9">Im2col</text>
  <text x="138" y="72" fill="#888" font-size="8">kernel</text>

  <!-- col_buffer -->
  <text x="180" y="18" fill="#aaa" font-size="11">col_buffer  [N, Cin·K, L_out]</text>
  <rect x="180" y="24" width="130" height="60" rx="4" fill="#2a1f1a" stroke="#f59e0b" stroke-width="1.2"/>
  <text x="245" y="46" fill="#f59e0b" font-size="10" text-anchor="middle">N × (Cin·K) × L_out</text>
  <text x="245" y="62" fill="#888" font-size="9" text-anchor="middle">each col = one kernel</text>
  <text x="245" y="74" fill="#888" font-size="9" text-anchor="middle">application window</text>

  <!-- Weight -->
  <text x="20" y="116" fill="#aaa" font-size="11">weight  [Cout, Cin·K]</text>
  <rect x="20" y="122" width="110" height="44" rx="4" fill="#1a2a1a" stroke="#4ade80" stroke-width="1.2"/>
  <text x="75" y="148" fill="#4ade80" font-size="10" text-anchor="middle">Cout × (Cin·K)</text>

  <!-- GEMM arrow -->
  <line x1="136" y1="144" x2="176" y2="144" stroke="#c084fc" stroke-width="1.5" marker-end="url(#arr3)"/>
  <text x="138" y="138" fill="#c084fc" font-size="9">GEMM</text>
  <text x="138" y="158" fill="#888" font-size="8">(reuse</text>
  <text x="138" y="168" fill="#888" font-size="8">existing)</text>

  <!-- Output -->
  <text x="180" y="116" fill="#aaa" font-size="11">output  [N, Cout, L_out]</text>
  <rect x="180" y="122" width="130" height="44" rx="4" fill="#1a3a2a" stroke="#4ade80" stroke-width="1.8"/>
  <text x="245" y="148" fill="#4ade80" font-size="10" text-anchor="middle">N × Cout × L_out</text>

  <!-- Im2col detail box -->
  <rect x="14" y="188" width="610" height="62" rx="5" fill="none" stroke="#333" stroke-dasharray="4"/>
  <text x="22" y="204" fill="#888" font-size="10">Im2col kernel — one thread per col_buffer element (embarrassingly parallel)</text>
  <text x="22" y="220" fill="#f59e0b" font-size="10">flat_id → (n, cin, k, l_out)   →   l_in = l_out·stride + k·dilation − pad</text>
  <text x="22" y="236" fill="#f59e0b" font-size="10">col_buf[n, cin·K+k, l_out] = (l_in valid) ? input[n, cin, l_in] : 0.0f</text>
  <text x="22" y="248" fill="#888" font-size="10">zero-padding handled implicitly — no conditional branches in the hot path</text>
</svg>
<figcaption>Im2col reshapes the input so that each convolution window becomes a column. The result is a standard dense matrix multiply — reusing SOFIE's existing GPU GEMM kernel.</figcaption>
</figure>

### Pseudocode

```cpp
// Step 1: Im2col kernel — restructures input into col_buffer
// One thread per col_buffer element → embarrassingly parallel
struct Im2ColKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        float const* input,    // [N, Cin, L]
        float*       col_buf,  // [N, Cin*K, L_out]  ← filled by this kernel
        int N, int Cin, int L, int K, int L_out,
        int stride, int dilation, int pad
    ) const {
        int const CK  = Cin * K;
        int const gid = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
        if (gid >= N * CK * L_out) return;

        // Recover (n, ck, l_out) from the flat index
        int const l_out = gid % L_out;
        int const ck    = (gid / L_out) % CK;
        int const n     = gid / (L_out * CK);

        int const cin   = ck / K;          // which input channel
        int const k     = ck % K;          // position within kernel

        // Compute the source input position — handles stride and dilation
        int const l_in  = l_out * stride + k * dilation - pad;

        // Implicit zero-padding: out-of-bounds → write 0
        float val = (l_in >= 0 && l_in < L)
                    ? input[n * Cin * L + cin * L + l_in]
                    : 0.0f;

        col_buf[n * CK * L_out + ck * L_out + l_out] = val;
    }
};

// Step 2: host orchestration — Im2col then GEMM
void launch_conv1d(float const* d_in, float const* d_weight,
                   float const* d_bias, float* d_out,
                   int N, int Cin, int Cout, int L, int K,
                   int stride, int dilation, int pad)
{
    int const L_out = (L + 2*pad - dilation*(K-1) - 1) / stride + 1;
    int const CK    = Cin * K;

    // Allocate col_buffer on device
    float* d_col = alpaka::allocBuf<float>(dev, N * CK * L_out);

    // Launch Im2col — one thread per output element
    int const total = N * CK * L_out;
    auto workDiv = WorkDiv1D{ (total + 255) / 256, 1, 256 };
    alpaka::exec<TAcc>(queue, workDiv, Im2ColKernel{},
                       d_in, d_col, N, Cin, L, K, L_out,
                       stride, dilation, pad);

    // GEMM per batch element — reuses SOFIE's existing GPU MatMul
    // weight [Cout, CK]  ×  col[n] [CK, L_out]  →  out[n] [Cout, L_out]
    for (int n = 0; n < N; ++n)
        launch_matmul(d_weight,
                      d_col  + n * CK    * L_out,
                      d_out  + n * Cout  * L_out,
                      Cout, CK, L_out);

    // Optional bias broadcast-add
    if (d_bias) launch_add_bias(d_out, d_bias, N, Cout, L_out);

    alpaka::freeBuf(dev, d_col);
}
```

> **Why Im2col instead of a direct kernel?** A direct Conv1D kernel must tile over Cin, K, stride, and dilation simultaneously — roughly 200 lines that are hard to make efficient. Im2col separates the problem: a simple 30-line restructuring kernel (embarrassingly parallel, easy to verify) feeds a heavily-optimised GEMM that already exists in SOFIE. The only cost is the col_buffer allocation, which is acceptable for HEP model sizes.
