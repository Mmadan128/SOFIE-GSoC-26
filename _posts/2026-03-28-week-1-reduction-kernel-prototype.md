---
title: "Week 1 - Reduction Kernel Prototype"
layout: single
author_profile: false
summary: "Defined staged reduction flow, drafted kernel structure, and set correctness checks against CPU baseline."
---

## Planned

- Define decomposition strategy for Group B reduction operators.
- Prepare one portable implementation path using alpaka.
- Establish correctness checks against CPU baseline.

## Completed

- Drafted first kernel flow for staged reduction.
- Mapped worker responsibilities for partial and final merge passes.
- Documented edge cases for shape handling and axis reduction.

## Evidence

- Initial pseudo-code and task notes tracked in the code samples repository.
- Validation checklist prepared for parity testing.

## Code Notes

```cpp
// Prototype structure for staged reduction execution.
ALPAKA_FN_ACC void operator()(Acc const& acc) const {
  const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];

  // Stage 1: local accumulation
  float local = 0.0f;
  for (std::size_t i = idx; i < n; i += stride) {
    local += input[i];
  }

  // Stage 2: block-level reduction (placeholder)
  // TODO: replace with optimized shared-memory reduction path.
  output[idx] = local;
}
```

## Risks or Blockers

- Backend-specific behavior may affect reduction determinism.
- Memory access patterns are not tuned yet.

## Next Actions

- Implement one executable operator path end-to-end.
- Run parity checks with reference CPU implementation.
- Record baseline timings before optimization.
