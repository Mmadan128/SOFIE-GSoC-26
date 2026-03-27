---
layout: home
author_profile: true
sidebar:
  nav: "sidebar"
---

## SOFIE GSoC 2026 Engineering Logbook

This logbook documents weekly engineering progress for the CERN-HSF Google Summer of Code 2026 project focused on accelerating SOFIE inference kernels on heterogeneous hardware.

### Project Snapshot

| Field | Value |
| --- | --- |
| Program | Google Summer of Code 2026 |
| Organization | CERN-HSF |
| Project | SOFIE Parallel Kernel Development |
| Focus Area | Portable GPU backend using alpaka |
| Contributor | Mohit Madan |

### Current Status

- Phase: Community Bonding and technical scoping.
- Current objective: Resolve critical pipeline issues from the evaluation implementation.
- Next milestone: Parallel reduction kernels for Group B operators.

### Weekly Log Structure

Each weekly update follows the same format to keep the history readable and auditable:

- Planned work and scope boundary.
- Implemented changes and rationale.
- Validation artifacts, benchmarks, and logs.
- Open risks, blockers, and next actions.

### Working Principles

- Maintain reproducible experiments and benchmark traces.
- Keep implementation notes close to code and pull requests.
- Prioritize correctness first, then throughput improvements.
- Record tradeoffs for architecture-level decisions.

### Resources

- [Technical Code Samples](https://github.com/mmadan128/SOFIE-Parallel-Kernels)
- [Project Proposal (PDF)](/assets/SOFIE_GSOC_26.pdf)
