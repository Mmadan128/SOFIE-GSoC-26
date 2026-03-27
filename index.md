---
layout: home
author_profile: true
title: "Project Dashboard"
sidebar:
  nav: "sidebar"
---

### **Project Overview**
[cite_start]This project focuses on porting the **SOFIE** inference engine to a portable GPU backend using **alpaka**[cite: 3, 22]. [cite_start]By addressing the four critical GPU pipeline bugs identified during the evaluation task, we enable native GPU execution across NVIDIA and AMD hardware[cite: 15, 18, 84].

### **Current Focus**
* [cite_start]**Phase**: Application Review / Community Bonding[cite: 74, 78].
* [cite_start]**Key Goal**: Fixing the Virtual Method Mismatch and Deduplication issues in the GPU pipeline[cite: 28, 34, 49].
* [cite_start]**Next Milestone**: Implementing Group A activation kernels and Group B parallel reduction kernels[cite: 78].

### **Quick Links**
* [**Parallel Code Samples**](https://github.com/mmadan128/SOFIE-Parallel-Kernels): Pseudo-code demonstrating GPU-native logic.
* [cite_start][**Full Project Proposal**](/assets/SOFIE_GSOC_26.pdf): Detailed timeline and operator roadmap[cite: 1, 2].
