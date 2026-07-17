---
title: GNX Sandbox
description: Start exploring GNX in a sandbox.
kernelspec:
  name: xcpp20-openmp
  display_name: C++20-OpenMP
---

# GNX Sandbox

---

All you have to do to start working with the **GNX** library is to include *the base* header file. Optionally, you can switch to `gnx` namespace as well:

```{code-cell} cpp
#include <gnx/base>

using namespace gnx;

// optional but makes the output neater
std::locale::global(std::locale("en_US.UTF-8"));
```
Let's just download a viral genome to have a sequence to work with:
```{code-cell} cpp
auto HIV_1 = "genome://GCF_000864765.1_ViralProj15476"_sq;
```
+++
```{code-cell} cpp
describe(HIV_1);
```
+++
```{code-cell} cpp
summary(HIV_1);
```
Now it's your turn. Take it away...
+++
```{code-cell} cpp
// start exploring with GNX

```

