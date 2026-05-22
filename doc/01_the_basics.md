---
title: The Basics
subtitle: Using gnx in a nutshell
subject: gnx Quickstart Tutorial
description: Basic steps and concepts to work with gnx.
kernelspec:
  name: xcpp20-openmp
  display_name: C++20-OpenMP
---

# The Basics

---

(header_file)=
## The header file

The `gnx` library is *header-only*. That means we can start using it by just including the necessary header files. That often comes down to *the base* header that contains the core components:
+++
```{code-cell} cpp
:tags: [remove-cell]

std::locale::global(std::locale("en_US.UTF-8"));
```
+++
```{code-cell} cpp

#include <gnx/base>
```

:::::{seealso} `cling` include paths
:class: dropdown

If `gnx` is not installed in a standard folder for headers, you can add it to the [cling include paths](xref:cling#chapters/grammar) using one of the following methods:

::::{tab-set}
:::{tab-item} #pragma

```cpp
#pragma cling add_include_path("gnx/include/path")
```
:::

:::{tab-item} .(command)
In a `cling` REPL, but not Jupyter Notebook you can use `.I`:
```cpp
.I "gnx/include/path"
```

:::

::::

:::::

## The `gnx::sq` class
Making a biological sequence in `gnx` is easy:
```{code-cell} cpp
:tags: [hide-output]

gnx::sq s{"ACGT"};
s
```
Or even easier using *string literals*:
```{code-cell} cpp
:tags: [hide-output]

auto t = "ACGT"_sq;
(s == t)
```

## Non-owning sequence views

`gnx::sq_view` is a lightweight, non-owning view over a `gnx::sq` (or any compatible container), similar to [`std::string_view`](https://en.cppreference.com/cpp/string/basic_string_view). It avoids copying while letting you slice and compare.

```{code-cell} cpp
:tags: [hide-output]
gnx::sq_view v{s};         // view over s, no copy
v
```
+++
```{code-cell} cpp
:tags: [hide-output]
// slicing without allocation
auto mid = v.subseq(1, 2);
mid
```
+++
```{code-cell} cpp
:tags: [hide-output]
t(1, 3)
```
+++
```{code-cell} cpp
:tags: [hide-output]
gnx::sq plasmid;
plasmid.load("GCF_000204255.1_ASM20425v1_genomic.fna.gz", 1);
plasmid(700, 5000)
```
