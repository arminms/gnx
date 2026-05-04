---
title: GNX
subtitle: Biosequences for Modern C++
description: GNX is a header-only Modern C++ library for biological sequence analysis with heterogeneous computing support
---

[![Build and Test (Linux/macOS/Windows)](https://github.com/arminms/gnx/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/arminms/gnx/actions/workflows/cmake-multi-platform.yml)
[![GitHub Release](https://img.shields.io/github/v/release/arminms/gnx?logo=github&logoColor=lightgray)](https://github.com/arminms/gnx/releases)
[![GitHub License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arminms/gnx/HEAD?labpath=quickstart.ipynb)

---

::::{grid} 1 1 2 2

:::{grid-item}

GNX (pronounced as jeenks) is a <wiki:header-only> Modern <wiki:C++> library for biological sequence analysis.

It provides efficient, composable types for representing <wiki:DNA>, <wiki:RNA>, and protein sequences, together with algorithms for alignment, validation, complementation, and counting — all with first-class support for heterogeneous computing on CPUs and GPUs.

Built on <wiki:C++20> features — Concepts, Ranges, and `std::span` — GNX is designed for zero-copy operations, SIMD acceleration, and seamless portability between <wiki:CUDA>, <wiki:ROCm>, and OpenMP backends.

:::

:::{grid-item}

```{code-block} cpp
#include <gnx/sq.hpp>
#include <gnx/algorithms/count.hpp>
#include <gnx/algorithms/local_align.hpp>

// Create sequences from literals
auto s1 = "ACGTACGT"_sq;
auto s2 = "ACGGACGT"_sq;

// Count nucleotide composition
auto counts = gnx::count(s1);

// Smith-Waterman local alignment
auto result = gnx::local_align(s1, s2);
```

:::

::::

:::::{aside}

::::{important} Try GNX in a Container
:class: dropdown
[Docker:](wiki:Docker_(software))
```bash
docker run -p 8888:8888 -it --rm arminms/gnx
```
[Apptainer:](wiki:Singularity_(software))
```bash
apptainer run docker://arminms/gnx:latest
```
::::

::::{seealso} Try GNX on Binder
:class: dropdown

👉 [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arminms/gnx/HEAD?labpath=quickstart.ipynb)

_Be advised sometimes it takes several minutes to start!_

::::

:::::

---

## Key Features 🥇

::::{grid} 1 1 2 2

:::{card}
:header: 🧬 Header-Only C++20 Library
:footer: [Get Started »](./quickstart.md)

Zero compilation overhead — just `#include` the headers and link `gnx::gnx` via CMake. Leverages C++20 Concepts, Ranges, and `std::span` throughout.
:::

:::{card}
:header: ⚡ Heterogeneous Computing
:footer: [Learn more »](./quickstart.md#execution-policies)

First-class execution policies: `seq`, `par`, `unseq`, `par_unseq` for CPU; `cuda`, `rocm`, `oneapi` for GPU. Write once, run anywhere.
:::

:::{card}
:header: 🗜️ 2-Bit Packed Sequences
:footer: [Learn more »](./psq.md)

`gnx::psq2` stores nucleotides in 2 bits each — **4× less memory** than one byte per character. Critical for whole-genome workloads and GPU-resident data.
:::

:::{card}
:header: 📖 Extensible Tagged Metadata
:footer: [Learn more »](./quickstart.md#tagged-metadata)

Sequences carry arbitrary typed metadata via `std::any` tags (`_id`, `_qs`, `_desc`, ...). Extensible visitors handle serialization/deserialization automatically.
:::

:::{card}
:header: 🚀 FASTA/FASTQ Streaming
:footer: [Learn more »](./quickstart.md#loading-sequences)

Load records from gzip-compressed or plain FASTA/FASTQ files — or stdin — by name or index, with near-zero RAM overhead via `gnx::virtual_vector` random access.
:::

:::{card}
:header: 🔬 Sequence Algorithms
:footer: [Learn more »](./quickstart.md#algorithms)

Smith-Waterman local alignment, nucleotide/amino-acid counting, complement, validation, and random generation — all with multi-backend execution policy support.
:::

::::

---

(features)=
## Feature Highlights 🪄

::::{grid} 1 1 2 2

:::{card}
:header: 🧬 [Sequence Types](./quickstart.md#sequence-types)
:footer: [Learn more »](./quickstart.md#sequence-types)

**`gnx::sq`** (`std::vector<char>` backed) and **`gnx::psq2`** (2-bit packed) cover typical and memory-critical use cases. Both carry tagged metadata and inter-convert freely.
:::

:::{card}
:header: 🔭 [Non-Owning Views](./quickstart.md#non-owning-sequence-views)
:footer: [Learn more »](./quickstart.md#non-owning-sequence-views)

`gnx::sq_view` is a zero-copy, `std::string_view`-style view over any `gnx::sq`. Slice, compare, and iterate without allocation.
:::

:::{card}
:header: 📚 [Sequence Banks](./quickstart.md#sequence-banks)
:footer: [Learn more »](./quickstart.md#sequence-banks)

`gnx::sequence_bank` wraps any range-compatible backend (`forward_stream`, `virtual_vector`) into a uniform iterable collection — stream through a genome or random-access a chromosome.
:::

:::{card}
:header: 🔀 [Smith-Waterman Alignment](./local_align.md)
:footer: [Learn more »](./local_align.md)

Optimal local alignment with flexible match/mismatch/gap scoring and substitution matrix support (BLOSUM 45/62/80, PAM 30/120/250) for protein sequences.
:::

:::{card}
:header: 🧮 [Nucleotide Counting](./count.md)
:footer: [Learn more »](./count.md)

`gnx::count` returns a `std::map<char, size_t>` of case-normalized base or amino-acid counts, accelerated by lookup tables and your chosen execution policy.
:::

:::{card}
:header: 🔄 [Complement Algorithm](./complement.md)
:footer: [Learn more »](./complement.md)

In-place Watson-Crick complementation with full IUPAC ambiguity code support, case preservation, and SIMD/GPU acceleration.
:::

::::

---

## Supported Platforms & Compilers

| Platform | Compiler | GPU Backend |
|----------|----------|-------------|
| Linux    | GCC, Clang | CUDA (nvcc), ROCm (hipcc) |
| macOS    | Apple Clang, GCC | — |
| Windows  | MSVC | CUDA (nvcc) |

---

## Quick Installation

GNX is header-only. The fastest way to use it is via CMake `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(gnx
  GIT_REPOSITORY https://github.com/arminms/gnx.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(gnx)
target_link_libraries(my_target PRIVATE gnx::gnx)
```

Or install locally and use `find_package`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix $HOME/.local
```

```cmake
find_package(gnx CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE gnx::gnx)
```

---

## A First Look

```{code-block} cpp
#include <gnx/sq.hpp>
#include <gnx/io/fastaqz.hpp>
#include <gnx/algorithms/count.hpp>
#include <gnx/algorithms/complement.hpp>
#include <gnx/algorithms/local_align.hpp>

// 1. Create sequences from string literals
auto dna = "ACGTACGT"_sq;

// 2. Tagged metadata (arbitrary types via std::any)
dna["_id"] = std::string("my-read");
dna["_len"] = static_cast<int>(dna.size());

// 3. Load a record from a gzip-compressed FASTA by name
gnx::sq plasmid;
plasmid.load("genome.fa.gz", "NC_003888.3", gnx::in::faqz{});

// 4. Count nucleotide composition
auto counts = gnx::count(plasmid);          // A, C, G, T, N, ...

// 5. In-place Watson-Crick complement
gnx::complement(dna);                       // "ACGTACGT" → "TGCATGCA"

// 6. Smith-Waterman local alignment
auto aln = gnx::local_align("ACGTACGT"_sq, "CGTACG"_sq);
fmt::print("score={} seq1={} seq2={}\n",
           aln.score, aln.seq1_aligned, aln.seq2_aligned);

// 7. 2-bit packed sequence (4× memory reduction)
auto packed = "ACGTACGT"_psq2;
assert(packed == dna);                      // cross-comparison works
```

---

## Documentation

```{toctree}
:maxdepth: 2
quickstart
psq
count
complement
local_align
```
