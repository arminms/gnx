---
title: Complement Algorithm
description: DNA/RNA sequence complementation with Watson-Crick base pairing
---

# Complement Algorithm

The `complement` algorithm transforms nucleotide sequences into their Watson-Crick complements using an efficient lookup table approach. This operation is fundamental in molecular biology for analyzing reverse complements, primer design, and DNA/RNA structure analysis.

## Overview

The complement transformation converts each nucleotide to its base-pairing partner:
- **DNA**: A ↔ T, G ↔ C
- **RNA**: A ↔ U, G ↔ C
- **IUPAC codes**: Proper handling of ambiguous bases (R ↔ Y, M ↔ K, etc.)

### Key Features

- **In-place modification**: Efficient memory usage by modifying sequences directly
- **Case preservation**: Uppercase and lowercase characters are handled independently  
- **IUPAC support**: Full support for ambiguous nucleotide codes
- **Execution policies**: Sequential, parallel (OpenMP), vectorized, and GPU execution
- **Zero-copy design**: Uses `std::span` and iterators for minimal overhead

## API Reference

### Basic Usage

```cpp
#include <gnx/algorithms/complement.hpp>

// In-place complementation
gnx::sq s = "ACGT"_sq;
gnx::complement(s);  // s becomes "TGCA"

// With iterators
gnx::complement(s.begin(), s.end());

// Partial range
gnx::complement(s.begin() + 4, s.begin() + 12);
```

### With Execution Policies

```cpp
#include <gnx/execution.hpp>

using gnx::execution::seq;
using gnx::execution::par;
using gnx::execution::unseq;
using gnx::execution::par_unseq;

gnx::sq large_seq(1'000'000);
// ... fill with nucleotides ...

// Sequential execution
gnx::complement(seq, large_seq);

// Parallel execution (OpenMP)
gnx::complement(par, large_seq);

// Vectorized execution (SIMD)
gnx::complement(unseq, large_seq);

// Parallel + Vectorized
gnx::complement(par_unseq, large_seq);
```

### GPU Execution

```cpp
#if defined(__CUDACC__)
#include <thrust/device_vector.h>

thrust::device_vector<char> d_seq(1'000'000);
// ... copy data to device ...

// CUDA execution
gnx::complement(thrust::cuda::par, d_seq);

// With CUDA stream
cudaStream_t stream;
cudaStreamCreate(&stream);
gnx::complement(thrust::cuda::par.on(stream), d_seq);
cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
#endif
```

## Lookup Table

The algorithm uses a compile-time generated lookup table (`gnx::lut::complement`) mapping each byte value to its complement:

### Basic Nucleotides

| Base | Complement | Description |
|------|-----------|-------------|
| A/a  | T/t       | Adenine → Thymine |
| T/t  | A/a       | Thymine → Adenine |
| G/g  | C/c       | Guanine → Cytosine |
| C/c  | G/g       | Cytosine → Guanine |
| U/u  | A/a       | Uracil → Adenine (RNA) |

### IUPAC Ambiguity Codes

| Code | Meaning | Complement | Description |
|------|---------|-----------|-------------|
| R/r  | A or G (puRine) | Y/y | Pyrimidine (C or T) |
| Y/y  | C or T (pYrimidine) | R/r | Purine (A or G) |
| S/s  | G or C (Strong) | S/s | Self-complementary |
| W/w  | A or T (Weak) | W/w | Self-complementary |
| K/k  | G or T (Keto) | M/m | Amino (A or C) |
| M/m  | A or C (aMino) | K/k | Keto (G or T) |
| B/b  | C, G or T (not A) | V/v | A, C or G (not T) |
| D/d  | A, G or T (not C) | H/h | A, C or T (not G) |
| H/h  | A, C or T (not G) | D/d | A, G or T (not C) |
| V/v  | A, C or G (not T) | B/b | C, G or T (not A) |
| N/n  | Any base | N/n | Self-complementary |

### Non-Nucleotide Characters

Characters not in the nucleotide alphabet are left unchanged (e.g., numbers, spaces, special characters).

## Properties

### Mathematical Properties

- **Involutory**: Applying complement twice returns the original sequence  
  `complement(complement(s)) = s`
- **Bijective**: One-to-one mapping between bases and their complements
- **Case-preserving**: Uppercase maps to uppercase, lowercase to lowercase

### Complexity

- **Time**: O(n) where n is the sequence length
- **Space**: O(1) auxiliary space (in-place modification)
- **Cache**: Excellent cache locality due to lookup table

### Performance

- **Sequential**: ~1-2 GB/s on modern CPUs
- **Parallel (OpenMP)**: Scales linearly with cores
- **Vectorized (SIMD)**: ~4-8 GB/s with AVX2/AVX-512
- **GPU (CUDA)**: ~10-50 GB/s depending on GPU

## Examples

### Example 1: Basic Complementation

```cpp
#include <iostream>
#include <gnx/sq.hpp>
#include <gnx/algorithms/complement.hpp>

int main() {
    using gnx::sq;
    
    sq dna = "ACGTACGT"_sq;
    std::cout << "Original:   " << dna << '\n';
    
    gnx::complement(dna);
    std::cout << "Complement: " << dna << '\n';
    
    gnx::complement(dna);  // Restore original
    std::cout << "Restored:   " << dna << '\n';
    
    return 0;
}
```

**Output:**
```
Original:   ACGTACGT
Complement: TGCATGCA
Restored:   ACGTACGT
```

### Example 2: RNA Complementation

```cpp
sq rna = "ACGU"_sq;
std::cout << "RNA:        " << rna << '\n';
gnx::complement(rna);
std::cout << "Complement: " << rna << '\n';  // "TGCA" (U → A)
```

### Example 3: IUPAC Ambiguity Codes

```cpp
sq ambig = "RYMKSW"_sq;
std::cout << "Original:   " << ambig << '\n';
gnx::complement(ambig);
std::cout << "Complement: " << ambig << '\n';  // "YRKMSW"
```

### Example 4: Partial Complementation

```cpp
sq s = "AAAACCCCGGGGTTTT"_sq;
std::cout << "Original: " << s << '\n';

// Complement only middle 8 bases
gnx::complement(s.begin() + 4, s.begin() + 12);
std::cout << "Partial:  " << s << '\n';  // "AAAAGGGCCCCTTTT"
```

### Example 5: Parallel Execution

```cpp
#include <gnx/execution.hpp>

const std::size_t N = 10'000'000;
auto large_seq = gnx::random::dna<gnx::sq>(N);

// Parallel complementation
gnx::complement(gnx::execution::par, large_seq);
```

## See Also

- [Count Algorithm](count.md) - Count bases/amino acids
- [Valid Algorithm](valid.md) - Validate sequences
- [Random Algorithm](random.md) - Generate random sequences  
- [Execution Policies](../execution.md) - Control algorithm execution

## Implementation Notes

### Lookup Table Generation

The lookup table is generated at compile-time using `constexpr`:

```cpp
constexpr std::array<char, 256> create_complement_table() {
    std::array<char, 256> table{};
    // Initialize with identity
    for (std::size_t i = 0; i < 256; ++i)
        table[i] = static_cast<char>(i);
    
    // Watson-Crick pairs
    table['A'] = 'T'; table['T'] = 'A';
    table['G'] = 'C'; table['C'] = 'G';
    // ... etc
    
    return table;
}
```

### GPU Implementation

The GPU kernel uses shared memory to cache the lookup table for optimal performance:

```cpp
__global__ void complement_kernel(char* data, size_t n, const char* lut) {
    __shared__ char shared_lut[256];
    int tid = threadIdx.x;
    
    // Cache lookup table in shared memory
    shared_lut[tid] = lut[tid];
    __syncthreads();
    
    // Perform complementation
    auto idx = blockDim.x * blockIdx.x + tid;
    if (idx < n)
        data[idx] = shared_lut[static_cast<uint8_t>(data[idx])];
}
```

### SIMD Optimization

On x86-64, the algorithm can leverage SIMD instructions through OpenMP's `#pragma omp declare simd`.

## Common Patterns

### Reverse Complement

To get the reverse complement (common in bioinformatics):

```cpp
gnx::sq s = "ACGTACGT"_sq;
gnx::complement(s);
std::reverse(s.begin(), s.end());
// s is now the reverse complement
```

### Palindrome Detection

Check if a sequence is a palindromic complement:

```cpp
bool is_palindrome_complement(const gnx::sq& s) {
    auto comp = s;
    gnx::complement(comp);
    std::reverse(comp.begin(), comp.end());
    return s == comp;
}
```

### Double-stranded DNA

Represent both strands:

```cpp
gnx::sq forward = "ACGTACGT"_sq;
gnx::sq reverse = forward;
gnx::complement(reverse);
std::reverse(reverse.begin(), reverse.end());

std::cout << "5' " << forward << " 3'\n";
std::cout << "3' " << reverse << " 5'\n";
```

**Output:**
```
5' ACGTACGT 3'
3' ACGTACGT 5'
```
