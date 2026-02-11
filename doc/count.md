---
title: Count Algorithm
description: Case-insensitive counting of bases and amino acids in biological sequences
---

# Count Algorithm

The `gnx::count` algorithm provides efficient, case-insensitive counting of bases (nucleotides) or amino acids in biological sequences. It returns the results as a `std::map<char, std::size_t>` where keys are uppercase characters and values are their counts.

## Features

- **Case-insensitive**: Automatically normalizes lowercase letters to uppercase for counting
- **Lookup table-based**: Uses compile-time lookup tables for efficient character normalization
- **Execution policies**: Supports sequential, parallel, SIMD, and GPU execution
- **Zero-copy operations**: Works with iterators and ranges without unnecessary copies
- **Heterogeneous computing**: Full CUDA and ROCm support for GPU acceleration

## Usage

### Basic Counting

```cpp
#include <gnx/sq.hpp>
#include <gnx/algorithms/count.hpp>

using gnx::sq;

// Count bases in a DNA sequence
sq dna = "ACGTacgtNNNN"_sq;
auto counts = gnx::count(dna);

// Result: A:2, C:2, G:2, N:4, T:2 (case-insensitive)
for (const auto& [base, count] : counts)
    fmt::print("{}: {}\n", base, count);
```

### Using Execution Policies

```cpp
#include <gnx/algorithms/count.hpp>
#include <gnx/execution.hpp>

using gnx::execution::seq;      // Sequential
using gnx::execution::par;      // Parallel (OpenMP)
using gnx::execution::unseq;    // Unsequenced (SIMD)
using gnx::execution::par_unseq; // Parallel + SIMD

// Sequential execution
auto result1 = gnx::count(seq, sequence);

// Parallel execution (OpenMP)
auto result2 = gnx::count(par, sequence);

// SIMD execution
auto result3 = gnx::count(unseq, sequence);

// Parallel + SIMD
auto result4 = gnx::count(par_unseq, sequence);
```

### GPU Acceleration

```cpp
#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/device_vector.h>

// Automatic GPU execution for device vectors
thrust::device_vector<char> d_seq = ...;
auto gpu_counts = gnx::count(d_seq); // Runs on GPU automatically
#endif
```

### Iterator-based Counting

```cpp
// Count a subsequence using iterators
sq sequence = "ACGTACGTacgtacgt"_sq;
auto partial = gnx::count(sequence.begin(), sequence.begin() + 8);
// Only counts first 8 characters
```

## API Reference

### Function Overloads

```cpp
// Range-based (policy-free)
template<std::ranges::input_range Range>
std::map<char, std::size_t> count(const Range& seq);

// Iterator-based (policy-free)
template<typename Iterator>
std::map<char, std::size_t> count(Iterator first, Iterator last);

// Range-based with execution policy
template<typename ExecPolicy, std::ranges::input_range Range>
std::map<char, std::size_t> count(ExecPolicy&& policy, const Range& seq);

// Iterator-based with execution policy
template<typename ExecPolicy, typename Iterator>
std::map<char, std::size_t> count(ExecPolicy&& policy, Iterator first, Iterator last);
```

### Parameters

- **seq**: A sequence range (e.g., `gnx::sq`, `std::vector<char>`)
- **first, last**: Iterator pair defining the sequence range
- **policy**: Execution policy (e.g., `gnx::execution::par`)

### Return Value

Returns `std::map<char, std::size_t>` where:
- Keys are **uppercase** characters found in the sequence
- Values are the total counts (combining upper and lowercase occurrences)

## Examples

### Counting Amino Acids

```cpp
sq peptide = "ARNDCQEGHILKMFPSTWYVarnDcqeghilkmfpstwyv"_sq;
auto aa_counts = gnx::count(peptide);

// Each amino acid appears twice (upper + lower case)
// Result: A:2, C:2, D:2, E:2, F:2, G:2, H:2, I:2, K:2, L:2, 
//         M:2, N:2, P:2, Q:2, R:2, S:2, T:2, V:2, W:2, Y:2
```

### Verifying Sequence Length

```cpp
auto counts = gnx::count(sequence);
std::size_t total = 0;
for (const auto& [key, value] : counts)
    total += value;

assert(total == sequence.size());
```

### Performance Comparison

```cpp
const std::size_t N = 1'000'000;
auto large_seq = gnx::random::dna<sq>(N);

// Sequential: slower but deterministic
auto seq_result = gnx::count(seq, large_seq);

// Parallel: faster for large sequences
auto par_result = gnx::count(par, large_seq);

// Results are identical
assert(seq_result == par_result);
```

## Performance Characteristics

| Execution Policy | Time Complexity | Space Complexity | Use Case |
|-----------------|-----------------|------------------|----------|
| `seq` | O(n) | O(k) | Small sequences, deterministic |
| `unseq` | O(n) | O(n + k) | Medium sequences, SIMD |
| `par` | O(n/p) | O(k*p) | Large sequences, multi-core |
| `par_unseq` | O(n/p) | O(n + k*p) | Very large sequences |
| GPU (CUDA/ROCm) | O(n/p) | O(n + k) | Massive sequences on GPU |

Where:
- n = sequence length
- k = number of unique characters (typically ≤ 20 for biological sequences)
- p = number of processing elements (CPU cores or GPU threads)

## Implementation Details

### Case Normalization

The algorithm uses a compile-time lookup table (`gnx::lut::case_fold`) that maps both uppercase and lowercase ASCII letters to their uppercase equivalents:

```cpp
// Defined in gnx/lut/case_fold.hpp
inline constexpr auto case_fold = create_case_fold_table();

// Usage
char normalized = case_fold[static_cast<uint8_t>('a')]; // Returns 'A'
```

### GPU Implementation

For GPU execution (CUDA/ROCm), the algorithm:
1. Copies the lookup table to device memory
2. Transforms all characters to uppercase using `thrust::transform`
3. Sorts the normalized characters with `thrust::sort`
4. Counts occurrences using `thrust::reduce_by_key`
5. Copies results back to host as a `std::map`

### Parallel CPU Implementation

For parallel execution on CPUs:
1. Each thread builds a local `std::map` of counts
2. Local maps are merged in a critical section
3. Final result is returned

## See Also

- [gnx::compare](compare.md) - Case-insensitive sequence comparison
- [gnx::valid](valid.md) - Sequence validation
- [Execution Policies](../execution.md) - Controlling parallelization
- [Lookup Tables](../lut.md) - Compile-time lookup tables
