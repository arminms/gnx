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

---

## K-mer Counting

The `gnx::count` algorithm also provides k-mer (word) counting functionality. K-mers are subsequences of length k within a biological sequence. This is useful for motif finding, sequence composition analysis, and many bioinformatics applications.

### Basic K-mer Counting

```cpp
#include <gnx/sq.hpp>
#include <gnx/algorithms/count.hpp>

using gnx::sq;

// Count dinucleotides (2-mers)
sq dna = "ACGTACGT"_sq;
auto dinucs = gnx::count(dna, 2);

// Result: AC:2, CG:2, GT:2, TA:1
for (const auto& [kmer, count] : dinucs)
    fmt::print("{}: {}\n", kmer, count);

// Count trinucleotides (3-mers)
auto trinucs = gnx::count(dna, 3);
// Result: ACG:2, CGT:2, GTA:1, TAC:1
```

### K-mer Counting with Execution Policies

```cpp
#include <gnx/execution.hpp>

using gnx::execution::par;

// Parallel k-mer counting on large sequences
sq large_seq = gnx::random::dna<sq>(1'000'000);

// Count 5-mers in parallel
auto kmers = gnx::count(par, large_seq, 5);

fmt::print("Found {} unique 5-mers\n", kmers.size());

// Find most frequent k-mer
auto max_kmer = std::max_element(
    kmers.begin(), kmers.end(),
    [](const auto& a, const auto& b) { return a.second < b.second; }
);
fmt::print("Most frequent 5-mer: {} ({}x)\n", 
           max_kmer->first, max_kmer->second);
```

### K-mer Counting API

```cpp
// Range-based (policy-free)
template<std::ranges::input_range Range>
std::unordered_map<std::string, std::size_t> count
(   const Range& seq
,   std::size_t word_length
);

// Iterator-based (policy-free)
template<typename Iterator>
std::unordered_map<std::string, std::size_t> count
(   Iterator first
,   Iterator last
,   std::size_t word_length
);

// Range-based with execution policy
template<typename ExecPolicy, std::ranges::input_range Range>
std::unordered_map<std::string, std::size_t> count
(   ExecPolicy&& policy
,   const Range& seq
,   std::size_t word_length
);

// Iterator-based with execution policy
template<typename ExecPolicy, typename Iterator>
std::unordered_map<std::string, std::size_t> count
(   ExecPolicy&& policy
,   Iterator first
,   Iterator last
,   std::size_t word_length
);
```

### K-mer Properties

- **Overlapping**: K-mers are counted with overlapping windows. For example, "AAAA" contains 3 occurrences of "AA"
- **Case-insensitive**: Like single-character counting, k-mers are normalized to uppercase
- **Total count**: For a sequence of length n, there are (n - k + 1) k-mers of length k

```cpp
sq seq = "ACGTACGT"_sq;  // 8 bases

auto kmers3 = gnx::count(seq, 3);  // 3-mers
std::size_t total = 0;
for (const auto& [kmer, count] : kmers3)
    total += count;

// total == 8 - 3 + 1 == 6
assert(total == 6);
```

### K-mer Performance

| Execution Policy | Time Complexity | Space Complexity | Use Case |
|-----------------|-----------------|------------------|----------|
| `seq` | O(n*k) | O(m*k) | Small sequences |
| `par` | O((n*k)/p) | O(m*k*p) | Large sequences |
| GPU (CUDA/ROCm) | O((n*k)/p) | O(m*k) | Massive sequences |

Where:
- n = sequence length
- k = k-mer length
- m = number of unique k-mers
- p = number of processing elements

### GPU K-mer Implementation

The GPU implementation currently provides a fallback to the CPU version by copying device data to the host. A future optimized version will use:

- **CUDA**: [cuCollections](https://github.com/NVIDIA/cuCollections) concurrent hash maps
- **ROCm**: [hipCollections](https://github.com/ROCm/hipCollections) concurrent hash maps

These libraries provide high-performance concurrent hash tables optimized for GPU architectures.

### Common K-mer Applications

**Dinucleotides (k=2)**: CpG island detection, codon usage bias
```cpp
auto dinucs = gnx::count(dna, 2);
double cpg_ratio = static_cast<double>(dinucs["CG"]) / dna.size();
```

**Trinucleotides (k=3)**: Codon analysis, reading frame detection
```cpp
auto codons = gnx::count(dna, 3);
fmt::print("ATG (start codon) count: {}\n", codons["ATG"]);
```

**5-mers and longer**: Motif finding, sequence signatures
```cpp
// Find over-represented 6-mers
auto kmers6 = gnx::count(dna, 6);
double expected_freq = 1.0 / std::pow(4.0, 6.0);  // Random expectation
for (const auto& [kmer, count] : kmers6) {
    double observed_freq = static_cast<double>(count) / (dna.size() - 6 + 1);
    if (observed_freq > 2.0 * expected_freq)
        fmt::print("Over-represented: {} ({}x expected)\n", kmer, observed_freq / expected_freq);
}
```

