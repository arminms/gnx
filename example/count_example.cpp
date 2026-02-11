// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
// Example demonstrating the gnx::count algorithm for counting bases/amino
// acids in sequences with case-insensitive counting.

#include <fmt/core.h>
#include <gnx/sq.hpp>
#include <gnx/algorithms/count.hpp>
#include <gnx/algorithms/random.hpp>

using gnx::sq;
using gnx::execution::seq;
using gnx::execution::par;
using gnx::execution::unseq;
using gnx::execution::par_unseq;

int main()
{   fmt::print("=== gnx::count Algorithm Example ===\n\n");

    // Example 1: Count bases in a simple DNA sequence
    fmt::print("Example 1: Simple DNA sequence\n");
    sq dna = "ACGTacgtNNNN"_sq;
    fmt::print("  Sequence: {}\n", dna);
    
    auto result1 = gnx::count(dna);
    fmt::print("  Counts (case-insensitive):\n");
    for (const auto& [base, count] : result1)
        fmt::print("    {} : {}\n", base, count);
    fmt::print("\n");

    // Example 2: Count amino acids in a peptide sequence
    fmt::print("Example 2: Peptide sequence\n");
    sq peptide = "ARNDCQEGHILKMFPSTWYVarnDcqeghilkmfpstwyv"_sq;
    fmt::print("  Sequence: {}\n", peptide);
    
    auto result2 = gnx::count(peptide);
    fmt::print("  Amino acid counts:\n");
    for (const auto& [aa, count] : result2)
        fmt::print("    {} : {}\n", aa, count);
    fmt::print("\n");

    // Example 3: Count using execution policies
    fmt::print("Example 3: Large sequence with execution policies\n");
    const std::size_t N = 1'000'000;
    auto large_seq = gnx::random::dna<sq>(N);
    fmt::print("  Sequence length: {} bases\n", large_seq.size());
    
    // Sequential execution
    auto result_seq = gnx::count(seq, large_seq);
    fmt::print("  Sequential execution - base counts:\n");
    std::size_t total = 0;
    for (const auto& [base, count] : result_seq)
    {   fmt::print("    {} : {}\n", base, count);
        total += count;
    }
    fmt::print("    Total: {}\n", total);
    fmt::print("\n");

    // Parallel execution
    auto result_par = gnx::count(par, large_seq);
    fmt::print("  Parallel execution - base counts:\n");
    total = 0;
    for (const auto& [base, count] : result_par)
    {   fmt::print("    {} : {}\n", base, count);
        total += count;
    }
    fmt::print("    Total: {}\n", total);
    fmt::print("\n");

    // Example 4: Mixed-case counting
    fmt::print("Example 4: Case-insensitive counting demonstration\n");
    sq mixed = "AaAaCcCcGgGgTtTt"_sq;
    fmt::print("  Sequence: {}\n", mixed);
    
    auto result4 = gnx::count(mixed);
    fmt::print("  All bases normalized to uppercase:\n");
    for (const auto& [base, count] : result4)
        fmt::print("    {} : {}\n", base, count);
    fmt::print("\n");

    // Example 5: Using iterators
    fmt::print("Example 5: Count using iterators (subsequence)\n");
    sq seq = "ACGTACGTacgtacgt"_sq;
    fmt::print("  Full sequence: {}\n", seq);
    
    // Count only the first 8 characters
    auto result5 = gnx::count(seq.begin(), seq.begin() + 8);
    fmt::print("  First 8 bases:\n");
    for (const auto& [base, count] : result5)
        fmt::print("    {} : {}\n", base, count);
    fmt::print("\n");

    return 0;
}
