// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
// Example demonstrating the gnx::count algorithm for counting bases/amino
// acids in sequences with case-insensitive counting.

#include <algorithm>
#include <vector>
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

    // Example 6: K-mer counting (2-mers/dinucleotides)
    fmt::print("Example 6: K-mer counting - dinucleotides (2-mers)\n");
    sq dna2 = "ACGTACGTACGT"_sq;
    fmt::print("  Sequence: {}\n", dna2);
    
    auto kmers2 = gnx::count(dna2, 2);  // Count 2-mers
    fmt::print("  2-mer counts:\n");
    for (const auto& [kmer, count] : kmers2)
        fmt::print("    {} : {}\n", kmer, count);
    fmt::print("\n");

    // Example 7: K-mer counting (3-mers/trinucleotides)
    fmt::print("Example 7: K-mer counting - trinucleotides (3-mers)\n");
    sq dna3 = "ACGTACGTACGTACGT"_sq;
    fmt::print("  Sequence: {}\n", dna3);
    
    auto kmers3 = gnx::count(dna3, 3);  // Count 3-mers
    fmt::print("  3-mer counts:\n");
    for (const auto& [kmer, count] : kmers3)
        fmt::print("    {} : {}\n", kmer, count);
    fmt::print("\n");

    // Example 8: K-mer counting with parallel execution
    fmt::print("Example 8: Parallel k-mer counting on large sequence\n");
    auto large_dna = gnx::random::dna<sq>(10000);
    fmt::print("  Sequence length: {} bases\n", large_dna.size());
    
    // Count 5-mers in parallel
    auto kmers5_par = gnx::count(par, large_dna, 5);
    fmt::print("  Number of unique 5-mers found: {}\n", kmers5_par.size());
    
    // Show top 10 most frequent 5-mers
    std::vector<std::pair<std::string, std::size_t>> sorted_kmers(kmers5_par.begin(), kmers5_par.end());
    std::sort(sorted_kmers.begin(), sorted_kmers.end(), 
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    fmt::print("  Top 10 most frequent 5-mers:\n");
    for (std::size_t i = 0; i < std::min<std::size_t>(10, sorted_kmers.size()); ++i)
        fmt::print("    {} : {}\n", sorted_kmers[i].first, sorted_kmers[i].second);
    fmt::print("\n");

    // Example 9: Case-insensitive k-mer counting
    fmt::print("Example 9: Case-insensitive k-mer counting\n");
    sq mixed_case = "AcGtAcGtAcGt"_sq;
    fmt::print("  Sequence: {}\n", mixed_case);
    
    auto kmers_mixed = gnx::count(mixed_case, 4);
    fmt::print("  4-mer counts (normalized to uppercase):\n");
    for (const auto& [kmer, count] : kmers_mixed)
        fmt::print("    {} : {}\n", kmer, count);
    fmt::print("\n");

    return 0;
}
