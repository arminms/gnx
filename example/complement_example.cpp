// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
// Example demonstrating the complement algorithm for nucleotide sequences

#include <iostream>
#include <gnx/sq.hpp>
#include <gnx/algorithms/complement.hpp>
#include <gnx/execution.hpp>

int main()
{   using gnx::sq;
    using gnx::execution::par;

    // Basic DNA complementation
    std::cout << "=== Basic DNA Complementation ===\n";
    sq s1 = "ACGTACGT"_sq;
    std::cout << "Original:   " << s1 << '\n';
    gnx::complement(s1);
    std::cout << "Complement: " << s1 << '\n';
    gnx::complement(s1);  // Double complement = identity
    std::cout << "Restored:   " << s1 << '\n';
    std::cout << '\n';

    // Case preservation
    std::cout << "=== Case Preservation ===\n";
    sq s2 = "AcGtNn"_sq;
    std::cout << "Original:   " << s2 << '\n';
    gnx::complement(s2);
    std::cout << "Complement: " << s2 << '\n';
    std::cout << '\n';

    // RNA complementation
    std::cout << "=== RNA Complementation ===\n";
    sq s3 = "ACGU"_sq;
    std::cout << "Original:   " << s3 << '\n';
    gnx::complement(s3);
    std::cout << "Complement: " << s3 << " (U -> A)\n";
    std::cout << '\n';

    // IUPAC ambiguity codes
    std::cout << "=== IUPAC Ambiguity Codes ===\n";
    sq s4 = "RYMKSWBDHVN"_sq;
    std::cout << "Original:   " << s4 << '\n';
    gnx::complement(s4);
    std::cout << "Complement: " << s4 << '\n';
    std::cout << "  R (puRine: A|G)        <-> Y (pYrimidine: C|T)\n";
    std::cout << "  M (aMino: A|C)         <-> K (Keto: G|T)\n";
    std::cout << "  B (not A: C|G|T)       <-> V (not T: A|C|G)\n";
    std::cout << "  D (not C: A|G|T)       <-> H (not G: A|C|T)\n";
    std::cout << "  S (Strong: G|C)        <-> S (self-complementary)\n";
    std::cout << "  W (Weak: A|T)          <-> W (self-complementary)\n";
    std::cout << "  N (aNy base)           <-> N (self-complementary)\n";
    std::cout << '\n';

    // Parallel execution
    std::cout << "=== Parallel Execution ===\n";
    sq large(100000);
    // (Assume it's filled with nucleotides)
    std::fill(large.begin(), large.end(), 'A');
    std::cout << "Large sequence (100,000 bases):\n";
    std::cout << "First 20:   " << sq(large.begin(), large.begin() + 20) << '\n';
    gnx::complement(par, large);  // Parallel complementation
    std::cout << "Complement: " << sq(large.begin(), large.begin() + 20) << '\n';
    std::cout << '\n';

    // Partial complementation
    std::cout << "=== Partial Complementation ===\n";
    sq s5 = "AAAACCCCGGGGTTTT"_sq;
    std::cout << "Original:   " << s5 << '\n';
    gnx::complement(s5.begin() + 4, s5.begin() + 12);  // Complement middle section only
    std::cout << "Partial:    " << s5 << " (middle 8 bases complemented)\n";

    return 0;
}
