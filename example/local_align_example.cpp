// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
// Example demonstrating Smith-Waterman local alignment using gnx::local_align

#include <fmt/core.h>
#include <gnx/sq.hpp>
#include <gnx/algorithms/local_align.hpp>

void print_alignment(const gnx::alignment_result& result)
{   fmt::print("Alignment Score: {}\n", result.score);
    fmt::print("Position: ({}, {})\n", result.max_i, result.max_j);
    fmt::print("Aligned Sequence 1: {}\n", result.aligned_seq1);
    fmt::print("Aligned Sequence 2: {}\n", result.aligned_seq2);
    
    // Print alignment visualization
    fmt::print("Alignment:\n");
    fmt::print("  {}\n", result.aligned_seq1);
    fmt::print("  ");
    for (size_t i = 0; i < result.aligned_seq1.length(); ++i)
    {   if (result.aligned_seq1[i] == result.aligned_seq2[i])
            fmt::print("|");
        else if (result.aligned_seq1[i] == '-' || result.aligned_seq2[i] == '-')
            fmt::print(" ");
        else
            fmt::print("x");
    }
    fmt::print("\n  {}\n\n", result.aligned_seq2);
}

int main()
{   using gnx::sq;
    using gnx::local_align;
    
    fmt::print("Smith-Waterman Local Alignment Examples\n");
    fmt::print("========================================\n\n");
    
    // Example 1: Perfect match
    {   fmt::print("Example 1: Perfect match\n");
        sq seq1{"ACGTACGT"};
        sq seq2{"ACGTACGT"};
        auto result = local_align(seq1, seq2);
        print_alignment(result);
    }
    
    // Example 2: Subsequence alignment
    {   fmt::print("Example 2: Finding subsequence\n");
        sq seq1{"AAAAAACGTACGTTTTTTT"};
        sq seq2{"ACGTACGT"};
        auto result = local_align(seq1, seq2);
        print_alignment(result);
    }
    
    // Example 3: Alignment with mismatches
    {   fmt::print("Example 3: Alignment with mismatches (SNP)\n");
        sq seq1{"ATCGATCGATCG"};
        sq seq2{"ATCGCTCGATCG"};  // Single nucleotide change
        auto result = local_align(seq1, seq2);
        print_alignment(result);
    }
    
    // Example 4: Alignment with indel
    {   fmt::print("Example 4: Alignment with insertion/deletion\n");
        sq seq1{"ATCGATCGATCG"};
        sq seq2{"ATCGTCGATCG"};  // Missing 'A' at position 5
        auto result = local_align(seq1, seq2);
        print_alignment(result);
    }
    
    // Example 5: Using custom scoring parameters
    {   fmt::print("Example 5: Custom scoring (match=5, mismatch=-3, gap=-2)\n");
        std::string seq1 = "ACGTACGT";
        std::string seq2 = "ACGTACGT";
        auto result = local_align(seq1.begin(), seq1.end(), 
                                  seq2.begin(), seq2.end(), 
                                  5, -3, -2);
        print_alignment(result);
    }
    
    // Example 6: Divergent sequences
    {   fmt::print("Example 6: Finding best local alignment in divergent sequences\n");
        sq seq1{"AAAAACGTTTTTTGCATTTTT"};
        sq seq2{"CCCCGCACCCCCACGTCCCCC"};
        auto result = local_align(seq1, seq2);
        print_alignment(result);
    }
    
    return 0;
}
