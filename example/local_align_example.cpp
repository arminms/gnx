// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
// Example demonstrating Smith-Waterman local alignment using gnx::local_align

#include <iostream>
#include <iomanip>
#include <gnx/sq.hpp>
#include <gnx/algorithms/local_align.hpp>

void print_alignment(const gnx::alignment_result& result)
{   std::cout << "Alignment Score: " << result.score << "\n";
    std::cout << "Position: (" << result.max_i << ", " << result.max_j << ")\n";
    std::cout << "Aligned Sequence 1: " << result.aligned_seq1 << "\n";
    std::cout << "Aligned Sequence 2: " << result.aligned_seq2 << "\n";
    
    // Print alignment visualization
    std::cout << "Alignment:\n";
    std::cout << "  " << result.aligned_seq1 << "\n";
    std::cout << "  ";
    for (size_t i = 0; i < result.aligned_seq1.length(); ++i)
    {   if (result.aligned_seq1[i] == result.aligned_seq2[i])
            std::cout << '|';
        else if (result.aligned_seq1[i] == '-' || result.aligned_seq2[i] == '-')
            std::cout << ' ';
        else
            std::cout << 'x';
    }
    std::cout << "\n  " << result.aligned_seq2 << "\n";
    std::cout << std::endl;
}

int main()
{   using gnx::sq;
    using gnx::local_align;
    
    std::cout << "Smith-Waterman Local Alignment Examples\n";
    std::cout << "========================================\n\n";
    
    // Example 1: Perfect match
    {   std::cout << "Example 1: Perfect match\n";
        sq seq1{"ACGTACGT"};
        sq seq2{"ACGTACGT"};
        auto result = local_align(seq1, seq2);
        print_alignment(result);
    }
    
    // Example 2: Subsequence alignment
    {   std::cout << "Example 2: Finding subsequence\n";
        sq seq1{"AAAAAACGTACGTTTTTTT"};
        sq seq2{"ACGTACGT"};
        auto result = local_align(seq1, seq2);
        print_alignment(result);
    }
    
    // Example 3: Alignment with mismatches
    {   std::cout << "Example 3: Alignment with mismatches (SNP)\n";
        sq seq1{"ATCGATCGATCG"};
        sq seq2{"ATCGCTCGATCG"};  // Single nucleotide change
        auto result = local_align(seq1, seq2);
        print_alignment(result);
    }
    
    // Example 4: Alignment with indel
    {   std::cout << "Example 4: Alignment with insertion/deletion\n";
        sq seq1{"ATCGATCGATCG"};
        sq seq2{"ATCGTCGATCG"};  // Missing 'A' at position 5
        auto result = local_align(seq1, seq2);
        print_alignment(result);
    }
    
    // Example 5: Using custom scoring parameters
    {   std::cout << "Example 5: Custom scoring (match=5, mismatch=-3, gap=-2)\n";
        std::string seq1 = "ACGTACGT";
        std::string seq2 = "ACGTACGT";
        auto result = local_align(seq1.begin(), seq1.end(), 
                                  seq2.begin(), seq2.end(), 
                                  5, -3, -2);
        print_alignment(result);
    }
    
    // Example 6: Divergent sequences
    {   std::cout << "Example 6: Finding best local alignment in divergent sequences\n";
        sq seq1{"AAAAACGTTTTTTGCATTTTT"};
        sq seq2{"CCCCGCACCCCCACGTCCCCC"};
        auto result = local_align(seq1, seq2);
        print_alignment(result);
    }
    
    return 0;
}
