// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
// Example demonstrating Smith-Waterman local alignment with substitution matrices
// for protein sequence alignment

#include <fmt/core.h>
#include <gnx/sq.hpp>
#include <gnx/algorithms/local_align.hpp>

void print_alignment(const std::string& title, const gnx::alignment_result& result)
{   fmt::print("{}\n", title);
    fmt::print("{}\n", std::string(60, '-'));
    fmt::print("Alignment Score: {}\n", result.score);
    fmt::print("Position: ({}, {})\n", result.max_i, result.max_j);
    
    // Print alignment visualization
    fmt::print("Aligned Sequence 1: {}\n", result.aligned_seq1);
    fmt::print("                    ");
    for (size_t i = 0; i < result.aligned_seq1.length(); ++i)
    {   if (result.aligned_seq1[i] == result.aligned_seq2[i])
            fmt::print("|");
        else if (result.aligned_seq1[i] == '-' || result.aligned_seq2[i] == '-')
            fmt::print(" ");
        else
            fmt::print(":");  // Conservative substitution indicator
    }
    fmt::print("\nAligned Sequence 2: {}\n", result.aligned_seq2);
    fmt::print("\n");
}

int main()
{   using gnx::sq;
    using gnx::local_align;
    using namespace gnx::lut;
    
    fmt::print("Protein Sequence Alignment with Substitution Matrices\n");
    fmt::print("======================================================\n\n");
    
    // Example 1: Identical protein sequences with different matrices
    {   fmt::print("Example 1: Comparing substitution matrices on identical sequences\n");
        fmt::print("{}\n\n", std::string(60, '='));
        
        sq hemoglobin{"MVHLTPEEKSAV"};
        
        auto result_blosum62 = local_align(hemoglobin, hemoglobin, blosum62);
        auto result_blosum80 = local_align(hemoglobin, hemoglobin, blosum80);
        auto result_pam250 = local_align(hemoglobin, hemoglobin, pam250);
        
        print_alignment("BLOSUM62", result_blosum62);
        print_alignment("BLOSUM80", result_blosum80);
        print_alignment("PAM250", result_pam250);
        
        fmt::print("Note: BLOSUM80 is more stringent (for closely related sequences)\n");
        fmt::print("      BLOSUM62 is balanced (most commonly used)\n");
        fmt::print("      PAM250 is for distantly related sequences\n\n");
    }
    
    // Example 2: Conservative amino acid substitution
    {   fmt::print("Example 2: Conservative substitution (Leucine <-> Isoleucine)\n");
        fmt::print("{}\n\n", std::string(60, '='));
        
        sq seq1{"ACDEFGHIKLMNPQRSTV"};
        sq seq2{"ACDEFGHLKLMNPQRSTV"};  // I->L (both hydrophobic)
        
        auto result = local_align(seq1, seq2, blosum62);
        print_alignment("BLOSUM62", result);
        
        fmt::print("Leucine and Isoleucine are both hydrophobic amino acids\n");
        fmt::print("and often substitutable in protein structure.\n\n");
    }
    
    // Example 3: Human vs Mouse protein comparison
    {   fmt::print("Example 3: Cross-species protein comparison\n");
        fmt::print("{}\n\n", std::string(60, '='));
        
        // Simplified hemoglobin fragments
        sq human{"VLSPADKTNVKAAW"};
        sq mouse{"VLSAADKTNVKAAW"};  // P->A substitution
        
        auto result = local_align(human, mouse, blosum62);
        print_alignment("Human vs Mouse Hemoglobin Fragment (BLOSUM62)", result);
        
        fmt::print("Even with differences, conserved regions score well.\n\n");
    }
    
    // Example 4: Enzyme catalytic site alignment
    {   fmt::print("Example 4: Enzyme catalytic site alignment\n");
        fmt::print("{}\n\n", std::string(60, '='));
        
        sq enzyme1{"HDSGICN"};  // Catalytic triad-like sequence
        sq enzyme2{"HDSGVCN"};  // I->V conservative change
        
        auto result_62 = local_align(enzyme1, enzyme2, blosum62);
        auto result_80 = local_align(enzyme1, enzyme2, blosum80);
        
        print_alignment("BLOSUM62", result_62);
        print_alignment("BLOSUM80", result_80);
        
        fmt::print("Active site residues are often highly conserved.\n\n");
    }
    
    // Example 5: Finding conserved domain in longer sequence
    {   fmt::print("Example 5: Finding conserved domain in longer sequence\n");
        fmt::print("{}\n\n", std::string(60, '='));
        
        sq full_protein{"MKTIIALSYIFCLVFAACDEFGHIKLMNPQRSTVW"};
        sq conserved_domain{"ACDEFGHIKLMNPQRSTVW"};
        
        auto result = local_align(full_protein, conserved_domain, blosum62);
        print_alignment("Signal Peptide + Domain vs Domain Only", result);
        
        fmt::print("Local alignment finds the matching domain region.\n\n");
    }
    
    // Example 6: Impact of gap penalties
    {   fmt::print("Example 6: Impact of gap penalties\n");
        fmt::print("{}\n\n", std::string(60, '='));
        
        sq seq_with_del{"ARNDCQEG"};
        sq seq_normal  {"ARNDQEG"};  // C deleted
        
        auto result_gap8  = local_align(seq_with_del, seq_normal, blosum62, -8);
        auto result_gap12 = local_align(seq_with_del, seq_normal, blosum62, -12);
        
        print_alignment("Gap Penalty = -8", result_gap8);
        print_alignment("Gap Penalty = -12 (more stringent)", result_gap12);
        
        fmt::print("Higher gap penalties discourage insertions/deletions.\n\n");
    }
    
    // Example 7: Handling ambiguous amino acids
    {   fmt::print("Example 7: Handling ambiguous amino acids\n");
        fmt::print("{}\n\n", std::string(60, '='));
        
        sq seq1{"ACDEFG"};
        sq seq2{"ACBEFG"};  // D->B (B = D or N)
        
        auto result = local_align(seq1, seq2, blosum62);
        print_alignment("With Ambiguous Code B (D or N)", result);
        
        fmt::print("B represents either Aspartic acid (D) or Asparagine (N)\n");
        fmt::print("Z represents either Glutamic acid (E) or Glutamine (Q)\n");
        fmt::print("X represents any amino acid\n\n");
    }
    
    // Example 8: Realistic protein alignment
    {   fmt::print("Example 8: Realistic protein alignment\n");
        fmt::print("{}\n\n", std::string(60, '='));
        
        // First 30 amino acids of human beta-globin
        sq human_globin{"MVHLTPEEKSAVTALWGKVNVDEVGGEALG"};
        sq mouse_globin{"MVHLTDAEKSAVTALWGKVNVDEVGGEALG"};  // P->D at position 6
        
        auto result = local_align(human_globin, mouse_globin, blosum62);
        print_alignment("Human vs Mouse Beta-Globin (N-terminal)", result);
        
        fmt::print("Despite species differences, globins are highly conserved.\n\n");
    }
    
    // Example 9: Comparing PAM matrices
    {   fmt::print("Example 9: PAM matrix family comparison\n");
        fmt::print("{}\n\n", std::string(60, '='));
        
        sq seq{"MVHLTPEEK"};
        
        auto result_pam30  = local_align(seq, seq, pam30);
        auto result_pam120 = local_align(seq, seq, pam120);
        auto result_pam250 = local_align(seq, seq, pam250);
        
        print_alignment("PAM30 (very close)", result_pam30);
        print_alignment("PAM120 (moderate)", result_pam120);
        print_alignment("PAM250 (distant)", result_pam250);
        
        fmt::print("PAM number indicates evolutionary distance:\n");
        fmt::print("  PAM30  = 30 accepted mutations per 100 residues\n");
        fmt::print("  PAM120 = 120 accepted mutations per 100 residues\n");
        fmt::print("  PAM250 = 250 accepted mutations per 100 residues\n\n");
    }
    
    return 0;
}
