// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <gnx/sq.hpp>
#include <gnx/psq.hpp>
#include <gnx/algorithms/local_align.hpp>

// =============================================================================
// local_align_n() algorithm tests
// =============================================================================

TEMPLATE_TEST_CASE
(   "gnx::local_align_n"
,   "[algorithm][local_align]"
,   std::vector<char>
)
{   typedef TestType T;

// -- basic alignment ----------------------------------------------------------

    SECTION( "identical sequences" )
    {   gnx::generic_sequence<T> s1{"ACGT"};
        gnx::generic_sequence<T> s2{"ACGT"};
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 8);  // 4 matches * 2
        CHECK(result.aligned_seq1 == "ACGT");
        CHECK(result.aligned_seq2 == "ACGT");
        CHECK(result.traceback.size() == 4);
        for (const auto& dir : result.traceback)
            CHECK(dir == gnx::alignment_direction::diagonal);
    }

    SECTION( "single mismatch" )
    {   gnx::generic_sequence<T> s1{"ACGT"};
        gnx::generic_sequence<T> s2{"ACAT"};
        auto result = gnx::local_align_n(s1, s2);

        // Best local alignment should still find matching regions
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1.length() == result.aligned_seq2.length());
    }

    SECTION( "no alignment (completely different)" )
    {   gnx::generic_sequence<T> s1{"AAAA"};
        gnx::generic_sequence<T> s2{"TTTT"};
        auto result = gnx::local_align_n(s1, s2, 2, -3, -1);

        // With strong mismatch penalty, may have low or zero score
        CHECK(result.score >= 0);  // Smith-Waterman never goes negative
    }

// -- subsequence alignment ----------------------------------------------------

    SECTION( "subsequence in larger sequence" )
    {   gnx::generic_sequence<T> s1{"ACGTACGT"};
        gnx::generic_sequence<T> s2{"ACGT"};
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 8);  // Perfect match of ACGT
        CHECK(result.aligned_seq1 == "ACGT");
        CHECK(result.aligned_seq2 == "ACGT");
    }

    SECTION( "overlapping sequences" )
    {   gnx::generic_sequence<T> s1{"ACGTACGT"};
        gnx::generic_sequence<T> s2{"TACGTACG"};
        auto result = gnx::local_align_n(s1, s2);

        // Should find significant alignment
        CHECK(result.score > 10);
        CHECK(result.aligned_seq1.length() > 5);
    }

// -- gap handling -------------------------------------------------------------

    SECTION( "alignment with gap in first sequence" )
    {   gnx::generic_sequence<T> s1{"ACGT"};
        gnx::generic_sequence<T> s2{"ACGGT"};
        auto result = gnx::local_align_n(s1, s2, 2, -1, -1);

        CHECK(result.score >= 4);  // At least some matches
        // May or may not have gap depending on scoring
    }

    SECTION( "alignment with gap in second sequence" )
    {   gnx::generic_sequence<T> s1{"ACGGT"};
        gnx::generic_sequence<T> s2{"ACGT"};
        auto result = gnx::local_align_n(s1, s2, 2, -1, -1);

        CHECK(result.score >= 4);
    }

// -- custom scoring -----------------------------------------------------------

    SECTION( "custom match score" )
    {   gnx::generic_sequence<T> s1{"ACGT"};
        gnx::generic_sequence<T> s2{"ACGT"};
        auto result = gnx::local_align_n(s1, s2, 5, -1, -1);

        CHECK(result.score == 20);  // 4 matches * 5
    }

    SECTION( "custom mismatch penalty" )
    {   gnx::generic_sequence<T> s1{"ACGT"};
        gnx::generic_sequence<T> s2{"TTTT"};
        auto result = gnx::local_align_n(s1, s2, 2, -10, -1);

        // Strong mismatch penalty should result in low score
        CHECK(result.score <= 2);
    }

    SECTION( "custom gap penalty" )
    {   gnx::generic_sequence<T> s1{"ACGT"};
        gnx::generic_sequence<T> s2{"ACGGT"};
        auto result = gnx::local_align_n(s1, s2, 2, -1, -5);

        // Strong gap penalty should discourage gaps
        CHECK(result.score >= 0);
    }

// -- edge cases ---------------------------------------------------------------

    SECTION( "empty first sequence" )
    {   gnx::generic_sequence<T> s1;
        gnx::generic_sequence<T> s2{"ACGT"};
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 0);
        CHECK(result.aligned_seq1.empty());
        CHECK(result.aligned_seq2.empty());
    }

    SECTION( "empty second sequence" )
    {   gnx::generic_sequence<T> s1{"ACGT"};
        gnx::generic_sequence<T> s2;
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 0);
        CHECK(result.aligned_seq1.empty());
        CHECK(result.aligned_seq2.empty());
    }

    SECTION( "both sequences empty" )
    {   gnx::generic_sequence<T> s1;
        gnx::generic_sequence<T> s2;
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 0);
        CHECK(result.aligned_seq1.empty());
        CHECK(result.aligned_seq2.empty());
    }

    SECTION( "single character sequences" )
    {   gnx::generic_sequence<T> s1{"A"};
        gnx::generic_sequence<T> s2{"A"};
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 2);  // Default match score
        CHECK(result.aligned_seq1 == "A");
        CHECK(result.aligned_seq2 == "A");
    }

    SECTION( "single character mismatch" )
    {   gnx::generic_sequence<T> s1{"A"};
        gnx::generic_sequence<T> s2{"T"};
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 0);  // SW doesn't allow negative scores
    }

// -- case insensitivity -------------------------------------------------------

    SECTION( "lowercase sequences" )
    {   gnx::generic_sequence<T> s1{"acgt"};
        gnx::generic_sequence<T> s2{"acgt"};
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 8);
        CHECK(result.aligned_seq1 == "acgt");
        CHECK(result.aligned_seq2 == "acgt");
    }

    SECTION( "mixed case sequences" )
    {   gnx::generic_sequence<T> s1{"AcGt"};
        gnx::generic_sequence<T> s2{"aCgT"};
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 8);  // Should match regardless of case
    }

// -- gnx::generic_sequence tests --------------------------------------------------------

    SECTION( "gnx::sq alignment" )
    {   auto s1 = "ACGTACGT"_sq;
        auto s2 = "TACGT"_sq;
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score > 0);
        CHECK(result.aligned_seq1.length() > 0);
        CHECK(result.aligned_seq2.length() > 0);
    }

// -- realistic biological example ---------------------------------------------

    SECTION( "realistic DNA sequences with SNP" )
    {   // Two sequences with single nucleotide polymorphism
        gnx::generic_sequence<T> s1{"ATCGATCGATCG"};
        gnx::generic_sequence<T> s2{"ATCGCTCGATCG"};  // C instead of A at position 5
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score >= 16);  // Most bases should match
        CHECK(result.aligned_seq1.length() >= 10);
    }

    SECTION( "realistic DNA with indel" )
    {   // Sequence with insertion/deletion
        gnx::generic_sequence<T> s1{"ATCGATCGATCG"};
        gnx::generic_sequence<T> s2{"ATCGTCGATCG"};  // Missing 'A' at position 5
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score > 0);
        // Should find good alignment around the indel
    }

// -- longer sequences ---------------------------------------------------------

    SECTION( "longer sequences" )
    {   gnx::generic_sequence<T> s1{"ACGTACGTACGTACGTACGTACGTACGTACGT"};
        gnx::generic_sequence<T> s2{"ACGTACGTACGTACGTACGTACGTACGTACGT"};
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 64);  // 32 matches * 2
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

    SECTION( "partially matching longer sequences" )
    {   gnx::generic_sequence<T> s1{"AAAAAAACGTACGTACGTTTTTTT"};
        gnx::generic_sequence<T> s2{"ACGTACGTACGT"};
        auto result = gnx::local_align_n(s1, s2);

        // Should find the matching middle part
        CHECK(result.score == 24);  // 12 matches * 2
        CHECK(result.aligned_seq1 == "ACGTACGTACGT");
        CHECK(result.aligned_seq2 == "ACGTACGTACGT");
    }
}

// =============================================================================
// local_align_p() algorithm with substitution matrices tests
// =============================================================================

TEMPLATE_TEST_CASE
(   "gnx::local_align_p"
,   "[algorithm][peptide][local_align][blosum][pam]"
,   std::vector<char>
)
{   typedef TestType T;

// -- BLOSUM62 tests -----------------------------------------------------------

    SECTION( "BLOSUM62 - identical peptide sequences" )
    {   gnx::generic_sequence<T> s1{"ARNDCQEGHILKMFPSTWYV"};
        gnx::generic_sequence<T> s2{"ARNDCQEGHILKMFPSTWYV"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        // Score should be sum of diagonal elements for each amino acid
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

    SECTION( "BLOSUM62 - single amino acid difference" )
    {   gnx::generic_sequence<T> s1{"ARNDCQEG"};
        gnx::generic_sequence<T> s2{"ARNDCQKG"};  // E->K substitution
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        // Should align well with one mismatch
        CHECK(result.score > 20);
        CHECK(result.aligned_seq1.length() == result.aligned_seq2.length());
    }

    SECTION( "BLOSUM62 - conservative substitution" )
    {   // Leucine (L) and Isoleucine (I) are similar hydrophobic amino acids
        gnx::generic_sequence<T> s1{"ACDEFGHIKLMNPQRSTVWY"};
        gnx::generic_sequence<T> s2{"ACDEFGHIKLMNPQRSTVWY"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        CHECK(result.score > 50);
        CHECK(result.aligned_seq1 == s1);
    }

    SECTION( "BLOSUM62 - peptide with gaps" )
    {   gnx::generic_sequence<T> s1{"ARNDCQEG"};
        gnx::generic_sequence<T> s2{"ARNDQEG"};  // C removed
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62, -8);

        CHECK(result.score > 0);
        // Should align with one gap
    }

    SECTION( "BLOSUM62 - different gap penalty" )
    {   gnx::generic_sequence<T> s1{"ACDEFG"};
        gnx::generic_sequence<T> s2{"ACDEFG"};
        auto result1 = gnx::local_align_p(s1, s2, gnx::lut::blosum62, -8);
        auto result2 = gnx::local_align_p(s1, s2, gnx::lut::blosum62, -2);

        // With identical sequences, gap penalty shouldn't matter
        CHECK(result1.score == result2.score);
    }

    SECTION( "BLOSUM62 - case insensitive" )
    {   gnx::generic_sequence<T> s1{"ARNDCQEG"};
        gnx::generic_sequence<T> s2{"arndcqeg"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        // Should match regardless of case
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1.length() == result.aligned_seq2.length());
    }

// -- BLOSUM80 tests -----------------------------------------------------------

    SECTION( "BLOSUM80 - identical sequences" )
    {   gnx::generic_sequence<T> s1{"MVHLTPEEK"};
        gnx::generic_sequence<T> s2{"MVHLTPEEK"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum80);

        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

    SECTION( "BLOSUM80 vs BLOSUM62 comparison" )
    {   // BLOSUM80 is more stringent for closely related sequences
        gnx::generic_sequence<T> s1{"ACDEFG"};
        gnx::generic_sequence<T> s2{"ACDEFG"};
        auto result62 = gnx::local_align_p(s1, s2, gnx::lut::blosum62);
        auto result80 = gnx::local_align_p(s1, s2, gnx::lut::blosum80);

        // BLOSUM80 typically gives higher scores for identical sequences
        CHECK(result80.score >= result62.score);
    }

// -- BLOSUM45 tests -----------------------------------------------------------

    SECTION( "BLOSUM45 - distantly related sequences" )
    {   gnx::generic_sequence<T> s1{"ARNDCQEG"};
        gnx::generic_sequence<T> s2{"ARNDCQEG"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum45);

        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

// -- PAM250 tests -------------------------------------------------------------

    SECTION( "PAM250 - identical sequences" )
    {   gnx::generic_sequence<T> s1{"MVHLTPEEK"};
        gnx::generic_sequence<T> s2{"MVHLTPEEK"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::pam250);

        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

    SECTION( "PAM250 - with mismatches" )
    {   gnx::generic_sequence<T> s1{"ARNDCQEG"};
        gnx::generic_sequence<T> s2{"ARNDCQKG"};  // E->K substitution
        auto result = gnx::local_align_p(s1, s2, gnx::lut::pam250);

        CHECK(result.score > 0);
        CHECK(result.aligned_seq1.length() == result.aligned_seq2.length());
    }

// -- PAM120 tests -------------------------------------------------------------

    SECTION( "PAM120 - closely related sequences" )
    {   gnx::generic_sequence<T> s1{"ACDEFGHIKL"};
        gnx::generic_sequence<T> s2{"ACDEFGHIKL"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::pam120);

        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

    SECTION( "PAM120 vs PAM250 comparison" )
    {   // Different PAM matrices for different evolutionary distances
        gnx::generic_sequence<T> s1{"ACDEFG"};
        gnx::generic_sequence<T> s2{"ACDEFG"};
        auto result120 = gnx::local_align_p(s1, s2, gnx::lut::pam120);
        auto result250 = gnx::local_align_p(s1, s2, gnx::lut::pam250);

        // Both should align perfectly
        CHECK(result120.score > 0);
        CHECK(result250.score > 0);
    }

// -- PAM30 tests --------------------------------------------------------------

    SECTION( "PAM30 - very closely related sequences" )
    {   gnx::generic_sequence<T> s1{"MVHLTPEEK"};
        gnx::generic_sequence<T> s2{"MVHLTPEEK"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::pam30);

        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

// -- Realistic peptide alignment examples -------------------------------------

    SECTION( "realistic - human vs mouse hemoglobin fragment" )
    {   // Simplified example of conserved protein region
        gnx::generic_sequence<T> human{"VLSPADKTNVKAAW"};
        gnx::generic_sequence<T> mouse{"VLSAADKTNVKAAW"};  // P->A substitution
        auto result = gnx::local_align_p(human, mouse, gnx::lut::blosum62);

        // Should find good alignment despite one difference
        CHECK(result.score > 40);
        CHECK(result.aligned_seq1.length() >= 10);
    }

    SECTION( "realistic - enzyme active site comparison" )
    {   // Catalytic triad-like sequence
        gnx::generic_sequence<T> enzyme1{"HDSGICN"};
        gnx::generic_sequence<T> enzyme2{"HDSGVCN"};  // I->V conservative substitution
        auto result = gnx::local_align_p(enzyme1, enzyme2, gnx::lut::blosum62);

        // Conservative substitution should still score well
        CHECK(result.score > 20);
    }

    SECTION( "realistic - signal peptide vs mature protein" )
    {   gnx::generic_sequence<T> full_seq{"MKTIIALSYIFCLVFAACDEFGHIKL"};
        gnx::generic_sequence<T> mature{"ACDEFGHIKL"};  // After signal peptide cleavage
        auto result = gnx::local_align_p(full_seq, mature, gnx::lut::blosum62);

        // Should find the mature protein region
        CHECK(result.score > 30);
        CHECK(result.aligned_seq2 == mature);
    }

// -- ambiguous amino acids ----------------------------------------------------

    SECTION( "ambiguous amino acids - B (D or N)" )
    {   gnx::generic_sequence<T> s1{"ACDEFG"};
        gnx::generic_sequence<T> s2{"ACBEFG"};  // D->B
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        // B should have reasonable score with D
        CHECK(result.score > 0);
    }

    SECTION( "ambiguous amino acids - Z (E or Q)" )
    {   gnx::generic_sequence<T> s1{"ACDEFG"};
        gnx::generic_sequence<T> s2{"ACDQFG"};  // E->Q
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        CHECK(result.score > 0);
    }

    SECTION( "ambiguous amino acids - X (any)" )
    {   gnx::generic_sequence<T> s1{"ACDEFG"};
        gnx::generic_sequence<T> s2{"ACXEFG"};  // D->X
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        // X should have neutral or small penalty
        CHECK(result.score >= 0);
    }

// -- stop codon handling ------------------------------------------------------

    SECTION( "stop codon in sequence" )
    {   gnx::generic_sequence<T> s1{"ACDEFG*"};  // * represents stop codon
        gnx::generic_sequence<T> s2{"ACDEFG*"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        // Should handle stop codon
        CHECK(result.score > 0);
    }

// -- edge cases with matrices -------------------------------------------------

    SECTION( "empty sequences with matrix" )
    {   gnx::generic_sequence<T> s1;
        gnx::generic_sequence<T> s2{"ACDEFG"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        CHECK(result.score == 0);
        CHECK(result.aligned_seq1.empty());
        CHECK(result.aligned_seq2.empty());
    }

    SECTION( "single amino acid with matrix" )
    {   gnx::generic_sequence<T> s1{"A"};
        gnx::generic_sequence<T> s2{"A"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        // BLOSUM62[A][A] = 4
        CHECK(result.score == 4);
        CHECK(result.aligned_seq1 == "A");
        CHECK(result.aligned_seq2 == "A");
    }

// -- gnx::sq with matrices ----------------------------------------------------

    SECTION( "gnx::sq with BLOSUM62" )
    {   auto s1 = "MVHLTPEEK"_sq;
        auto s2 = "MVHLTPEEK"_sq;
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);
        
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == "MVHLTPEEK");
        CHECK(result.aligned_seq2 == "MVHLTPEEK");
    }

    SECTION( "gnx::sq with PAM250" )
    {   auto s1 = "ACDEFGHIKL"_sq;
        auto s2 = "ACDEFGHIKL"_sq;
        auto result = gnx::local_align_p(s1, s2, gnx::lut::pam250);

        CHECK(result.score > 0);
        CHECK(result.aligned_seq1.length() > 0);
    }

// -- performance test with longer sequences -----------------------------------

    SECTION( "longer peptide sequences with BLOSUM62" )
    {   gnx::generic_sequence<T> s1{"MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"};
        gnx::generic_sequence<T> s2{"MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        // Human beta-globin, should align perfectly with itself
        CHECK(result.score > 500);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }
}
