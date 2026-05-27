// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#include <catch2/catch_all.hpp>

#include <string>
#include <vector>

#include <gnx/sq.hpp>
#include <gnx/psq.hpp>
#include <gnx/algorithms/translate.hpp>

// Codon reference (NCBI standard genetic code, table 1):
//   TTT=F  TTC=F  TTA=L  TTG=L
//   TCT=S  TCC=S  TCA=S  TCG=S
//   TAT=Y  TAC=Y  TAA=*  TAG=*
//   TGT=C  TGC=C  TGA=*  TGG=W
//   CTT=L  CTC=L  CTA=L  CTG=L
//   CCT=P  CCC=P  CCA=P  CCG=P
//   CAT=H  CAC=H  CAA=Q  CAG=Q
//   CGT=R  CGC=R  CGA=R  CGG=R
//   ATT=I  ATC=I  ATA=I  ATG=M (start)
//   ACT=T  ACC=T  ACA=T  ACG=T
//   AAT=N  AAC=N  AAA=K  AAG=K
//   AGT=S  AGC=S  AGA=R  AGG=R
//   GTT=V  GTC=V  GTA=V  GTG=V
//   GCT=A  GCC=A  GCA=A  GCG=A
//   GAT=D  GAC=D  GAA=E  GAG=E
//   GGT=G  GGC=G  GGA=G  GGG=G

// =============================================================================
// gnx::translate() algorithm tests – gnx::generic_sequence<T>
// =============================================================================

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::translate"
,   "[algorithm][translate][cuda]"
,   std::vector<char>
,   thrust::host_vector<char>
)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::translate"
,   "[algorithm][translate][rocm]"
,   std::vector<char>
,   thrust::host_vector<char>
)
#else
TEMPLATE_TEST_CASE( "gnx::translate", "[algorithm][translate]", std::vector<char>)
#endif
{   typedef TestType T;

// -- empty / trivial ----------------------------------------------------------

    SECTION( "empty sequence yields empty protein" )
    {   gnx::generic_sequence<T> dna("");
        std::string protein;
(void) gnx::translate(dna.begin(), dna.end(), std::back_inserter(protein));
        CHECK(protein.empty());
    }

    SECTION( "sequence shorter than one codon is silently ignored" )
    {   gnx::generic_sequence<T> dna("AT");
        std::string protein;
(void) gnx::translate(dna.begin(), dna.end(), std::back_inserter(protein));
        CHECK(protein.empty());
    }

// -- single codons ------------------------------------------------------------

    SECTION( "ATG encodes Met (start codon)" )
    {   gnx::generic_sequence<T> dna("ATG");
        std::string protein;
(void) gnx::translate(dna.begin(), dna.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 1);
        CHECK(protein[0] == 'M');
    }

    SECTION( "TAA encodes stop (*)" )
    {   gnx::generic_sequence<T> dna("TAA");
        std::string protein;
(void) gnx::translate(dna.begin(), dna.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 1);
        CHECK(protein[0] == '*');
    }

    SECTION( "TAG encodes stop (*)" )
    {   gnx::generic_sequence<T> dna("TAG");
        std::string protein;
(void) gnx::translate(dna.begin(), dna.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 1);
        CHECK(protein[0] == '*');
    }

    SECTION( "TGA encodes stop (*)" )
    {   gnx::generic_sequence<T> dna("TGA");
        std::string protein;
(void) gnx::translate(dna.begin(), dna.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 1);
        CHECK(protein[0] == '*');
    }

    SECTION( "TGG encodes Trp (W)" )
    {   gnx::generic_sequence<T> dna("TGG");
        std::string protein;
(void) gnx::translate(dna.begin(), dna.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 1);
        CHECK(protein[0] == 'W');
    }

    SECTION( "NNN (invalid bases) yields X" )
    {   gnx::generic_sequence<T> dna("NNN");
        std::string protein;
(void) gnx::translate(dna.begin(), dna.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 1);
        CHECK(protein[0] == 'X');
    }

// -- multi-codon sequences ----------------------------------------------------

    SECTION( "ATG-GCT-AGT-ACT-TAA encodes MAST*" )
    {   gnx::generic_sequence<T> dna("ATGGCTAGTACTTAA");
        std::string protein;
(void) gnx::translate(dna.begin(), dna.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 5);
        CHECK(protein == "MAST*");
    }

    SECTION( "all three stop codons in a row" )
    {   gnx::generic_sequence<T> stops("TAATAGTGA");
        std::string protein;
(void) gnx::translate(stops.begin(), stops.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 3);
        CHECK(protein == "***");
    }

    SECTION( "TTT-CTT-ATT-GTT encodes FLIV" )
    {   gnx::generic_sequence<T> dna("TTTCTTATTGTT");
        std::string protein;
(void) gnx::translate(dna.begin(), dna.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 4);
        CHECK(protein == "FLIV");
    }

    SECTION( "CAT-CCT-CGT-CAA encodes HPRQ" )
    {   gnx::generic_sequence<T> dna("CATCCTCGTCAA");
        std::string protein;
(void) gnx::translate(dna.begin(), dna.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 4);
        CHECK(protein == "HPRQ");
    }

    SECTION( "GAT-GAA-GGT-GCT encodes DEGA" )
    {   gnx::generic_sequence<T> dna("GATGAAGGCGCT");
        std::string protein;
(void) gnx::translate(dna.begin(), dna.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 4);
        CHECK(protein == "DEGA");
    }

// -- partial trailing codon ---------------------------------------------------

    SECTION( "partial trailing codon (5 bases) produces 1 amino acid" )
    {   gnx::generic_sequence<T> dna("ATGAC");  // ATG=M + AC (ignored)
        std::string protein;
(void) gnx::translate(dna.begin(), dna.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 1);
        CHECK(protein[0] == 'M');
    }

    SECTION( "partial trailing codon (7 bases) produces 2 amino acids" )
    {   gnx::generic_sequence<T> dna("ATGGCTA");  // ATG=M + GCT=A + A (ignored)
        std::string protein;
(void) gnx::translate(dna.begin(), dna.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 2);
        CHECK(protein == "MA");
    }

// -- case-insensitive input ---------------------------------------------------

    SECTION( "lowercase DNA input is supported" )
    {   gnx::generic_sequence<T> dna("atggctagtacttaa");
        std::string protein;
(void) gnx::translate(dna.begin(), dna.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 5);
        CHECK(protein == "MAST*");
    }

// -- RNA input ----------------------------------------------------------------

    SECTION( "RNA input (U in place of T) is supported" )
    {   gnx::generic_sequence<T> rna("AUGGCUAGUACUUAA");
        std::string protein;
(void) gnx::translate(rna.begin(), rna.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 5);
        CHECK(protein == "MAST*");
    }

    SECTION( "lowercase RNA input is supported" )
    {   gnx::generic_sequence<T> rna("auggcuaguacuuaa");
        std::string protein;
(void) gnx::translate(rna.begin(), rna.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 5);
        CHECK(protein == "MAST*");
    }

// -- range overload -----------------------------------------------------------

    SECTION( "range overload (seq, out)" )
    {   gnx::generic_sequence<T> dna("ATGGCTTAA");
        std::string protein(3, '\0');
(void) gnx::translate(dna, protein);
        CHECK(protein == "MA*");
    }
}

// =============================================================================
// gnx::translate() execution policy tests
// =============================================================================

TEMPLATE_TEST_CASE
(   "gnx::translate execution policies"
,   "[algorithm][translate][policy]"
,   std::vector<char>
)
{   typedef TestType T;

    using gnx::execution::seq;
    using gnx::execution::par;
    using gnx::execution::unseq;
    using gnx::execution::par_unseq;

    // Reference: ATGGCTAGTACTTAA → MAST*
    const std::string input_str = "ATGGCTAGTACTTAA";
    gnx::generic_sequence<T> dna(input_str);
    const std::string expected  = "MAST*";

// -- sequential policy --------------------------------------------------------

    SECTION( "translate with seq policy" )
    {   std::string protein(5, '\0');
(void) gnx::translate(seq, dna.begin(), dna.end(), protein.begin());
        CHECK(protein == expected);
    }

// -- unsequenced policy -------------------------------------------------------

    SECTION( "translate with unseq policy" )
    {   std::string protein(5, '\0');
(void) gnx::translate(unseq, dna.begin(), dna.end(), protein.begin());
        CHECK(protein == expected);
    }

// -- parallel policy ----------------------------------------------------------

    SECTION( "translate with par policy" )
    {   std::string protein(5, '\0');
(void) gnx::translate(par, dna.begin(), dna.end(), protein.begin());
        CHECK(protein == expected);
    }

// -- parallel-unsequenced policy ----------------------------------------------

    SECTION( "translate with par_unseq policy" )
    {   std::string protein(5, '\0');
(void) gnx::translate(par_unseq, dna.begin(), dna.end(), protein.begin());
        CHECK(protein == expected);
    }

// -- consistency across policies ----------------------------------------------

    SECTION( "all policies produce identical results" )
    {   const int N = 10'000;
        // Build a large DNA string (multiples of 3: ATG = M repeated)
        std::string big_dna_str;
        big_dna_str.reserve(N * 3);
        const char* codons[] = { "ATG", "GCT", "TAT", "GTT", "TAA" };
        for (int i = 0; i < N; ++i)
            big_dna_str += codons[i % 5];

        gnx::generic_sequence<T> big_dna(big_dna_str);
        const std::size_t ncodons = big_dna.size() / 3;

        std::string p_seq(ncodons, '\0');
        std::string p_unseq(ncodons, '\0');
        std::string p_par(ncodons, '\0');
        std::string p_par_unseq(ncodons, '\0');

(void) gnx::translate(seq,       big_dna.begin(), big_dna.end(), p_seq.begin());
(void) gnx::translate(unseq,     big_dna.begin(), big_dna.end(), p_unseq.begin());
(void) gnx::translate(par,       big_dna.begin(), big_dna.end(), p_par.begin());
(void) gnx::translate(par_unseq, big_dna.begin(), big_dna.end(), p_par_unseq.begin());

        CHECK(p_seq == p_unseq);
        CHECK(p_seq == p_par);
        CHECK(p_seq == p_par_unseq);
    }

// -- range overload with policies ---------------------------------------------

    SECTION( "range overload with par policy" )
    {   std::string protein(5, '\0');
(void) gnx::translate(par, dna, protein);
        CHECK(protein == expected);
    }
}

// =============================================================================
// gnx::translate() algorithm tests – gnx::packed_generic_sequence_2bit<T>
//
// psq2 only stores ACGT (2-bit encoding). Translate is applied after converting
// to generic_sequence via to_sq(), which is the idiomatic way to translate a
// packed sequence.
// =============================================================================

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::translate psq2"
,   "[algorithm][translate][psq2][cuda]"
,   std::vector<uint8_t>
,   thrust::universal_vector<uint8_t>
)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::translate psq2"
,   "[algorithm][translate][psq2][rocm]"
,   std::vector<uint8_t>
,   thrust::universal_vector<uint8_t>
,   gnx::unified_vector<uint8_t>
)
#else
TEMPLATE_TEST_CASE
(   "gnx::translate psq2"
,   "[algorithm][translate][psq2]"
,   std::vector<uint8_t>
)
#endif
{   typedef TestType T;
    typedef gnx::packed_generic_sequence_2bit<T> Psq;

// -- empty / trivial ----------------------------------------------------------

    SECTION( "empty psq2 yields empty protein" )
    {   Psq psq("");
        auto sq = psq.to_sq();
        std::string protein;
(void) gnx::translate(sq.begin(), sq.end(), std::back_inserter(protein));
        CHECK(protein.empty());
    }

    SECTION( "psq2 shorter than one codon is silently ignored" )
    {   Psq psq("AT");
        auto sq = psq.to_sq();
        std::string protein;
(void) gnx::translate(sq.begin(), sq.end(), std::back_inserter(protein));
        CHECK(protein.empty());
    }

// -- single codons ------------------------------------------------------------

    SECTION( "ATG encodes Met" )
    {   Psq psq("ATG");
        auto sq = psq.to_sq();
        std::string protein;
(void) gnx::translate(sq.begin(), sq.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 1);
        CHECK(protein[0] == 'M');
    }

    SECTION( "TAA encodes stop (*)" )
    {   Psq psq("TAA");
        auto sq = psq.to_sq();
        std::string protein;
(void) gnx::translate(sq.begin(), sq.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 1);
        CHECK(protein[0] == '*');
    }

    SECTION( "TGA encodes stop (*)" )
    {   Psq psq("TGA");
        auto sq = psq.to_sq();
        std::string protein;
(void) gnx::translate(sq.begin(), sq.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 1);
        CHECK(protein[0] == '*');
    }

    SECTION( "TAG encodes stop (*)" )
    {   Psq psq("TAG");
        auto sq = psq.to_sq();
        std::string protein;
(void) gnx::translate(sq.begin(), sq.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 1);
        CHECK(protein[0] == '*');
    }

    SECTION( "TGG encodes Trp (W)" )
    {   Psq psq("TGG");
        auto sq = psq.to_sq();
        std::string protein;
(void) gnx::translate(sq.begin(), sq.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 1);
        CHECK(protein[0] == 'W');
    }

// -- multi-codon sequences ----------------------------------------------------

    SECTION( "ATG-GCT-AGT-ACT-TAA encodes MAST*" )
    {   Psq psq("ATGGCTAGTACTTAA");
        auto sq = psq.to_sq();
        std::string protein;
(void) gnx::translate(sq.begin(), sq.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 5);
        CHECK(protein == "MAST*");
    }

    SECTION( "all three stop codons in a row" )
    {   Psq psq("TAATAGTGA");
        auto sq = psq.to_sq();
        std::string protein;
(void) gnx::translate(sq.begin(), sq.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 3);
        CHECK(protein == "***");
    }

    SECTION( "TTT-CTT-ATT-GTT encodes FLIV" )
    {   Psq psq("TTTCTTATTGTT");
        auto sq = psq.to_sq();
        std::string protein;
(void) gnx::translate(sq.begin(), sq.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 4);
        CHECK(protein == "FLIV");
    }

    SECTION( "GAT-GAA-GGC-GCT encodes DEGA" )
    {   Psq psq("GATGAAGGCGCT");
        auto sq = psq.to_sq();
        std::string protein;
(void) gnx::translate(sq.begin(), sq.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 4);
        CHECK(protein == "DEGA");
    }

// -- partial trailing codon ---------------------------------------------------

    SECTION( "partial trailing codon (5 bases) produces 1 amino acid" )
    {   Psq psq("ATGAC");  // ATG=M + AC (ignored)
        auto sq = psq.to_sq();
        std::string protein;
(void) gnx::translate(sq.begin(), sq.end(), std::back_inserter(protein));
        REQUIRE(protein.size() == 1);
        CHECK(protein[0] == 'M');
    }

// -- round-trip: psq2 → sq → translate ----------------------------------------

    SECTION( "psq2 round-trip matches direct generic_sequence translate" )
    {   const std::string dna_str = "ATGGCTAGTACTTAA";
        Psq psq(dna_str);
        gnx::generic_sequence<std::vector<char>> sq_direct(dna_str);

        auto sq_from_psq = psq.to_sq();

        std::string protein_direct;
        std::string protein_psq;

(void) gnx::translate(sq_direct.begin(), sq_direct.end(),
                        std::back_inserter(protein_direct));
(void) gnx::translate(sq_from_psq.begin(), sq_from_psq.end(),
                        std::back_inserter(protein_psq));

        CHECK(protein_direct == protein_psq);
        CHECK(protein_psq == "MAST*");
    }

// -- range overload -----------------------------------------------------------

    SECTION( "range overload on converted psq2" )
    {   Psq psq("ATGGCTTAA");
        auto sq = psq.to_sq();
        std::string protein(3, '\0');
(void) gnx::translate(sq, protein);
        CHECK(protein == "MA*");
    }

// -- large sequence consistency -----------------------------------------------

    SECTION( "large psq2 sequence matches generic_sequence result" )
    {   const int N = 1000;
        std::string dna_str;
        dna_str.reserve(N * 3);
        const char* codons[] = { "ATG", "GCT", "TAT", "GTT", "TAA" };
        for (int i = 0; i < N; ++i)
            dna_str += codons[i % 5];

        Psq psq(dna_str);
        gnx::generic_sequence<std::vector<char>> sq_direct(dna_str);

        auto sq_from_psq = psq.to_sq();
        const std::size_t ncodons = dna_str.size() / 3;

        std::string p_direct(ncodons, '\0');
        std::string p_psq(ncodons, '\0');

(void) gnx::translate(sq_direct.begin(), sq_direct.end(), p_direct.begin());
(void) gnx::translate(sq_from_psq.begin(), sq_from_psq.end(), p_psq.begin());

        CHECK(p_direct == p_psq);
    }
}
