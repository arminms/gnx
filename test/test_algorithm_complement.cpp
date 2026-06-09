// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#include <catch2/catch_all.hpp>

#include <gnx/sq.hpp>
#include <gnx/psq.hpp>
#include <gnx/algorithms/complement.hpp>
#include <gnx/algorithms/random.hpp>

const uint64_t seed_pi{3141592654};

// =============================================================================
// complement() algorithm tests
// =============================================================================

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::complement"
,   "[algorithm][complement][cuda]"
,   std::vector<char>
,   thrust::host_vector<char>
,   thrust::device_vector<char>
)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::complement"
,   "[algorithm][complement][rocm]"
,   std::vector<char>
,   thrust::host_vector<char>
,   thrust::device_vector<char>
,   gnx::unified_vector<char>
)
#else
TEMPLATE_TEST_CASE( "gnx::complement", "[algorithm][complement]", std::vector<char>)
#endif
{   typedef TestType T;

    // Test data
    gnx::generic_sequence<T> s1("ACGT");
    gnx::generic_sequence<T> s2("acgt");
    gnx::generic_sequence<T> s3("ACGTACGT");

// -- basic Watson-Crick complementation ---------------------------------------

    SECTION( "complement of DNA bases uppercase" )
    {   gnx::generic_sequence<T> s("ACGT");
        gnx::complement(s);
        CHECK(s == "TGCA");
    }

    SECTION( "complement of DNA bases lowercase" )
    {   gnx::generic_sequence<T> s("acgt");
        gnx::complement(s);
        CHECK(s == "tgca");
    }

    SECTION( "complement preserves case" )
    {   gnx::generic_sequence<T> mixed("AcGt");
        gnx::complement(mixed);
        CHECK(mixed == "TgCa");
    }

    SECTION( "complement is involutory (double complement = identity)" )
    {   gnx::generic_sequence<T> s("ACGTACGT");
        auto original = s;
        gnx::complement(s);
        gnx::complement(s);
        CHECK(s == original);
    }

// -- RNA complementation ------------------------------------------------------

    SECTION( "complement of RNA sequence" )
    {   gnx::generic_sequence<T> rna("ACGU");
        gnx::complement(rna);
        CHECK(rna == "TGCA");  // U -> A
    }

    SECTION( "complement of lowercase RNA" )
    {   gnx::generic_sequence<T> rna("acgu");
        gnx::complement(rna);
        CHECK(rna == "tgca");
    }

// -- IUPAC ambiguity codes ----------------------------------------------------

    SECTION( "complement of IUPAC ambiguity codes" )
    {   // Let's verify specific transformations
        gnx::generic_sequence<T> r("R"); gnx::complement(r); CHECK(r == "Y");
        gnx::generic_sequence<T> y("Y"); gnx::complement(y); CHECK(y == "R");
        gnx::generic_sequence<T> m("M"); gnx::complement(m); CHECK(m == "K");
        gnx::generic_sequence<T> k("K"); gnx::complement(k); CHECK(k == "M");
        gnx::generic_sequence<T> s("S"); gnx::complement(s); CHECK(s == "S");
        gnx::generic_sequence<T> w("W"); gnx::complement(w); CHECK(w == "W");
        gnx::generic_sequence<T> b("B"); gnx::complement(b); CHECK(b == "V");
        gnx::generic_sequence<T> d("D"); gnx::complement(d); CHECK(d == "H");
        gnx::generic_sequence<T> h("H"); gnx::complement(h); CHECK(h == "D");
        gnx::generic_sequence<T> v("V"); gnx::complement(v); CHECK(v == "B");
        gnx::generic_sequence<T> n("N"); gnx::complement(n); CHECK(n == "N");
    }

    SECTION( "complement of lowercase IUPAC codes" )
    {   gnx::generic_sequence<T> r("r"); gnx::complement(r); CHECK(r == "y");
        gnx::generic_sequence<T> y("y"); gnx::complement(y); CHECK(y == "r");
        gnx::generic_sequence<T> m("m"); gnx::complement(m); CHECK(m == "k");
        gnx::generic_sequence<T> k("k"); gnx::complement(k); CHECK(k == "m");
    }

// -- iterator-based complementation -------------------------------------------

    SECTION( "complement with iterators" )
    {   gnx::generic_sequence<T> s("ACGTACGT");
        gnx::complement(s.begin(), s.end());
        CHECK(s == "TGCATGCA");
    }

    SECTION( "complement partial range" )
    {   gnx::generic_sequence<T> s("ACGTACGT");
        gnx::complement(s.begin(), s.begin() + 4);
        CHECK(s == "TGCAACGT");
    }

// -- empty and single character -----------------------------------------------

    SECTION( "complement of empty sequence" )
    {   T empty;
        gnx::complement(empty);
        CHECK(empty.empty());
    }

    SECTION( "complement of single base" )
    {   gnx::generic_sequence<T> a("A");
        gnx::complement(a);
        CHECK(a == "T");
        
        gnx::generic_sequence<T> c("C");
        gnx::complement(c);
        CHECK(c == "G");
    }

// -- non-nucleotide characters ------------------------------------------------

    SECTION( "complement leaves non-nucleotide characters unchanged" )
    {   gnx::generic_sequence<T> s("ACGTXacgtx123");
        gnx::complement(s);
        // A->T, C->G, G->C, T->A, X->X (unchanged), etc.
        CHECK(s[0] == 'T');
        CHECK(s[1] == 'G');
        CHECK(s[2] == 'C');
        CHECK(s[3] == 'A');
        CHECK(s[4] == 'X');  // X unchanged
        CHECK(s[5] == 't');
        CHECK(s[6] == 'g');
        CHECK(s[7] == 'c');
        CHECK(s[8] == 'a');
        CHECK(s[9] == 'x');  // x unchanged
        CHECK(s[10] == '1'); // 1 unchanged
        CHECK(s[11] == '2'); // 2 unchanged
        CHECK(s[12] == '3'); // 3 unchanged
    }
}

// =============================================================================
// complement() algorithm execution policy tests
// =============================================================================

TEMPLATE_TEST_CASE
(   "gnx::complement execution policies"
,   "[algorithm][complement][policy]"
,   std::vector<char>
)
{   typedef TestType T;

    using gnx::execution::seq;
    using gnx::execution::par;
    using gnx::execution::unseq;
    using gnx::execution::par_unseq;

    const auto N{10'000};

    gnx::generic_sequence<T> original(N);
    gnx::rand(original.begin(), N, "ACGT", seed_pi);

// -- sequential policy --------------------------------------------------------

    SECTION( "complement with seq policy" )
    {   gnx::generic_sequence<T> s(original);
        gnx::complement(seq, s);
        // Verify all bases were complemented correctly
        for (std::size_t i = 0; i < N; ++i)
        {   char orig = original[i];
            char comp = s[i];
            if (orig == 'A') CHECK(comp == 'T');
            else if (orig == 'T') CHECK(comp == 'A');
            else if (orig == 'C') CHECK(comp == 'G');
            else if (orig == 'G') CHECK(comp == 'C');
        }
    }

// -- unsequenced policy -------------------------------------------------------

    SECTION( "complement with unseq policy" )
    {   gnx::generic_sequence<T> s(original);
        gnx::complement(unseq, s);
        // Double complement should restore original
        gnx::complement(unseq, s);
        CHECK(s == original);
    }

// -- parallel policy ----------------------------------------------------------

    SECTION( "complement with par policy" )
    {   gnx::generic_sequence<T> s(original);
        gnx::complement(par, s);
        // Double complement should restore original
        gnx::complement(par, s);
        CHECK(s == original);
    }

// -- parallel unsequenced policy ----------------------------------------------

    SECTION( "complement with par_unseq policy" )
    {   gnx::generic_sequence<T> s(original);
        gnx::complement(par_unseq, s);
        // Double complement should restore original
        gnx::complement(par_unseq, s);
        CHECK(s == original);
    }

// -- consistency across policies ----------------------------------------------

    SECTION( "all policies produce same result" )
    {   gnx::generic_sequence<T> s_seq(original);
        gnx::generic_sequence<T> s_unseq(original);
        gnx::generic_sequence<T> s_par(original);
        gnx::generic_sequence<T> s_par_unseq(original);

        gnx::complement(seq, s_seq);
        gnx::complement(unseq, s_unseq);
        gnx::complement(par, s_par);
        gnx::complement(par_unseq, s_par_unseq);

        CHECK(s_seq == s_unseq);
        CHECK(s_seq == s_par);
        CHECK(s_seq == s_par_unseq);
    }

// -- large sequence test ------------------------------------------------------

    SECTION( "complement large sequence with policies" )
    {   const auto large_N{100'000};
        gnx::generic_sequence<T> large_seq(large_N);
        gnx::rand(large_seq.begin(), large_N, "ACGT", seed_pi);

        auto original_copy = large_seq;

        // Test each policy on large sequence
        gnx::complement(seq, large_seq);
        gnx::complement(seq, large_seq);
        CHECK(large_seq == original_copy);

        gnx::complement(par, large_seq);
        gnx::complement(par, large_seq);
        CHECK(large_seq == original_copy);
    }
}

// =============================================================================
// complement() algorithm device tests
// =============================================================================

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::complement::device"
,   "[algorithm][complement][cuda]"
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
)
{   typedef TestType T;

    const auto N{10'000};

    gnx::generic_sequence<T> s(N);
    gnx::rand(s.begin(), N, "ACGT", seed_pi);

    SECTION( "device vector complement" )
    {   auto original = s;
        gnx::complement(thrust::cuda::par, s);
        gnx::complement(thrust::cuda::par, s);
        CHECK(s == original);
    }

    SECTION( "cuda stream complement" )
    {   cudaStream_t streamA;
        cudaStreamCreate(&streamA);
        
        auto original = s;
        gnx::complement(thrust::cuda::par.on(streamA), s);
        cudaStreamSynchronize(streamA);
        gnx::complement(thrust::cuda::par_nosync.on(streamA), s);
        cudaStreamSynchronize(streamA);
        
        CHECK(s == original);
        
        cudaStreamDestroy(streamA);
    }
}
#endif //__CUDACC__

#if defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::complement::device"
,   "[algorithm][complement][rocm]"
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
,   gnx::unified_vector<char>
)
{   typedef TestType T;

    const auto N{10'000};

    gnx::generic_sequence<T> s(N);
    gnx::rand(s.begin(), N, "ACGT", seed_pi);

    SECTION( "device vector complement" )
    {   auto original = s;
        gnx::complement(thrust::hip::par, s);
        gnx::complement(thrust::hip::par, s);
        CHECK(s == original);
    }

    SECTION( "hip stream complement" )
    {   hipStream_t streamA;
        hipStreamCreate(&streamA);
        
        auto original = s;
        gnx::complement(thrust::hip::par.on(streamA), s);
        hipStreamSynchronize(streamA);
        gnx::complement(thrust::hip::par_nosync.on(streamA), s);
        hipStreamSynchronize(streamA);
        
        CHECK(s == original);
        
        hipStreamDestroy(streamA);
    }
}
#endif //__HIPCC__
