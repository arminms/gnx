// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <gnx/sq.hpp>
#include <gnx/psq.hpp>
#include <gnx/algorithms/count.hpp>
#include <gnx/algorithms/random.hpp>

const uint64_t seed_pi{3141592654};

// =============================================================================
// count() algorithm tests
// =============================================================================

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::count"
,   "[algorithm][count][cuda]"
,   std::vector<char>
,   thrust::host_vector<char>
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::count"
,   "[algorithm][count][rocm]"
,   std::vector<char>
,   thrust::host_vector<char>
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
,   gnx::unified_vector<char>
)
#else
TEMPLATE_TEST_CASE( "gnx::count", "[algorithm][count]", std::vector<char>)
#endif
{   typedef TestType T;

    // Test data
    gnx::generic_sequence<T> s1("ACGTacgt");
    gnx::generic_sequence<T> s2("AAAACCCCGGGGTTTT");
    gnx::generic_sequence<T> s3("AaCcGgTtNn");

// -- basic counting -----------------------------------------------------------

    SECTION( "count all bases in mixed-case sequence" )
    {   auto result = gnx::count(s1);
        CHECK(result['A'] == 2);
        CHECK(result['C'] == 2);
        CHECK(result['G'] == 2);
        CHECK(result['T'] == 2);
        CHECK(result.size() == 4);
    }

    SECTION( "count bases in homogeneous blocks" )
    {   auto result = gnx::count(s2);
        CHECK(result['A'] == 4);
        CHECK(result['C'] == 4);
        CHECK(result['G'] == 4);
        CHECK(result['T'] == 4);
        CHECK(result.size() == 4);
    }

    SECTION( "count with ambiguous nucleotides" )
    {   auto result = gnx::count(s3);
        CHECK(result['A'] == 2);
        CHECK(result['C'] == 2);
        CHECK(result['G'] == 2);
        CHECK(result['T'] == 2);
        CHECK(result['N'] == 2);
        CHECK(result.size() == 5);
    }

// -- case-insensitive counting ------------------------------------------------

    SECTION( "verify case-insensitive counting" )
    {   gnx::generic_sequence<T> lower("acgt");
        gnx::generic_sequence<T> upper("ACGT");
        
        auto result_lower = gnx::count(lower);
        auto result_upper = gnx::count(upper);
        
        CHECK(result_lower == result_upper);
        CHECK(result_lower['A'] == 1);
        CHECK(result_lower.count('a') == 0);  // lowercase not in map
    }

// -- iterator counting --------------------------------------------------------

    SECTION( "count with iterators" )
    {   auto result = gnx::count(s1.begin(), s1.end());
        CHECK(result['A'] == 2);
        CHECK(result['C'] == 2);
        CHECK(result['G'] == 2);
        CHECK(result['T'] == 2);
    }

// -- empty sequences ----------------------------------------------------------

    SECTION( "count empty sequence" )
    {   T empty;
        auto result = gnx::count(empty);
        CHECK(result.empty());
    }

// -- single character sequences -----------------------------------------------

    SECTION( "count single character" )
    {   gnx::generic_sequence<T> single_a("A");
        gnx::generic_sequence<T> single_lower("a");
        
        auto result1 = gnx::count(single_a);
        auto result2 = gnx::count(single_lower);
        
        CHECK(result1['A'] == 1);
        CHECK(result1.size() == 1);
        CHECK(result1 == result2);
    }

// -- amino acid sequences -----------------------------------------------------

    SECTION( "count amino acids" )
    {   gnx::generic_sequence<T> peptide("ARNDCQEGHILKMFPSTWYVarnDcqeghilkmfpstwyv");
        auto result = gnx::count(peptide);
        
        // Each standard amino acid should appear twice (upper and lower)
        CHECK(result['A'] == 2);
        CHECK(result['R'] == 2);
        CHECK(result['N'] == 2);
        CHECK(result['D'] == 2);
        CHECK(result.size() == 20);
    }
}

// =============================================================================
// count() algorithm execution policy tests
// =============================================================================

TEMPLATE_TEST_CASE
(   "gnx::count execution policies"
,   "[algorithm][count][policy]"
,   std::vector<char>
)
{   typedef TestType T;

    using gnx::execution::seq;
    using gnx::execution::par;
    using gnx::execution::unseq;
    using gnx::execution::par_unseq;

    const auto N{10'000};

    gnx::generic_sequence<T> s1(N);
    gnx::rand(s1.begin(), N, "ACGTacgt", seed_pi);

    // Get baseline result
    auto expected = gnx::count(s1);

// -- sequential policy --------------------------------------------------------

    SECTION( "count with seq policy" )
    {   auto result = gnx::count(seq, s1);
        CHECK(result == expected);
        // Verify case-insensitive counting
        CHECK(result['A'] > 0);
        CHECK(result.count('a') == 0);  // lowercase not in result
    }

// -- unsequenced policy -------------------------------------------------------

    SECTION( "count with unseq policy" )
    {   auto result = gnx::count(unseq, s1);
        CHECK(result == expected);
    }

// -- parallel policy ----------------------------------------------------------

    SECTION( "count with par policy" )
    {   auto result = gnx::count(par, s1);
        CHECK(result == expected);
    }

// -- parallel unsequenced policy ----------------------------------------------

    SECTION( "count with par_unseq policy" )
    {   auto result = gnx::count(par_unseq, s1);
        CHECK(result == expected);
    }

// -- verify total count matches length ----------------------------------------

    SECTION( "total count equals sequence length" )
    {   auto result = gnx::count(par, s1);
        std::size_t total = 0;
        for (const auto& [key, value] : result)
            total += value;
        CHECK(total == N);
    }
}

// =============================================================================
// count() algorithm device tests
// =============================================================================

#if defined(__CUDACC__) || defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::count::device"
,   "[algorithm][count][device]"
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
)
{   typedef TestType T;
    const auto N{10'000};

    gnx::generic_sequence<T> s(N); // device
    gnx::sq baseline(N); // host
    gnx::rand(s.begin(), N, "ACGTacgt", seed_pi);
    gnx::rand(baseline.begin(), N, "ACGTacgt", seed_pi);

    // Get baseline result
    auto expected = gnx::count(baseline);

#if defined(__CUDACC__)
    auto policy = thrust::cuda::par;
#else
    auto policy = thrust::hip::par;
#endif

// -- device comparison --------------------------------------------------------

    SECTION( "count on device" )
    {   auto result = gnx::count(policy, s);
        CHECK(result == expected);
    }

// -- streams on device --------------------------------------------------------

#if defined(__CUDACC__)
    SECTION( "CUDA stream as policy" )
    {   cudaStream_t stream; cudaStreamCreate(&stream);
        auto result = gnx::count(thrust::cuda::par.on(stream), s);
        CHECK(result == expected);
        cudaStreamSynchronize(stream); cudaStreamDestroy(stream);
    }
#else // __HIPCC__
    SECTION( "HIP stream as policy" )
    {   hipStream_t stream; hipStreamCreate(&stream);
        auto result = gnx::count(thrust::hip::par.on(stream), s);
        CHECK(result == expected);
        hipStreamSynchronize(stream); hipStreamDestroy(stream);
    }
#endif // __CUDACC__
}
#endif // __CUDACC__ || __HIPCC__

// =============================================================================
// count() k-mer counting tests
// =============================================================================

TEMPLATE_TEST_CASE
(   "gnx::count k-mers"
,   "[algorithm][count][kmer]"
,   std::vector<char>
)
{   typedef TestType T;

// -- basic k-mer counting -----------------------------------------------------

    SECTION( "count 2-mers (dinucleotides)" )
    {   gnx::generic_sequence<T> s("ACGTACGT");
        auto result = gnx::count(s, 2);

        CHECK(result["AC"] == 2);
        CHECK(result["CG"] == 2);
        CHECK(result["GT"] == 2);
        CHECK(result["TA"] == 1);
        CHECK(result.size() == 4);
    }

    SECTION( "count 3-mers (trinucleotides)" )
    {   gnx::generic_sequence<T> s("ACGTACGTACGT");
        auto result = gnx::count(s, 3);

        CHECK(result["ACG"] == 3);
        CHECK(result["CGT"] == 3);
        CHECK(result["GTA"] == 2);
        CHECK(result["TAC"] == 2);
        CHECK(result.size() == 4);
    }

    SECTION( "count 4-mers (tetranucleotides)" )
    {   gnx::generic_sequence<T> s("ACGTACGTACGT");
        auto result = gnx::count(s, 4);

        CHECK(result["ACGT"] == 3);
        CHECK(result["CGTA"] == 2);
        CHECK(result["GTAC"] == 2);
        CHECK(result["TACG"] == 2);
        CHECK(result.size() == 4);
    }

// -- case-insensitive k-mer counting ------------------------------------------

    SECTION( "k-mer counting is case-insensitive" )
    {   gnx::generic_sequence<T> lower("acgtacgt");
        gnx::generic_sequence<T> upper("ACGTACGT");
        gnx::generic_sequence<T> mixed("AcGtAcGt");

        auto result_lower = gnx::count(lower, 2);
        auto result_upper = gnx::count(upper, 2);
        auto result_mixed = gnx::count(mixed, 2);

        CHECK(result_lower == result_upper);
        CHECK(result_lower == result_mixed);
        CHECK(result_lower["AC"] == 2);
        CHECK(result_lower.count("ac") == 0);  // should be uppercase
    }

// -- edge cases ---------------------------------------------------------------

    SECTION( "empty sequence" )
    {   T empty;
        auto result = gnx::count(empty, 2);
        CHECK(result.empty());
    }

    SECTION( "sequence shorter than word_length" )
    {   gnx::generic_sequence<T> short_seq("ACG");
        auto result = gnx::count(short_seq, 5);
        CHECK(result.empty());
    }

    SECTION( "sequence equal to word_length" )
    {   gnx::generic_sequence<T> exact("ACGT");
        auto result = gnx::count(exact, 4);
        CHECK(result["ACGT"] == 1);
        CHECK(result.size() == 1);
    }

    SECTION( "single k-mer" )
    {   gnx::generic_sequence<T> single("AAA");
        auto result = gnx::count(single, 2);
        CHECK(result["AA"] == 2);
        CHECK(result.size() == 1);
    }

// -- iterator k-mer counting --------------------------------------------------

    SECTION( "count k-mers with iterators" )
    {   gnx::generic_sequence<T> s("ACGTACGT");
        auto result = gnx::count(s.begin(), s.end(), 2);
        
        CHECK(result["AC"] == 2);
        CHECK(result["CG"] == 2);
        CHECK(result["GT"] == 2);
        CHECK(result["TA"] == 1);
    }

// -- overlapping k-mers -------------------------------------------------------

    SECTION( "overlapping k-mers are counted" )
    {   gnx::generic_sequence<T> s("AAAA");
        auto result = gnx::count(s, 2);
        
        CHECK(result["AA"] == 3);  // AA at positions 0,1,2
        CHECK(result.size() == 1);
    }

    SECTION( "verify total k-mer count" )
    {   gnx::generic_sequence<T> s("ACGTACGT");  // 8 bases
        auto result = gnx::count(s, 3);  // 3-mers
        
        std::size_t total = 0;
        for (const auto& [kmer, count] : result)
            total += count;
        
        // For an n-length sequence, there are (n - k + 1) k-mers
        CHECK(total == 8 - 3 + 1);  // 6 total 3-mers
    }
}

// =============================================================================
// count() k-mer execution policy tests
// =============================================================================

TEMPLATE_TEST_CASE
(   "gnx::count k-mers execution policies"
,   "[algorithm][count][kmer][policy]"
,   std::vector<char>
)
{   typedef TestType T;

    using gnx::execution::seq;
    using gnx::execution::par;
    using gnx::execution::unseq;
    using gnx::execution::par_unseq;

    const auto N{10'000};

    gnx::generic_sequence<T> s1(N);
    gnx::rand(s1.begin(), N, "ACGTacgt", seed_pi);

    // Get baseline result
    auto expected = gnx::count(s1, 5);  // count 5-mers

// -- sequential policy --------------------------------------------------------

    SECTION( "count k-mers with seq policy" )
    {   auto result = gnx::count(seq, s1, 5);
        CHECK(result == expected);
    }

// -- unsequenced policy -------------------------------------------------------

    SECTION( "count k-mers with unseq policy" )
    {   auto result = gnx::count(unseq, s1, 5);
        CHECK(result == expected);
    }

// -- parallel policy ----------------------------------------------------------

    SECTION( "count k-mers with par policy" )
    {   auto result = gnx::count(par, s1, 5);
        CHECK(result == expected);
    }

// -- parallel unsequenced policy ----------------------------------------------

    SECTION( "count k-mers with par_unseq policy" )
    {   auto result = gnx::count(par_unseq, s1, 5);
        CHECK(result == expected);
    }

// -- verify total count -------------------------------------------------------

    SECTION( "total k-mer count is correct" )
    {   auto result = gnx::count(par, s1, 5);
        std::size_t total = 0;
        for (const auto& [kmer, count] : result)
            total += count;
        // For a sequence of length N, there are N - k + 1 k-mers
        CHECK(total == N - 5 + 1);
    }
}

// =============================================================================
// count() k-mer device tests
// =============================================================================

#if defined(__CUDACC__) || defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::count k-mers::device"
,   "[algorithm][count][kmer][device]"
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
)
{   typedef TestType T;
    const auto N{1'000};

    gnx::generic_sequence<T> s(N); // device
    gnx::sq baseline(N); // host
    gnx::rand(s.begin(), N, "ACGTacgt", seed_pi);
    gnx::rand(baseline.begin(), N, "ACGTacgt", seed_pi);

    // Get baseline result
    auto expected = gnx::count(baseline, 5);

#if defined(__CUDACC__)
    auto policy = thrust::cuda::par;
#else
    auto policy = thrust::hip::par;
#endif

// -- device k-mer counting ----------------------------------------------------

    SECTION( "count k-mers on device" )
    {   auto result = gnx::count(policy, s, 5);
        CHECK(result == expected);
    }

// -- streams on device --------------------------------------------------------

#if defined(__CUDACC__)
    SECTION( "CUDA stream with k-mers" )
    {   cudaStream_t stream; cudaStreamCreate(&stream);
        auto result = gnx::count(thrust::cuda::par.on(stream), s, 5);
        CHECK(result == expected);
        cudaStreamSynchronize(stream); cudaStreamDestroy(stream);
    }
#else // __HIPCC__
    SECTION( "HIP stream with k-mers" )
    {   hipStream_t stream; hipStreamCreate(&stream);
        auto result = gnx::count(thrust::hip::par.on(stream), s, 5);
        CHECK(result == expected);
        hipStreamSynchronize(stream); hipStreamDestroy(stream);
    }
#endif // __CUDACC__
}
#endif // __CUDACC__ || __HIPCC__
