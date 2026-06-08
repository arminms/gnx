// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <gnx/sq.hpp>
#include <gnx/psq.hpp>
#include <gnx/algorithms/compare.hpp>
#include <gnx/algorithms/random.hpp>

const uint64_t seed_pi{3141592654};

// =============================================================================
// compare() algorithm tests
// =============================================================================

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::compare"
,   "[algorithm][compare][cuda]"
,   std::vector<char>
,   thrust::host_vector<char>
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::compare"
,   "[algorithm][compare][rocm]"
,   std::vector<char>
,   thrust::host_vector<char>
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
,   gnx::unified_vector<char>
)
#else
TEMPLATE_TEST_CASE( "gnx::compare", "[algorithm][compare]", std::vector<char>)
#endif
{   typedef TestType T;

    // Test data
    gnx::generic_sequence<T> s1("ACGTacgt");
    gnx::generic_sequence<T> s2("acgtACGT"); // Case difference
    gnx::generic_sequence<T> s3("ACGTacgZ"); // Mismatch at end
    gnx::generic_sequence<T> s4("ACGT");     // Different length

// -- basic comparison ---------------------------------------------------------

    SECTION( "compare identical sequences" )
    {   CHECK(gnx::compare(s1, s1));
    }

    SECTION( "compare case-insensitive match" )
    {   CHECK(gnx::compare(s1, s2));
    }

    SECTION( "compare mismatched sequences" )
    {   CHECK_FALSE(gnx::compare(s1, s3));
    }

    SECTION( "compare different length sequences" )
    {   CHECK_FALSE(gnx::compare(s1, s4));
    }

// -- iterator comparison ------------------------------------------------------

    SECTION( "compare with iterators" )
    {   CHECK(gnx::compare(s1.begin(), s1.end(), s2.begin(), s2.end()));
        CHECK_FALSE(gnx::compare(s1.begin(), s1.end(), s3.begin(), s3.end()));
    }

// -- empty sequences ----------------------------------------------------------

    SECTION( "compare empty sequences" )
    {   T empty1, empty2;
        CHECK(gnx::compare(empty1, empty2));
    }

    SECTION( "compare empty with non-empty" )
    {   T empty;
        CHECK_FALSE(gnx::compare(empty, s1));
        CHECK_FALSE(gnx::compare(s1, empty));
    }

// -- single character sequences -----------------------------------------------

    SECTION( "compare single characters" )
    {   T a1(1, 'A');
        T a2(1, 'a');
        T c1(1, 'C');
        CHECK(gnx::compare(a1, a2));
        CHECK_FALSE(gnx::compare(a1, c1));
    }
}

// =============================================================================
// compare() algorithm execution policy tests
// =============================================================================

TEMPLATE_TEST_CASE
(   "gnx::compare execution policies"
,   "[algorithm][compare][policy]"
,   std::vector<char>
)
{   typedef TestType T;

    using gnx::execution::seq;
    using gnx::execution::par;
    using gnx::execution::unseq;
    using gnx::execution::par_unseq;

    const auto N{10'000};

    gnx::generic_sequence<T> s1(N), s2(N), s3(N);
    gnx::rand(s1.begin(), N, "ACGTacgt", seed_pi);
    gnx::rand(s2.begin(), N, "ACGTacgt", seed_pi); // same seed (same as s1)
    gnx::rand(s3.begin(), N, "ACGTacgt");          // random seed (different)

// -- sequential policy --------------------------------------------------------

    SECTION( "compare with seq policy" )
    {   CHECK(gnx::compare(seq, s1, s2));
        CHECK_FALSE(gnx::compare(seq, s1, s3));
    }

// -- unsequenced policy -------------------------------------------------------

    SECTION( "compare with unseq policy" )
    {   CHECK(gnx::compare(unseq, s1, s2));
        CHECK_FALSE(gnx::compare(unseq, s1, s3));
    }

// -- parallel policy ----------------------------------------------------------

    SECTION( "compare with par policy" )
    {   CHECK(gnx::compare(par, s1, s2));
        CHECK_FALSE(gnx::compare(par, s1, s3));
    }

// -- parallel unsequenced policy ----------------------------------------------

    SECTION( "compare with par_unseq policy" )
    {   CHECK(gnx::compare(par_unseq, s1, s2));
        CHECK_FALSE(gnx::compare(par_unseq, s1, s3));
    }
}

// =============================================================================
// compare() algorithm device tests
// =============================================================================

#if defined(__CUDACC__) || defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::compare::device"
,   "[algorithm][compare][device]"
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
)
{   typedef TestType T;

    const auto N{10'000};

    gnx::generic_sequence<T> s1(N), s2(N), s3(N);
    gnx::rand(s1.begin(), N, "ACGTacgt", seed_pi);
    gnx::rand(s2.begin(), N, "ACGTacgt", seed_pi); // same seed (same as s1)
    gnx::rand(s3.begin(), N, "ACGTacgt");          // random seed (different)

#if defined(__CUDACC__)
    auto policy = thrust::cuda::par;
#else
    auto policy = thrust::hip::par;
#endif

// -- device comparison --------------------------------------------------------

    SECTION( "compare with device policy" )
    {   CHECK(gnx::compare(policy, s1, s2));
        CHECK_FALSE(gnx::compare(policy, s1, s3));
    }

// -- streams on device --------------------------------------------------------

#if defined(__CUDACC__)
    SECTION( "CUDA stream as policy" )
    {   cudaStream_t stream; cudaStreamCreate(&stream);
        CHECK(gnx::compare(thrust::cuda::par.on(stream), s1, s2));
        CHECK_FALSE(gnx::compare(thrust::cuda::par.on(stream), s1, s3));
        cudaStreamSynchronize(stream); cudaStreamDestroy(stream);
    }
#else // __HIPCC__
    SECTION( "HIP stream as policy" )
    {   hipStream_t stream; hipStreamCreate(&stream);
        CHECK(gnx::compare(thrust::hip::par.on(stream), s1, s2));
        CHECK_FALSE(gnx::compare(thrust::hip::par.on(stream), s1, s3));
        hipStreamSynchronize(stream); hipStreamDestroy(stream);
    }
#endif // __CUDACC__
}
#endif //__CUDACC__ || __HIPCC__
