// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <gnx/sq.hpp>
#include <gnx/psq.hpp>
#include <gnx/utility/gc_content.hpp>
#include <gnx/algorithms/valid.hpp>
#include <gnx/algorithms/random.hpp>

const uint64_t seed_pi{3141592654};

// =============================================================================
// random() algorithm tests
// =============================================================================

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::random"
,   "[algorithm][random][cuda]"
,   std::vector<char>
,   thrust::host_vector<char>
)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::random"
,   "[algorithm][random][rocm]"
,   std::vector<char>
,   thrust::host_vector<char>
)
#else
TEMPLATE_TEST_CASE( "gnx::random", "[algorithm][random]", std::vector<char>)
#endif //__CUDACC__ || __HIPCC__
{   typedef TestType T;
    gnx::generic_sequence<T> s(20);
    const auto N{10'000};

    SECTION( "random nucleotide sequence" )
    {   gnx::rand(s.begin(), 20, "ACGT", seed_pi);
        CHECK(gnx::valid_nucleotide(s));
        CHECK(s == "TTCGGCCGTCGTTAAACACG");
        auto t = gnx::random::dna<decltype(s)>(20, seed_pi);
        CHECK(s == t);
    }

    SECTION( "random sequence with execution policy" )
    {   gnx::generic_sequence<T> r(N);
        auto t = gnx::random::dna<decltype(s)>(N, seed_pi);
        CHECK(N == t.size());
        gnx::rand(gnx::execution::seq, r.begin(), N, "ACGT", seed_pi);
        CHECK(t == r);
        gnx::rand(gnx::execution::unseq, r.begin(), N, "ACGT", seed_pi);
        CHECK(t == r);
        gnx::rand(gnx::execution::par, r.begin(), N, "ACGT", seed_pi);
        CHECK(t == r);
        gnx::rand(gnx::execution::par_unseq, r.begin(), N, "ACGT", seed_pi);
        CHECK(t == r);
    }

    SECTION( "random rna sequence" )
    {   gnx::rand(s.begin(), 20, "ACGU", seed_pi);
        CHECK(gnx::valid_nucleotide(s));
        CHECK(s == "UUCGGCCGUCGUUAAACACG");
        auto t = gnx::random::rna<decltype(s)>(20, seed_pi);
        CHECK(s == t);
    }

    SECTION( "random protein sequence" )
    {   gnx::rand(s.begin(), 20, "ACDEFGHIKLMNPQRSTVWY", seed_pi);
        CHECK(gnx::valid_peptide(s));
        CHECK(s == "TTIQRHHMVKQSSFDALCLM");
        auto t = gnx::random::protein<decltype(s)>(20, seed_pi);
        CHECK(s == t);
    }

    SECTION( "random nucleotide sequence with weights" )
    {   gnx::generic_sequence<T> t1(N);
        gnx::rand(t1.begin(), N, "ACGT", {35, 15, 15, 35}, seed_pi);
        REQUIRE_THAT(gnx::gc_content(t1), Catch::Matchers::WithinRel(30, 0.1));
        auto t2 = gnx::random::dna(N, 30.0, seed_pi);
        REQUIRE_THAT(gnx::gc_content(t2), Catch::Matchers::WithinRel(30, 0.1));
    }
}

// =============================================================================
// random() algorithm device tests
// =============================================================================

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::random::device"
,   "[algorithm][random][cuda]"
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
)
{   typedef TestType T;
    gnx::generic_sequence<T> s(20);
    const auto N{10'000};

    SECTION( "device vector" )
    {   gnx::rand(thrust::cuda::par, s.begin(), 20, "ACGT", seed_pi);
        CHECK(gnx::valid_nucleotide(thrust::cuda::par, s));
        CHECK(s == "TTCGGCCGTCGTTAAACACG");
        auto t = gnx::random::dna<decltype(s)>(20, seed_pi);
        CHECK(s == t);
    }

    SECTION( "cuda stream" )
    {   auto r = gnx::random::dna<decltype(s)>(N, seed_pi);
        gnx::generic_sequence<T> t(N);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        gnx::rand(thrust::cuda::par.on(stream), t.begin(), N, "ACGT", seed_pi);
        CHECK(gnx::valid_nucleotide(thrust::cuda::par.on(stream), t));
        CHECK(r == t);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
}
#endif //__CUDACC__

#if defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::random::device"
,   "[algorithm][random][rocm]"
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
,   gnx::unified_vector<char>
)
{   typedef TestType T;
    gnx::generic_sequence<T> s(20);
    const auto N{10'000};

    SECTION( "device vector" )
    {   gnx::rand(thrust::hip::par, s.begin(), 20, "ACGT", seed_pi);
        CHECK(gnx::valid_nucleotide(thrust::hip::par, s));
        CHECK(s == "TTCGGCCGTCGTTAAACACG");
        auto t = gnx::random::dna<decltype(s)>(20, seed_pi);
        CHECK(s == t);
    }

    SECTION( "hip stream" )
    {   auto r = gnx::random::dna<decltype(s)>(N, seed_pi);
        gnx::generic_sequence<T> t(N);
        hipStream_t stream;
        hipStreamCreate(&stream);
        gnx::rand(thrust::hip::par.on(stream), t.begin(), N, "ACGT", seed_pi);
        CHECK(gnx::valid_nucleotide(thrust::hip::par.on(stream), t));
        CHECK(r == t);
        hipStreamSynchronize(stream);
        hipStreamDestroy(stream);
    }
}
#endif //__HIPCC__

// =============================================================================
// rand_packed() algorithm device tests
// =============================================================================

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::rand_packed::device"
,   "[algorithm][random][packed_2bit][cuda]"
 ,   thrust::device_vector<uint8_t>
)
{   typedef TestType T;
    using packed_type = gnx::packed_generic_sequence_2bit<T>;

    constexpr std::size_t packed_length{257};
    const auto byte_count = gnx::psq2::num_bytes(packed_length);

    packed_type expected(packed_length);
    gnx::rand_packed(thrust::cuda::par, expected.data(), byte_count, 2, seed_pi);
    cudaDeviceSynchronize();

    SECTION( "device execution policy" )
    {   packed_type generated(packed_length);
        gnx::rand_packed(thrust::cuda::par, generated.data(), byte_count, 2, seed_pi);
        cudaDeviceSynchronize();
        for (std::size_t i = 0; i < packed_length; ++i)
            CHECK(generated.get_base(i) == expected.get_base(i));
    }

    SECTION( "cuda stream" )
    {   packed_type generated(packed_length);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        gnx::rand_packed
        (   thrust::cuda::par.on(stream)
        ,   generated.data()
        ,   byte_count
        ,   2
        ,   seed_pi
        );
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);

        for (std::size_t i = 0; i < packed_length; ++i)
            CHECK(generated.get_base(i) == expected.get_base(i));
    }

    SECTION( "random::packed_2bit::dna uses device path" )
    {   auto generated = gnx::random::packed_2bit::dna<packed_type>(packed_length, seed_pi);
        for (std::size_t i = 0; i < packed_length; ++i)
            CHECK(generated.get_base(i) == expected.get_base(i));
    }
}
#endif //__CUDACC__

#if defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::rand_packed::device"
,   "[algorithm][random][packed_2bit][rocm]"
,   thrust::universal_vector<uint8_t>
,   gnx::unified_vector<uint8_t>
)
{   typedef TestType T;
    using packed_type = gnx::packed_generic_sequence_2bit<T>;

    constexpr std::size_t packed_length{257};
    const auto byte_count = gnx::psq2::num_bytes(packed_length);

    packed_type expected(packed_length);
    gnx::rand_packed(thrust::hip::par, expected.data(), byte_count, 2, seed_pi);
    hipDeviceSynchronize();

    SECTION( "device execution policy" )
    {   packed_type generated(packed_length);
        gnx::rand_packed(thrust::hip::par, generated.data(), byte_count, 2, seed_pi);
        hipDeviceSynchronize();
        for (std::size_t i = 0; i < packed_length; ++i)
            CHECK(generated.get_base(i) == expected.get_base(i));
    }

    SECTION( "hip stream" )
    {   packed_type generated(packed_length);
        hipStream_t stream;
        hipStreamCreate(&stream);
        gnx::rand_packed
        (   thrust::hip::par.on(stream)
        ,   generated.data()
        ,   byte_count
        ,   2
        ,   seed_pi
        );
        hipStreamSynchronize(stream);
        hipStreamDestroy(stream);

        for (std::size_t i = 0; i < packed_length; ++i)
            CHECK(generated.get_base(i) == expected.get_base(i));
    }

    SECTION( "random::packed_2bit::dna uses device path" )
    {   auto generated = gnx::random::packed_2bit::dna<packed_type>(packed_length, seed_pi);
        for (std::size_t i = 0; i < packed_length; ++i)
            CHECK(generated.get_base(i) == expected.get_base(i));
    }
}
#endif //__HIPCC__
