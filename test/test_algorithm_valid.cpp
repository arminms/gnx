// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#include <catch2/catch_all.hpp>

#include <gnx/sq.hpp>
#include <gnx/algorithms/valid.hpp>

// =============================================================================
// valid() algorithm tests
// =============================================================================

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::valid"
,   "[algorithm][valid][cuda]"
,   std::vector<char>
,   thrust::host_vector<char>
,   thrust::universal_vector<char>
)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::valid"
,   "[algorithm][valid][rocm]"
,   std::vector<char>
,   thrust::host_vector<char>
,   thrust::universal_vector<char>
,   gnx::unified_vector<char>
)
#else
TEMPLATE_TEST_CASE( "gnx::valid", "[algorithm][valid]", std::vector<char>)
#endif //__CUDACC__
{   typedef TestType T;

// -- nucleotide validation ----------------------------------------------------

    SECTION( "valid nucleotide sequences" )
    {   gnx::generic_sequence<T> s1{"ACGT"};
        CHECK(gnx::valid(s1, true));
        CHECK(gnx::valid_nucleotide(s1));
        gnx::generic_sequence<T> s2{"ACGTACGTNNN"};
        CHECK(gnx::valid_nucleotide(s2));
        // lowercase
        gnx::generic_sequence<T> s3{"acgtacgt"};
        CHECK(gnx::valid_nucleotide(s3));
        // mixed case
        gnx::generic_sequence<T> s4{"AcGtNn"};
        CHECK(gnx::valid_nucleotide(s4));
        // with RNA base U
        gnx::generic_sequence<T> s5{"ACGU"};
        CHECK(gnx::valid_nucleotide(s5));
        // with IupAC ambiguity codes
        gnx::generic_sequence<T> s6{"ACGTRYMKSWBDHVN"};
        CHECK(gnx::valid_nucleotide(s6));
        gnx::generic_sequence<T> s7{"acgtrymkswbdhvn"};
        CHECK(gnx::valid_nucleotide(s7));
    }

    SECTION( "invalid nucleotide sequences" )
    {   // with invalid character
        gnx::generic_sequence<T> s1{"ACGT123"};
        CHECK_FALSE(gnx::valid_nucleotide(s1));
        gnx::generic_sequence<T> s2{"ACGT X"};
        CHECK_FALSE(gnx::valid_nucleotide(s2));
        // with space
        gnx::generic_sequence<T> s3{"ACG T"};
        CHECK_FALSE(gnx::valid_nucleotide(s3));
        // with newline
        gnx::generic_sequence<T> s4{"ACGT\n"};
        CHECK_FALSE(gnx::valid_nucleotide(s4));
        // peptide sequence
        gnx::generic_sequence<T> s5{"MVHLTPEEK"};
        CHECK_FALSE(gnx::valid_nucleotide(s5));
    }

    SECTION( "empty nucleotide sequence" )
    {   gnx::generic_sequence<T> s{""};
        CHECK(gnx::valid_nucleotide(s));
    }

// -- peptide validation -------------------------------------------------------

    SECTION( "valid peptide sequences" )
    {   gnx::generic_sequence<T> s1{"ACDEFGHIKLMNPQRSTVWY"};
        CHECK(gnx::valid(s1));
        CHECK(gnx::valid(s1, false));
        CHECK(gnx::valid_peptide(s1));
        CHECK_FALSE(gnx::valid_nucleotide(s1));
        // lowercase
        gnx::generic_sequence<T> s2{"acdefghiklmnpqrstvwy"};
        CHECK(gnx::valid_peptide(s2));
        // mixed case
        gnx::generic_sequence<T> s3{"MvHlTpEeK"};
        CHECK(gnx::valid_peptide(s3));
        // with ambiguous codes
        gnx::generic_sequence<T> s4{"ACBZX"};
        CHECK(gnx::valid_peptide(s4));
        // with special amino acids
        gnx::generic_sequence<T> s5{"ACUO"};
        CHECK(gnx::valid_peptide(s5));
        // with stop codon
        gnx::generic_sequence<T> s6{"MVHLT*"};
        CHECK(gnx::valid_peptide(s6));
    }

    SECTION( "invalid peptide sequences" )
    {   // with number
        gnx::generic_sequence<T> s1{"ACDE123"};
        CHECK_FALSE(gnx::valid_peptide(s1));
        // with space
        gnx::generic_sequence<T> s2{"ACDE F"};
        CHECK_FALSE(gnx::valid_peptide(s2));
        // with newline
        gnx::generic_sequence<T> s3{"ACDEF\n"};
        CHECK_FALSE(gnx::valid_peptide(s3));
        // with invalid special character
        gnx::generic_sequence<T> s4{"ACDE-F"};
        CHECK_FALSE(gnx::valid_peptide(s4));
    }

    SECTION( "empty sequence" )
    {   gnx::generic_sequence<T> s{""};
        CHECK(gnx::valid(s));
    }

// -- iterator-based validation ------------------------------------------------

    SECTION( "validation with iterators" )
    {   gnx::generic_sequence<T> s{"ACGTACGT"};
        // full range
        CHECK(gnx::valid(s.begin(), s.end(), true));
        CHECK(gnx::valid_nucleotide(s.begin(), s.end()));
        // partial range
        CHECK(gnx::valid_nucleotide(s.begin(), s.begin() + 4));
        // subsequence view
        auto sub = s(0, 4);
        CHECK(gnx::valid_nucleotide(sub));
    }

    SECTION( "validation with execution policy" )
    {   gnx::generic_sequence<T> s;
        s.load(SAMPLE_GENOME, 0);
        CHECK(s.size() > 0);
        CHECK(gnx::valid_nucleotide(s));
        CHECK(gnx::valid_nucleotide(gnx::execution::seq, s));
        CHECK(gnx::valid_nucleotide(gnx::execution::unseq, s));
        CHECK(gnx::valid_nucleotide(gnx::execution::par, s));
        CHECK(gnx::valid_nucleotide(gnx::execution::par_unseq, s));

        // introduce an invalid character and ensure policy overload detects it
        s[2] = 'Z';
        CHECK_FALSE(gnx::valid_nucleotide(gnx::execution::seq, s));
        CHECK_FALSE(gnx::valid_nucleotide(gnx::execution::unseq, s));
        CHECK_FALSE(gnx::valid_nucleotide(gnx::execution::par, s));
        CHECK_FALSE(gnx::valid_nucleotide(gnx::execution::par_unseq, s));

        // peptide validation remains true with Z
        CHECK(gnx::valid_peptide(gnx::execution::par, s));
    }

    SECTION( "validation with sq_view" )
    {   gnx::generic_sequence<T> s{"ACGTACGT"};
        gnx::generic_sequence_view<T> view{s};
        CHECK(gnx::valid_nucleotide(view));
        CHECK(gnx::valid_nucleotide(view.begin(), view.end()));
    }

// -- compile-time validation --------------------------------------------------

    SECTION( "constexpr validation" )
    {   // These should compile if the function is constexpr
        constexpr std::array<char, 4> arr{'A', 'C', 'G', 'T'};
        constexpr bool result = gnx::valid_nucleotide(arr.begin(), arr.end());
        CHECK(result);
    }

// -- cross-validation tests ---------------------------------------------------

    SECTION( "sequences valid for one type but not another" )
    {   // Some characters valid for peptides but not nucleotides
        gnx::generic_sequence<T> s1{"EFIKLPQVWY"};
        CHECK(gnx::valid_peptide(s1));
        CHECK_FALSE(gnx::valid_nucleotide(s1));

        // All nucleotides are also valid peptides (overlap in alphabet)
        gnx::generic_sequence<T> s2{"ACGT"};
        CHECK(gnx::valid_nucleotide(s2));
        CHECK(gnx::valid_peptide(s2)); // A, C, G, T are also amino acids
    }

// -- range compatibility tests ------------------------------------------------

    SECTION( "view ranges compatibility" )
    {   gnx::generic_sequence<T> s{"ACGTACGTKQ"};
        CHECK(gnx::valid_nucleotide(s | std::views::take(8)));
        gnx::generic_sequence_view<T> v{s};
        v.remove_suffix(2); // drop 'KQ'
        CHECK(gnx::valid_nucleotide(v));
        CHECK(gnx::valid_nucleotide(v.begin(), v.end()));
        auto result = s
        |   std::views::take(8)
        |   std::views::transform([](auto c) { return std::tolower(c); })
        |   std::views::reverse;
        CHECK(gnx::valid_nucleotide(result));
    }
}

// =============================================================================
// valid() algorithm device tests
// =============================================================================

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::valid::device"
,   "[algorithm][valid][cuda]"
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
)
{   typedef TestType T;

    gnx::generic_sequence<T> s;
    s.load(SAMPLE_GENOME, 0);
    CHECK(s.size() > 0);

    SECTION( "device vector" )
    {   CHECK(gnx::valid_nucleotide(thrust::cuda::par, s));
        s[2] = 'Z';
        CHECK_FALSE(gnx::valid_nucleotide(thrust::cuda::par, s));
    }

    SECTION( "cuda stream" )
    {   cudaStream_t streamA;
        cudaStreamCreate(&streamA);
        CHECK(gnx::valid_nucleotide(thrust::cuda::par.on(streamA), s));
        cudaStreamSynchronize(streamA);
        s[2] = 'Z';
        CHECK_FALSE(gnx::valid_nucleotide(thrust::cuda::par_nosync.on(streamA), s));
        cudaStreamSynchronize(streamA);
        cudaStreamDestroy(streamA);
    }
}
#endif //__CUDACC__

#if defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::valid::device"
,   "[algorithm][valid][rocm]"
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
,   gnx::unified_vector<char>
)
{   typedef TestType T;

    gnx::generic_sequence<T> s;
    s.load(SAMPLE_GENOME, 0);
    CHECK(s.size() > 0);

    SECTION( "device vector" )
    {   CHECK(gnx::valid_nucleotide(thrust::hip::par, s));
        s[2] = 'Z';
        CHECK_FALSE(gnx::valid_nucleotide(thrust::hip::par, s));
    }

    SECTION( "cuda stream" )
    {   hipStream_t streamA;
        hipStreamCreate(&streamA);
        CHECK(gnx::valid_nucleotide(thrust::hip::par.on(streamA), s));
        hipStreamSynchronize(streamA);
        s[2] = 'Z';
        CHECK_FALSE(gnx::valid_nucleotide(thrust::hip::par_nosync.on(streamA), s));
        hipStreamSynchronize(streamA);
        hipStreamDestroy(streamA);
    }
}
#endif //__HIPCC__
