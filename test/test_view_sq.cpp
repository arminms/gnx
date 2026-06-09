// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#include <catch2/catch_all.hpp>

#include <gnx/sq.hpp>

// =============================================================================
// generic_sequence_view class tests
// =============================================================================

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::generic_sequence_view"
,   "[view][sq_view][cuda]"
,   std::vector<char>
,   thrust::host_vector<char>
,   thrust::universal_vector<char>
)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::generic_sequence_view"
,   "[view][sq_view][rocm]"
,   std::vector<char>
,   thrust::host_vector<char>
,   thrust::universal_vector<char>
,   gnx::unified_vector<char>
)
#else
TEMPLATE_TEST_CASE
(   "gnx::generic_sequence_view"
,   "[view][sq_view]"
,   std::vector<char>
)
#endif //__CUDACC__
{   typedef TestType T;

    gnx::generic_sequence<T> s{"ACGT"};

// -- constructors -------------------------------------------------------------

    SECTION( "construct from generic_sequence" )
    {   gnx::generic_sequence_view<T> v{s};
        CHECK(v.size() == s.size());
        CHECK_FALSE(v.empty());
        CHECK(v == s);
        CHECK(v == "ACGT");
    }

    SECTION( "construct from pointer+length" )
    {   const char* p = "ACGT";
        gnx::generic_sequence_view<T> v{p, 4};
        CHECK(v.size() == 4);
        CHECK(v[0] == 'A');
        CHECK(v.at(3) == 'T');
        CHECK(v.front() == 'A');
        CHECK(v.back() == 'T');
    }

// -- iterators ----------------------------------------------------------------

    SECTION( "iterate over view" )
    {   gnx::generic_sequence_view<T> v{s};
        std::string collected;
        for (auto it = v.begin(); it != v.end(); ++it)
            collected.push_back(*it);
        CHECK(collected == "ACGT");

        std::string rev;
        for (auto it = v.rbegin(); it != v.rend(); ++it)
            rev.push_back(*it);
        CHECK(rev == std::string{"TGCA"});
    }

// -- modifiers ----------------------------------------------------------------

    SECTION( "remove_prefix/suffix" )
    {   gnx::generic_sequence_view<T> v{s};
        v.remove_prefix(1); // drop 'A'
        CHECK(v == "CGT");
        v.remove_suffix(1); // drop trailing 'T'
        CHECK(v == "CG");
        // original sequence unchanged
        CHECK(s == "ACGT");
    }

    SECTION( "remove_prefix/suffix bounds" )
    {   gnx::generic_sequence_view<T> v{s};
        REQUIRE_THROWS_AS(v.remove_prefix(5), std::out_of_range);
        REQUIRE_THROWS_AS(v.remove_suffix(5), std::out_of_range);
    }

// -- operations ---------------------------------------------------------------

    SECTION( "subseq" )
    {   gnx::generic_sequence_view<T> v{s};
        auto sub = v.subseq(1, 2);
        CHECK(sub == "CG");
        auto to_end = v.subseq(2);
        CHECK(to_end == "GT");
        REQUIRE_THROWS_AS(v.subseq(10), std::out_of_range);
    }

// -- comparisons --------------------------------------------------------------

    SECTION( "compare to generic_sequence and C-string" )
    {   gnx::generic_sequence_view<T> v{s};
        CHECK(v == s);
        CHECK(s == v);
        CHECK(v == "ACGT");
        CHECK("ACGT" == v);
        CHECK(v != "acgt");
        CHECK("acgt" != v);
    }

// -- ranges concept support --------------------------------------------------

    SECTION( "std::ranges::view concept" )
    {   gnx::generic_sequence_view<T> v{s};
        // Verify that generic_sequence_view satisfies the view concept
        static_assert(std::ranges::view<gnx::generic_sequence_view<T>>);
        static_assert(std::ranges::range<gnx::generic_sequence_view<T>>);

        // Test composability with range adaptors
        auto transformed = v | std::views::transform
        (   [](auto c)
            {   return c == 'A' ? 'T' : c;
            }
        );
        std::string r(transformed.begin(), transformed.end());
        CHECK(r == "TCGT");
    }

    SECTION( "composable with multiple adaptors" )
    {   gnx::generic_sequence_view<T> v{s};
        // Chain multiple views
        auto result = v 
        |   std::views::reverse 
        |   std::views::transform([](auto c) { return std::tolower(c); })
        |   std::views::take(3);
        std::string r(result.begin(), result.end());
        CHECK(r == "tgc");
    }
}
