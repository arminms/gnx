// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#include <catch2/catch_all.hpp>

#include <gnx/psq.hpp>

// =============================================================================
// gnx::packed_generic_sequence_2bit_view (psq2_view) tests
// =============================================================================

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::packed_generic_sequence_2bit_view"
,   "[view][psq2][cuda]"
,   std::vector<uint8_t>
,   thrust::universal_vector<uint8_t>
)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::packed_generic_sequence_2bit_view"
,   "[view][psq2][rocm]"
,   std::vector<uint8_t>
,   thrust::universal_vector<uint8_t>
,   gnx::unified_vector<uint8_t>
)
#else
TEMPLATE_TEST_CASE
(   "gnx::packed_generic_sequence_2bit_view"
,   "[view][psq2]"
,   std::vector<uint8_t>
)
#endif
{   typedef TestType T;

    gnx::packed_generic_sequence_2bit<T> s{"ACGTACGT"};

// -- constructors -------------------------------------------------------------

    SECTION( "construct from packed_generic_sequence_2bit (full view)" )
    {   gnx::packed_generic_sequence_2bit_view<T> v{s};
        CHECK(v.size() == s.size());
        CHECK_FALSE(v.empty());
        CHECK(v == "ACGTACGT");
    }

    SECTION( "default constructor is empty" )
    {   gnx::packed_generic_sequence_2bit_view<T> v;
        CHECK(v.empty());
        CHECK(v.size() == 0);
    }

    SECTION( "construct from raw pointer + offset + count" )
    {   // Construct the underlying packed sequence first
        gnx::packed_generic_sequence_2bit<T> p{"ACGTTTTT"};
        gnx::packed_generic_sequence_2bit_view<T> v{p.data(), 0, p.size()};
        CHECK(v.size() == 8);
        CHECK(v == "ACGTTTTT");
    }

// -- operator() ---------------------------------------------------------------

    SECTION( "operator() returns full view by default" )
    {   auto v = s();
        CHECK(v.size() == s.size());
        CHECK(v == "ACGTACGT");
    }

    SECTION( "operator() with pos only (to end)" )
    {   auto v = s(4);
        CHECK(v.size() == 4);
        CHECK(v == "ACGT");
    }

    SECTION( "operator() with pos and count" )
    {   auto v = s(2, 4);
        CHECK(v.size() == 4);
        CHECK(v == "GTAC");
    }

    SECTION( "operator() pos=0 count=0 gives empty view" )
    {   auto v = s(0, 0);
        CHECK(v.empty());
        CHECK(v.size() == 0);
    }

// -- subseq -------------------------------------------------------------------

    SECTION( "subseq: middle of sequence" )
    {   gnx::packed_generic_sequence_2bit_view<T> v{s};
        auto sub = v.subseq(1, 3);
        CHECK(sub.size() == 3);
        CHECK(sub == "CGT");
    }

    SECTION( "subseq: from pos to end" )
    {   gnx::packed_generic_sequence_2bit_view<T> v{s};
        auto sub = v.subseq(4);
        CHECK(sub.size() == 4);
        CHECK(sub == "ACGT");
    }

    SECTION( "subseq: full range" )
    {   gnx::packed_generic_sequence_2bit_view<T> v{s};
        auto sub = v.subseq(0);
        CHECK(sub == v);
    }

    SECTION( "subseq out of range throws" )
    {   gnx::packed_generic_sequence_2bit_view<T> v{s};
        CHECK_THROWS_AS(v.subseq(100), std::out_of_range);
    }

    SECTION( "chained subseq" )
    {   gnx::packed_generic_sequence_2bit_view<T> v{s};
        // "ACGTACGT" -> subseq(2,5) = "GTACG" -> subseq(1,3) = "TAC"
        auto sub = v.subseq(2, 5).subseq(1, 3);
        CHECK(sub == "TAC");
    }

// -- element access -----------------------------------------------------------

    SECTION( "operator[] and get_base" )
    {   gnx::packed_generic_sequence_2bit_view<T> v{s};
        CHECK(char(v[0]) == 'A');
        CHECK(char(v[1]) == 'C');
        CHECK(char(v[2]) == 'G');
        CHECK(char(v[3]) == 'T');
        CHECK(v.get_base(4) == 'A');
        CHECK(v.get_base(7) == 'T');
    }

    SECTION( "front and back" )
    {   gnx::packed_generic_sequence_2bit_view<T> v{s};
        CHECK(char(v.front()) == 'A');
        CHECK(char(v.back())  == 'T');
    }

    SECTION( "at() valid index" )
    {   gnx::packed_generic_sequence_2bit_view<T> v{s};
        CHECK(char(v.at(0)) == 'A');
        CHECK(char(v.at(7)) == 'T');
    }

    SECTION( "at() out of range throws" )
    {   gnx::packed_generic_sequence_2bit_view<T> v{s};
        CHECK_THROWS_AS(v.at(8), std::out_of_range);
        CHECK_THROWS_AS(v.at(100), std::out_of_range);
    }

    SECTION( "element access on subseq view" )
    {   // s = "ACGTACGT", subseq(2,4) = "GTAC"
        auto v = s(2, 4);
        CHECK(char(v[0]) == 'G');
        CHECK(char(v[1]) == 'T');
        CHECK(char(v[2]) == 'A');
        CHECK(char(v[3]) == 'C');
        CHECK(char(v.front()) == 'G');
        CHECK(char(v.back())  == 'C');
    }

// -- iterators ----------------------------------------------------------------

    SECTION( "forward iteration" )
    {   gnx::packed_generic_sequence_2bit_view<T> v{s};
        std::string collected;
        for (auto it = v.begin(); it != v.end(); ++it)
            collected += char(*it);
        CHECK(collected == "ACGTACGT");
    }

    SECTION( "range-for loop" )
    {   gnx::packed_generic_sequence_2bit_view<T> v{s};
        std::string collected;
        for (auto ref : v)
            collected += char(ref);
        CHECK(collected == "ACGTACGT");
    }

    SECTION( "cbegin/cend" )
    {   gnx::packed_generic_sequence_2bit_view<T> v{s};
        std::string collected;
        for (auto it = v.cbegin(); it != v.cend(); ++it)
            collected += char(*it);
        CHECK(collected == "ACGTACGT");
    }

    SECTION( "reverse iteration" )
    {   gnx::packed_generic_sequence_2bit_view<T> v{s};
        std::string rev;
        for (auto it = v.rbegin(); it != v.rend(); ++it)
            rev += char(*it);
        CHECK(rev == "TGCATGCA");
    }

    SECTION( "random access iterator arithmetic" )
    {   gnx::packed_generic_sequence_2bit_view<T> v{s};
        auto it = v.begin();
        CHECK(char(*(it + 2)) == 'G');
        it += 4;
        CHECK(char(*it) == 'A');
        CHECK((v.end() - v.begin()) == 8);
    }

    SECTION( "iterator on subseq view" )
    {   auto v = s(1, 4); // "CGTA"
        std::string collected;
        for (auto ref : v)
            collected += char(ref);
        CHECK(collected == "CGTA");
    }

// -- comparisons --------------------------------------------------------------

    SECTION( "compare two views (same content)" )
    {   gnx::packed_generic_sequence_2bit<T> other{"ACGTACGT"};
        gnx::packed_generic_sequence_2bit_view<T> v1{s};
        gnx::packed_generic_sequence_2bit_view<T> v2{other};
        CHECK(v1 == v2);
        CHECK_FALSE(v1 != v2);
    }

    SECTION( "compare two views (different content)" )
    {   gnx::packed_generic_sequence_2bit<T> other{"TTTTTTTT"};
        gnx::packed_generic_sequence_2bit_view<T> v1{s};
        gnx::packed_generic_sequence_2bit_view<T> v2{other};
        CHECK(v1 != v2);
        CHECK_FALSE(v1 == v2);
    }

    SECTION( "compare two views (different sizes)" )
    {   auto v1 = s();     // length 8
        auto v2 = s(0, 4); // length 4
        CHECK(v1 != v2);
    }

    SECTION( "compare view to string_view and C-string" )
    {   auto v = s();
        CHECK(v == "ACGTACGT");
        CHECK("ACGTACGT" == v);
        CHECK(v != "TTTTTTTT");
        CHECK("TTTTTTTT" != v);
        std::string_view sv = "ACGTACGT";
        CHECK((v == sv));
        CHECK((sv  == v));
    }

    SECTION( "original sequence unmodified by view operations" )
    {   auto v = s(2, 3);   // "GTA"
        CHECK(v.size() == 3);
        CHECK(s == "ACGTACGT"); // original unchanged
    }

    SECTION( "zero-copy: view data pointer equals original data pointer" )
    {   gnx::packed_generic_sequence_2bit_view<T> v{s};
        // The view must reference the same packed bytes
        CHECK(v.data() == s.data());
        auto sub = s(4);
        // sub-view also shares the same backing array
        CHECK(sub.data() == s.data());
        CHECK(sub.offset() == 4u);
    }

// -- ranges concept -----------------------------------------------------------

    SECTION( "std::ranges::view concept" )
    {   static_assert
        (   std::ranges::view<gnx::packed_generic_sequence_2bit_view<T>>
        );
        static_assert
        (   std::ranges::range<gnx::packed_generic_sequence_2bit_view<T>>
        );
    }

    SECTION( "composable with range adaptors" )
    {   gnx::packed_generic_sequence_2bit_view<T> v{s};
        // transform each base char, collect result
        auto result = v | std::views::transform
        (   [](auto ref) -> char
            {   return char(ref) == 'A' ? 'T' : char(ref);
            }
        );
        std::string r(result.begin(), result.end());
        CHECK(r == "TCGTTCGT");
    }

    SECTION( "composable with reverse and take" )
    {   gnx::packed_generic_sequence_2bit_view<T> v{s};
        auto result = v
            | std::views::reverse
            | std::views::take(4);
        std::string r;
        for (auto ref : result)
            r += char(ref);
        CHECK(r == "TGCA");
    }
}
