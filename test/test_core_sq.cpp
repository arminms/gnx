// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2025 Armin Sobhani
//
#include <catch2/catch_all.hpp>

#include <gnx/sq.hpp>

template<typename T>
using aligned_vector = std::vector<T, gnx::aligned_allocator<T, gnx::Alignment::AVX512>>;

// =============================================================================
// gnx::generic_sequence class tests
// =============================================================================

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::sq"
,   "[class][cuda]"
,   std::vector<char>
// ,   aligned_vector<char>
,   thrust::host_vector<char>
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::sq"
,   "[class][rocm]"
,   std::vector<char>
,   thrust::host_vector<char>
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
,   gnx::unified_vector<char>
)
#else
TEMPLATE_TEST_CASE( "gnx::sq", "[class]", std::vector<char>)
#endif
{   typedef TestType T;

    gnx::generic_sequence<T> s{"ACGT"};
    std::string t{"ACGT"}, u{"acgt"}, v{"ACGT "};
    s["test-int"] = -33;

// -- comparison operators -----------------------------------------------------

    SECTION( "comparison operators" )
    {   REQUIRE(  s == gnx::generic_sequence<T>("ACGT"));
        REQUIRE(!(s == gnx::generic_sequence<T>("acgt")));
        REQUIRE(  s != gnx::generic_sequence<T>("acgt") );
        REQUIRE(!(s != gnx::generic_sequence<T>("ACGT")));

        REQUIRE(s == t);
        REQUIRE(t == s);
        REQUIRE(s != u);
        REQUIRE(u != s);
        REQUIRE(s != v);
        REQUIRE(v != s);

        REQUIRE(s == "ACGT");
        REQUIRE("ACGT" == s);
        REQUIRE(s != "acgt");
        REQUIRE("acgt" != s);
        REQUIRE(s != "ACGT ");
        REQUIRE("ACGT " != s);
    }

// -- constructors -------------------------------------------------------------

    SECTION( "single value constructor" )
    {
#if defined(__CUDACC__) || defined(__HIPCC__) // not supported for thrust::device_vector
        if constexpr
        (   !std::is_same_v<T, thrust::device_vector<char>>
        &&  !std::is_same_v<T, thrust::universal_vector<char>>
#if defined(__HIPCC__)
        &&  !std::is_same_v<T, gnx::unified_vector<char>>
#endif //__HIPCC__
        )
        {   gnx::generic_sequence<T> a4(4, 'A');
            CHECK(a4 == "AAAA");
            gnx::generic_sequence<T> c4(4, 'C');
            CHECK(c4 == "CCCC");
        }
#else
        gnx::generic_sequence<T> a4(4, 'A');
        CHECK(a4 == "AAAA");
        gnx::generic_sequence<T> c4(4, 'C');
        CHECK(c4 == "CCCC");
#endif //__CUDACC__
    }
    SECTION( "string_view constructor" )
    {   gnx::generic_sequence<T> c("ACGT");
        CHECK(s == c);
    }
    SECTION( "sq_view constructor" )
    {
#if defined(__CUDACC__) || defined(__HIPCC__) // not supported for thrust::device_vector
        if constexpr
        (   !std::is_same_v<T, thrust::device_vector<char>>
        )
        {   gnx::generic_sequence_view<T> sv(s);
            CHECK(s == sv);
        }
#else
        gnx::generic_sequence_view<T> sv(s);
        CHECK(s == sv);
#endif //__CUDACC__
    }
    SECTION( "iterator constructor" )
    {   std::string acgt{"ACGT"};
        gnx::generic_sequence<T> c(std::begin(acgt), std::end(acgt));
        CHECK(s == c);
    }
    SECTION( "copy constructor" )
    {   gnx::generic_sequence<T> c(s);
        CHECK(c == s);
        CHECK(-33 == std::get<int>(c["test-int"]));
    }
    SECTION( "move constructor" )
    {   gnx::generic_sequence<T> m(std::move(s));
        CHECK(s.empty());
        CHECK(m == gnx::generic_sequence<T>("ACGT"));
        CHECK(-33 == std::get<int>(m["test-int"]));
    }
    SECTION( "initializer list" )
    {   gnx::generic_sequence<T> c{'A', 'C', 'G', 'T'};
        CHECK(c == s);
    }

// -- copy assignment operators ------------------------------------------------

    SECTION( "copy assignment operator" )
    {   gnx::generic_sequence<T> c = s;
        CHECK(c == s);
        CHECK(-33 == std::get<int>(c["test-int"]));
    }
    SECTION( "move constructor" )
    {   gnx::generic_sequence<T> m = gnx::generic_sequence<T>("ACGT");
        CHECK(m == s);
    }
    SECTION( "initializer list" )
    {   gnx::generic_sequence<T> c = {'A', 'C', 'G', 'T'};
        CHECK(c == s);
    }

// -- iterators ----------------------------------------------------------------

    SECTION( "begin/end" )
    {   gnx::generic_sequence<T> t("AAAA");
        for (auto a : t)
            CHECK(a == 'A');
#if defined(__CUDACC__) || defined(__HIPCC__) // not supported for thrust::device_vector
        if constexpr
        (   !std::is_same_v<T, thrust::device_vector<char>>
#if defined(__HIPCC__)
        &&  !std::is_same_v<T, gnx::unified_vector<char>>
#endif //__HIPCC__
        )
        {   for (auto& a : t)
                a = 'T';
            CHECK(t == gnx::generic_sequence<T>("TTTT"));
        }
#else
        for (auto& a : t)
            a = 'T';
        CHECK(t == gnx::generic_sequence<T>("TTTT"));
#endif //__CUDACC__
        auto s_it = s.cbegin();
        for
        (   auto t_it = t.begin()
        ;   t_it != t.end()
        ;   ++t_it, ++s_it
        )
            *t_it = *s_it;
        CHECK(t == "ACGT");
    }
    SECTION( "cbegin/cend" )
    {   const gnx::generic_sequence<T> t("AAAA");
        auto s_it = s.begin();
        for
        (   auto t_it = t.cbegin()
        ;   t_it != t.cend()
        ;   ++t_it, ++s_it
        )
            *s_it = *t_it;
        CHECK(s == "AAAA");
    }
    SECTION( "rbegin/rend" )
    {   gnx::generic_sequence<T> t("AAAA");
        auto s_it = s.cbegin();
        for
        (   auto t_it = t.rbegin()
        ;   t_it != t.rend()
        ;   ++t_it, ++s_it
        )
            *t_it = *s_it;
        CHECK(t == "TGCA");
    }
    SECTION( "crbegin/crend" )
    {   const gnx::generic_sequence<T> t("ACGT");
        auto s_it = s.begin();
        for
        (   auto t_it = t.crbegin()
        ;   t_it != t.crend()
        ;   ++t_it, ++s_it
        )
            *s_it = *t_it;
        CHECK(s == "TGCA");
    }

// -- capacity -----------------------------------------------------------------

    SECTION( "empty()" )
    {   gnx::generic_sequence<T> e;
        CHECK(e.empty());
        e["test"] = 1;
        CHECK(!e.empty());
        CHECK(!s.empty());
    }
    SECTION( "size()" )
    {   gnx::generic_sequence<T> e;
        CHECK(0 == e.size());
        CHECK(4 == s.size());
    }

// -- subscript operator -------------------------------------------------------

    SECTION( "subscript/array index operator" )
    {   CHECK('A' == s[0]);
        CHECK('C' == s[1]);
        CHECK('G' == s[2]);
        CHECK('T' == s[3]);
        s[3] = 'U';
        CHECK('U' == s[3]);
    }

// -- subseq operator ----------------------------------------------------------

    SECTION( "subseq operator" )
    {   gnx::generic_sequence<T> org{"CCATACGTGAC"};
        CHECK(org(4, 4) == s);
        CHECK(org(0) == org);
        CHECK(org(4) == "ACGTGAC");
        CHECK_THROWS_AS(org(20) == "ACGTGAC", std::out_of_range);

        // casting sq_view to generic_sequence
        gnx::generic_sequence<T> sub = gnx::generic_sequence<T>(org(4, 10));
        CHECK(sub == "ACGTGAC");
    }

// -- managing tagged data -----------------------------------------------------

    SECTION( "tagged data" )
    {   CHECK(s.has("test-int"));
        CHECK(false == s.has("no"));

        s["int"] = 19;
        CHECK(s.has("int"));
        CHECK(19 == std::get<int>(s["int"]));

        s["float"] = 3.14f;
        CHECK(s.has("float"));
        CHECK(3.14f == std::get<float>(s["float"]));

        s["double"] = 3.14;
        CHECK(s.has("double"));
        CHECK(3.14 == std::get<double>(s["double"]));

        s["string"] = std::string("hello");
        CHECK(s.has("string"));
        CHECK("hello" == std::get<std::string>(s["string"]));

        sul::dynamic_bitset<> bits(12, 0b0100010110111);
        s["dynamic_bitset"] = bits;
        CHECK(s.has("dynamic_bitset"));
        CHECK(bits == std::get<sul::dynamic_bitset<>>(s["dynamic_bitset"]));

        std::string lvalue_tag{"check_lvalue_tag"};
        s[lvalue_tag] = 42;
        CHECK(s.has(lvalue_tag));
        CHECK(42 == std::get<int>(s[lvalue_tag]));
    }

// -- i/o operators ------------------------------------------------------------

    SECTION( "i/o operators")
    {   s["test-void"] = {};
        s["test-bool"] = true;
        s["test-unsigned"] = 33u;
        s["test-float"] = 3.14f;
        s["test-double"] = 3.14;
        s["test-string"] = std::string("hello");
        s["dynamic_bitset"] = sul::dynamic_bitset<>(12, 0b0100010110111);

        std::stringstream ss;
        ss << s;
        gnx::generic_sequence<T> t;
        ss >> t;

        CHECK(s == t);
        CHECK(s.has("test-void"));
        CHECK(t.has("test-void"));
        CHECK(std::get<bool>(s["test-bool"]) == std::get<bool>(t["test-bool"]));
        CHECK(std::get<int>(s["test-int"]) == std::get<int>(t["test-int"]));
        CHECK(std::get<unsigned>(s["test-unsigned"]) == std::get<unsigned>(t["test-unsigned"]));
        CHECK(std::get<float>(s["test-float"]) == std::get<float>(t["test-float"]));
        CHECK(std::get<double>(s["test-double"]) == std::get<double>(t["test-double"]));
        CHECK(std::get<std::string>(s["test-string"]) == std::get<std::string>(t["test-string"]));
        CHECK(sul::dynamic_bitset<>(12, 0b0100010110111) == std::get<sul::dynamic_bitset<>>(t["dynamic_bitset"]));
    }

// -- string literal operator --------------------------------------------------

    SECTION( "string literal operator" )
    {   auto t = "ACGT"_sq;
        CHECK(t == "ACGT");
        CHECK(t == "ACGT"_sq);
    }
}
