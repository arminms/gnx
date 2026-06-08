// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2025 Armin Sobhani
//
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <utility>
#include <filesystem>

#include <gnx/sq.hpp>
#include <gnx/views.hpp>
#include <gnx/psq.hpp>
#include <gnx/sqb.hpp>
#include <gnx/backend/forward_stream.hpp>
#include <gnx/backend/virtual_vector.hpp>
#include <gnx/io/fastaqz.hpp>
#include <gnx/algorithms/valid.hpp>
#include <gnx/algorithms/random.hpp>
#include <gnx/algorithms/compare.hpp>
#include <gnx/algorithms/local_align.hpp>
#include <gnx/algorithms/count.hpp>
#include <gnx/algorithms/complement.hpp>
#include <gnx/utility/gc_content.hpp>

const uint64_t seed_pi{3141592654};

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
        CHECK(-33 == std::any_cast<int>(c["test-int"]));
    }
    SECTION( "move constructor" )
    {   gnx::generic_sequence<T> m(std::move(s));
        CHECK(s.empty());
        CHECK(m == gnx::generic_sequence<T>("ACGT"));
        CHECK(-33 == std::any_cast<int>(m["test-int"]));
    }
    SECTION( "initializer list" )
    {   gnx::generic_sequence<T> c{'A', 'C', 'G', 'T'};
        CHECK(c == s);
    }

// -- copy assignment operators ------------------------------------------------

    SECTION( "copy assignment operator" )
    {   gnx::generic_sequence<T> c = s;
        CHECK(c == s);
        CHECK(-33 == std::any_cast<int>(c["test-int"]));
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
        CHECK(19 == std::any_cast<int>(s["int"]));

        s["float"] = 3.14f;
        CHECK(s.has("float"));
        CHECK(3.14f == std::any_cast<float>(s["float"]));

        s["double"] = 3.14;
        CHECK(s.has("double"));
        CHECK(3.14 == std::any_cast<double>(s["double"]));

        s["string"] = std::string("hello");
        CHECK(s.has("string"));
        CHECK("hello" == std::any_cast<std::string>(s["string"]));

        std::vector<int> v{ 1, 2, 3, 4 };
        s["vector_int"] = v;
        CHECK(s.has("vector_int"));
        CHECK(v == std::any_cast<std::vector<int>>(s["vector_int"]));

        std::string lvalue_tag{"check_lvalue_tag"};
        s[lvalue_tag] = 42;
        CHECK(s.has(lvalue_tag));
        CHECK(42 == std::any_cast<int>(s[lvalue_tag]));
    }

// -- i/o operators ------------------------------------------------------------

    SECTION( "i/o operators")
    {   s["test-void"] = {};
        s["test-bool"] = true;
        s["test-unsigned"] = 33u;
        s["test-float"] = 3.14f;
        s["test-double"] = 3.14;
        s["test-string"] = std::string("hello");
        s["test-vector-int"] = std::vector<int>{ 1, 2, 3, 4 };

        std::stringstream ss;
        ss << s;
        gnx::generic_sequence<T> t;
        ss >> t;

        CHECK(s == t);
        CHECK(s.has("test-void"));
        CHECK(t.has("test-void"));
        CHECK(std::any_cast<bool>(s["test-bool"]) == std::any_cast<bool>(t["test-bool"]));
        CHECK(std::any_cast<int>(s["test-int"]) == std::any_cast<int>(t["test-int"]));
        CHECK(std::any_cast<unsigned>(s["test-unsigned"]) == std::any_cast<unsigned>(t["test-unsigned"]));
        CHECK(std::any_cast<float>(s["test-float"]) == std::any_cast<float>(t["test-float"]));
        CHECK(std::any_cast<double>(s["test-double"]) == std::any_cast<double>(t["test-double"]));
        CHECK(std::any_cast<std::string>(s["test-string"]) == std::any_cast<std::string>(t["test-string"]));
        CHECK(4 == std::any_cast<std::vector<int>>(s["test-vector-int"]).size());
    }

// -- string literal operator --------------------------------------------------

    SECTION( "string literal operator" )
    {   auto t = "ACGT"_sq;
        CHECK(t == "ACGT");
        CHECK(t == "ACGT"_sq);
    }
}

// =============================================================================
// gnx::packed_generic_sequence_2bit (psq2) class tests
// =============================================================================

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::packed_generic_sequence_2bit"
,   "[class][psq2]"
,   std::vector<uint8_t>
,   thrust::universal_vector<uint8_t>
,   thrust::device_vector<uint8_t>
)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::packed_generic_sequence_2bit"
,   "[class][psq2][rocm]"
,   std::vector<uint8_t>
,   thrust::universal_vector<uint8_t>
,   gnx::unified_vector<uint8_t>
)
#else
TEMPLATE_TEST_CASE
(   "gnx::packed_generic_sequence_2bit"
,   "[class][psq2]"
,   std::vector<uint8_t>
)
#endif
{   typedef TestType T;

    gnx::packed_generic_sequence_2bit<T> s{"ACGT"};
    s["test-int"] = -33;

// -- static helpers -----------------------------------------------------------

    SECTION( "encode/decode round-trip" )
    {   CHECK(gnx::psq2::encode('A') == 0b00u);
        CHECK(gnx::psq2::encode('C') == 0b01u);
        CHECK(gnx::psq2::encode('G') == 0b10u);
        CHECK(gnx::psq2::encode('T') == 0b11u);
        CHECK(gnx::psq2::encode('a') == 0b00u);
        CHECK(gnx::psq2::encode('c') == 0b01u);
        CHECK(gnx::psq2::encode('g') == 0b10u);
        CHECK(gnx::psq2::encode('t') == 0b11u);
        // unknown character maps to A (0)
        CHECK(gnx::psq2::encode('X') == 0b00u);

        CHECK(gnx::psq2::decode(0b00u) == 'A');
        CHECK(gnx::psq2::decode(0b01u) == 'C');
        CHECK(gnx::psq2::decode(0b10u) == 'G');
        CHECK(gnx::psq2::decode(0b11u) == 'T');
    }

    SECTION( "num_bytes" )
    {   CHECK(gnx::psq2::num_bytes(0) == 0);
        CHECK(gnx::psq2::num_bytes(1) == 1);
        CHECK(gnx::psq2::num_bytes(4) == 1);
        CHECK(gnx::psq2::num_bytes(5) == 2);
        CHECK(gnx::psq2::num_bytes(8) == 2);
        CHECK(gnx::psq2::num_bytes(9) == 3);
    }

// -- comparison operators -----------------------------------------------------

    SECTION( "comparison operators" )
    {   CHECK(  s == gnx::packed_generic_sequence_2bit<T>("ACGT"));
        CHECK(!(s == gnx::packed_generic_sequence_2bit<T>("TTTT")));
        CHECK(  s != gnx::packed_generic_sequence_2bit<T>("TTTT"));
        CHECK(!(s != gnx::packed_generic_sequence_2bit<T>("ACGT")));

        CHECK(s == "ACGT");
        CHECK("ACGT" == s);
        CHECK(s != "TTTT");
        CHECK("TTTT" != s);
    }

// -- constructors -------------------------------------------------------------

    SECTION( "default constructor" )
    {   gnx::packed_generic_sequence_2bit<T> e;
        CHECK(e.empty());
        CHECK(e.size() == 0);
        CHECK(e.byte_size() == 0);
    }

    SECTION( "count constructor (all-A)" )
    {   gnx::packed_generic_sequence_2bit<T> a4(4);
        CHECK(a4.size() == 4);
        CHECK(a4 == "AAAA");
    }

    SECTION( "count + fill constructor" )
    {   gnx::packed_generic_sequence_2bit<T> a4(4, 'A');
        CHECK(a4 == "AAAA");
        gnx::packed_generic_sequence_2bit<T> c4(4, 'C');
        CHECK(c4 == "CCCC");
        gnx::packed_generic_sequence_2bit<T> g4(4, 'G');
        CHECK(g4 == "GGGG");
        gnx::packed_generic_sequence_2bit<T> t4(4, 'T');
        CHECK(t4 == "TTTT");
        // non-multiple-of-4 lengths
        gnx::packed_generic_sequence_2bit<T> c5(5, 'C');
        CHECK(c5 == "CCCCC");
        gnx::packed_generic_sequence_2bit<T> t7(7, 'T');
        CHECK(t7 == "TTTTTTT");
    }

    SECTION( "string_view constructor" )
    {   gnx::packed_generic_sequence_2bit<T> c("ACGT");
        CHECK(s == c);
    }

    SECTION( "iterator constructor" )
    {   std::string acgt{"ACGT"};
        gnx::packed_generic_sequence_2bit<T> c(std::begin(acgt), std::end(acgt));
        CHECK(s == c);
    }

    SECTION( "initializer list constructor" )
    {   gnx::packed_generic_sequence_2bit<T> c{'A', 'C', 'G', 'T'};
        CHECK(c == s);
    }

    SECTION( "copy constructor" )
    {   gnx::packed_generic_sequence_2bit<T> c(s);
        CHECK(c == s);
        CHECK(-33 == std::any_cast<int>(c["test-int"]));
    }

    SECTION( "move constructor" )
    {   gnx::packed_generic_sequence_2bit<T> m(std::move(s));
        CHECK(s.empty());
        CHECK(m == gnx::packed_generic_sequence_2bit<T>("ACGT"));
        CHECK(-33 == std::any_cast<int>(m["test-int"]));
    }

    SECTION( "construct from generic_sequence (with tagged data)" )
    {   gnx::sq src("ACGTACGT"_sq);
        src["_id"] = std::string("read-1");
        gnx::packed_generic_sequence_2bit<T> p(src);
        CHECK(p == "ACGTACGT");
        CHECK(p.has("_id"));
        CHECK(std::any_cast<std::string>(p["_id"]) == "read-1");
    }

    SECTION( "construct from generic_sequence (no tagged data)" )
    {   gnx::sq src{"TTGCAA"};
        gnx::packed_generic_sequence_2bit<T> p(src);
        CHECK(p == "TTGCAA");
        CHECK(!p.has("_id"));
    }

// -- assignment operators -----------------------------------------------------

    SECTION( "copy assignment operator" )
    {   gnx::packed_generic_sequence_2bit<T> c = s;
        CHECK(c == s);
        CHECK(-33 == std::any_cast<int>(c["test-int"]));
    }

    SECTION( "move assignment operator" )
    {   gnx::packed_generic_sequence_2bit<T> m
    =   gnx::packed_generic_sequence_2bit<T>("ACGT");
        CHECK(m == s);
    }

    SECTION( "string_view assignment" )
    {   gnx::packed_generic_sequence_2bit<T> c;
        c = std::string_view("ACGT");
        CHECK(c == "ACGT");
    }

    SECTION( "initializer list assignment" )
    {   gnx::packed_generic_sequence_2bit<T> c = {'A', 'C', 'G', 'T'};
        CHECK(c == s);
    }

// -- conversion to generic_sequence -------------------------------------------

    SECTION( "to_sq() roundtrip" )
    {   gnx::sq src("ACGTACGT"_sq);
        src["_id"] = std::string("read-1");
        gnx::packed_generic_sequence_2bit<T> p(src);
        auto back = p.to_sq();
        CHECK(back == "ACGTACGT");
        CHECK(back.has("_id"));
        CHECK(std::any_cast<std::string>(back["_id"]) == "read-1");
    }

    SECTION( "to_sq() lossless for all bases" )
    {   const std::string bases = "ACGTACGTACGTACGT";
        gnx::packed_generic_sequence_2bit<T> p(bases);
        auto sq = p.to_sq();
        CHECK(sq == bases);
    }

// -- iterators ----------------------------------------------------------------

    SECTION( "begin/end (read)" )
    {   std::string collected;
        for (char c : s)
            collected += c;
        CHECK(collected == "ACGT");
    }

    SECTION( "begin/end (write via proxy)" )
    {   gnx::packed_generic_sequence_2bit<T> t("AAAA");
        for (auto it = t.begin(); it != t.end(); ++it)
            *it = 'T';
        CHECK(t == "TTTT");
    }

    SECTION( "cbegin/cend" )
    {   const gnx::packed_generic_sequence_2bit<T> t("ACGT");
        std::string collected;
        for (auto it = t.cbegin(); it != t.cend(); ++it)
            collected += char(*it);
        CHECK(collected == "ACGT");
    }

    SECTION( "rbegin/rend" )
    {   gnx::packed_generic_sequence_2bit<T> t("AAAA");
        auto s_it = s.cbegin();
        for
        (   auto t_it = t.rbegin()
        ;   t_it != t.rend()
        ;   ++t_it, ++s_it
        )
            *t_it = char(*s_it);
        CHECK(t == "TGCA");
    }

    SECTION( "crbegin/crend" )
    {   const gnx::packed_generic_sequence_2bit<T> t("ACGT");
        gnx::packed_generic_sequence_2bit<T> out(4, 'A');
        auto out_it = out.begin();
        for
        (   auto t_it = t.crbegin()
        ;   t_it != t.crend()
        ;   ++t_it, ++out_it
        )
            *out_it = char(*t_it);
        CHECK(out == "TGCA");
    }

    SECTION( "random access iterator arithmetic" )
    {   gnx::packed_generic_sequence_2bit<T> t("ACGTTTTT");
        auto it = t.begin();
        CHECK(char(*(it + 2)) == 'G');
        it += 3;
        CHECK(char(*it) == 'T');
        CHECK((t.end() - t.begin()) == 8);
    }

// -- capacity -----------------------------------------------------------------

    SECTION( "empty()" )
    {   gnx::packed_generic_sequence_2bit<T> e;
        CHECK(e.empty());
        e["test"] = 1;
        CHECK(!e.empty());
        CHECK(!s.empty());
    }

    SECTION( "size() and byte_size()" )
    {   gnx::packed_generic_sequence_2bit<T> e;
        CHECK(0 == e.size());
        CHECK(0 == e.byte_size());
        CHECK(4 == s.size());
        CHECK(1 == s.byte_size()); // 4 bases packed in 1 byte
        gnx::packed_generic_sequence_2bit<T> s9("ACGTACGTA");
        CHECK(9 == s9.size());
        CHECK(3 == s9.byte_size()); // ceil(9/4) = 3
    }

    SECTION( "size_in_memory()" )
    {   CHECK(s.size_in_memory() >= s.byte_size());
    }

// -- element access -----------------------------------------------------------

    SECTION( "subscript operator (read)" )
    {   CHECK('A' == char(s[0]));
        CHECK('C' == char(s[1]));
        CHECK('G' == char(s[2]));
        CHECK('T' == char(s[3]));
    }

    SECTION( "subscript operator (write)" )
    {   gnx::packed_generic_sequence_2bit<T> t("ACGT");
        t[0] = 'T';
        t[3] = 'A';
        CHECK(t == "TCGA");
    }

    SECTION( "at() with valid index" )
    {   CHECK('A' == char(s.at(0)));
        CHECK('T' == char(s.at(3)));
    }

    SECTION( "at() out of range" )
    {   CHECK_THROWS_AS(s.at(10), std::out_of_range);
        const gnx::packed_generic_sequence_2bit<T> cs("ACGT");
        CHECK_THROWS_AS(cs.at(10), std::out_of_range);
    }

    SECTION( "get_base()" )
    {   CHECK('A' == s.get_base(0));
        CHECK('C' == s.get_base(1));
        CHECK('G' == s.get_base(2));
        CHECK('T' == s.get_base(3));
    }

    SECTION( "data() pointer" )
    {
#if defined(__CUDACC__) || defined(__HIPCC__)
        if constexpr (!std::is_same_v<T, thrust::device_vector<uint8_t>>)
        {   CHECK(s.data() != nullptr);
            const gnx::packed_generic_sequence_2bit<T> cs("ACGT");
            CHECK(cs.data() != nullptr);
            CHECK(cs.data()[0] == 0x1B);
        }
#else
        CHECK(s.data() != nullptr);
        const gnx::packed_generic_sequence_2bit<T> cs("ACGT");
        CHECK(cs.data() != nullptr);
        // packed byte: A=00, C=01, G=10, T=11 -> 0b00011011 = 0x1B
        CHECK(cs.data()[0] == 0x1B);
#endif //__CUDACC__
    }

// -- managing tagged data -----------------------------------------------------

    SECTION( "tagged data" )
    {   CHECK(s.has("test-int"));
        CHECK(false == s.has("no"));

        s["int"] = 19;
        CHECK(s.has("int"));
        CHECK(19 == std::any_cast<int>(s["int"]));

        s["float"] = 3.14f;
        CHECK(s.has("float"));
        CHECK(3.14f == std::any_cast<float>(s["float"]));

        s["double"] = 3.14;
        CHECK(s.has("double"));
        CHECK(3.14 == std::any_cast<double>(s["double"]));

        s["string"] = std::string("hello");
        CHECK(s.has("string"));
        CHECK("hello" == std::any_cast<std::string>(s["string"]));

        std::string lvalue_tag{"check_lvalue_tag"};
        s[lvalue_tag] = 42;
        CHECK(s.has(lvalue_tag));
        CHECK(42 == std::any_cast<int>(s[lvalue_tag]));

        // const access to existing tag
        const gnx::packed_generic_sequence_2bit<T>& cs = s;
        CHECK(19 == std::any_cast<int>(cs["int"]));

        // const access to missing tag throws
        CHECK_THROWS_AS(cs["no_such_tag"], std::out_of_range);
    }

// -- cross-type comparison with generic_sequence ------------------------------

    SECTION( "equality with generic_sequence" )
    {   gnx::sq sq_src("ACGT");
        CHECK(s == sq_src);
        CHECK(sq_src == s);
        CHECK(!(s != sq_src));

        gnx::sq sq_other("TTTT");
        CHECK(s != sq_other);
        CHECK(sq_other != s);
    }

// -- i/o operators ------------------------------------------------------------

    SECTION( "i/o operators (stream round-trip)" )
    {   s["test-void"]       = {};
        s["test-bool"]       = true;
        s["test-unsigned"]   = 33u;
        s["test-float"]      = 3.14f;
        s["test-double"]     = 3.14;
        s["test-string"]     = std::string("hello");

        std::stringstream ss;
        ss << s;
        gnx::packed_generic_sequence_2bit<T> t;
        ss >> t;

        CHECK(s == t);
        CHECK(t.has("test-void"));
        CHECK(std::any_cast<bool>(s["test-bool"])     == std::any_cast<bool>(t["test-bool"]));
        CHECK(std::any_cast<int>(s["test-int"])       == std::any_cast<int>(t["test-int"]));
        CHECK(std::any_cast<unsigned>(s["test-unsigned"])
              == std::any_cast<unsigned>(t["test-unsigned"]));
        CHECK(std::any_cast<float>(s["test-float"])   == std::any_cast<float>(t["test-float"]));
        CHECK(std::any_cast<double>(s["test-double"]) == std::any_cast<double>(t["test-double"]));
        CHECK(std::any_cast<std::string>(s["test-string"])
              == std::any_cast<std::string>(t["test-string"]));
    }

// -- string literal operator --------------------------------------------------

    SECTION( "string literal operator" )
    {   auto t = "ACGT"_psq2;
        CHECK(t == "ACGT");
        CHECK(t == "ACGT"_psq2);
    }

// -- packing correctness across lengths ---------------------------------------

    SECTION( "packing correctness for lengths 1-16" )
    {   const std::string src = "ACGTACGTACGTACGT";
        for (std::size_t len = 1; len <= 16; ++len)
        {   gnx::packed_generic_sequence_2bit<T> p(std::string_view(src).substr(0, len));
            CHECK(p.size() == len);
            for (std::size_t i = 0; i < len; ++i)
                CHECK(p.get_base(i) == src[i]);
        }
    }

    SECTION( "padding bits do not bleed into comparisons" )
    {   // Two sequences of different lengths that share a prefix should not
        // compare as equal even if the packed bytes happen to overlap.
        gnx::packed_generic_sequence_2bit<T> a5("ACGTA");
        gnx::packed_generic_sequence_2bit<T> a4("ACGT");
        CHECK(a5 != a4);
    }

// -- proxy reference ----------------------------------------------------------

    SECTION( "proxy reference equality" )
    {   gnx::packed_generic_sequence_2bit<T> t("ACGT");
        // const ref == char
        const gnx::packed_generic_sequence_2bit<T> ct("ACGT");
        CHECK(ct[0] == 'A');
        CHECK('A' == ct[0]);
        // mutable ref copy-assign from const ref
        gnx::packed_generic_sequence_2bit<T> u("AAAA");
        u[0] = t[0]; // A->A (no-op eff.)
        u[1] = t[1]; // write 'C'
        CHECK(char(u[1]) == 'C');
    }

// -- random algorithm ---------------------------------------------------------

    SECTION( "random algorithm" )
    {   auto t = gnx::random::packed_2bit::dna<decltype(s)>(20, seed_pi);
        // t.save("test_random_psq2.fa", gnx::out::fasta());
#if defined(__CUDACC__) || defined(__HIPCC__)
        if constexpr (std::is_same_v<T, std::vector<uint8_t>>)
            CHECK(t == "GGCAACACTAGAACTCTGCT");
        else
            CHECK(gnx::valid_nucleotide(t.to_sq()));
#else
        CHECK(t == "GGCAACACTAGAACTCTGCT");
#endif
    }
}

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

// =============================================================================
// I/O tests
// =============================================================================

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::io::fastaqz"
,   "[io][in][out][cuda]"
,   gnx::generic_sequence<std::vector<char>>
,   gnx::generic_sequence<thrust::device_vector<char>>
,   gnx::packed_generic_sequence_2bit<std::vector<uint8_t>>
// ,   gnx::packed_generic_sequence_2bit<thrust::device_vector<uint8_t>>
)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::io::fastaqz"
,   "[io][in][out][rocm]"
,   gnx::generic_sequence<std::vector<char>>
,   gnx::generic_sequence<thrust::device_vector<char>>
,   gnx::packed_generic_sequence_2bit<std::vector<uint8_t>>
// ,   gnx::packed_generic_sequence_2bit<thrust::device_vector<uint8_t>>
)
#else
TEMPLATE_TEST_CASE
(   "gnx::io::fastaqz"
,   "[io][in][out]"
,   gnx::generic_sequence<std::vector<char>>
,   gnx::packed_generic_sequence_2bit<std::vector<uint8_t>>
)
#endif
{   typedef TestType SequenceType;
    std::string desc("Chlamydia psittaci 6BC plasmid pCps6BC, complete sequence");
    SequenceType s, t;
    CHECK_THROWS_AS
    (   s.load("wrong.fa")
    ,   std::runtime_error
    );

    SequenceType wrong_ndx;
    wrong_ndx.load(SAMPLE_GENOME, 3);
    CHECK(wrong_ndx.empty());
    SequenceType bad_id;
    bad_id.load(SAMPLE_GENOME, "bad_id");
    CHECK(bad_id.empty());

    // REQUIRE_THAT
    // (   gnx::lut::phred33[static_cast<uint8_t>('J')]
    // ,   Catch::Matchers::WithinAbs(7.943282e-05, 0.000001)
    // );

    SECTION( "load with index" )
    {   s.load(SAMPLE_GENOME, 1, gnx::in::fast_aqz<decltype(s)>());
        CHECK(7553 == std::size(s));
        CHECK(s(0, 10) == "TATAATTAAA");
        CHECK(s( 7543) == "TCCAATTCTA");
        CHECK("NC_017288.1" == std::any_cast<std::string>(s["_id"]));
        CHECK(desc == std::any_cast<std::string>(s["_desc"]));
    }
    SECTION( "load with id" )
    {   s.load(SAMPLE_GENOME, "NC_017288.1");
        CHECK(7553 == std::size(s));
        CHECK(s(0, 10) == "TATAATTAAA");
        CHECK(s( 7543) == "TCCAATTCTA");
        CHECK("NC_017288.1" == std::any_cast<std::string>(s["_id"]));
        CHECK(desc == std::any_cast<std::string>(s["_desc"]));
    }
    SECTION( "save fasta" )
    {   s.load(SAMPLE_GENOME, 1);
        std::string filename = "test_output.fa";
        s.save(filename);
        t.load(filename);
        CHECK(s == t);
        std::remove(filename.c_str());
    }
    SECTION( "save fasta.gz" )
    {   s.load(SAMPLE_GENOME, 1);
        std::string filename = "test_output.fa.gz";
        s.save(filename);
        t.load(filename);
        CHECK(s == t);
        std::remove(filename.c_str());
    }
    SECTION( "save fastq" )
    {   s.load(SAMPLE_READS);
        std::string filename = "test_reads.fq";
        s.save(filename);
        t.load(filename);
        CHECK(s == t);
        std::remove(filename.c_str());
    }
    SECTION( "save fastq.gz" )
    {   s.load(SAMPLE_READS);
        std::string filename = "test_reads.fq.gz";
        s.save(filename);
        t.load(filename);
        CHECK(s == t);
        std::remove(filename.c_str());
    }
}

// =============================================================================
// gc_content() and at_content() utilities tests
// =============================================================================

TEMPLATE_TEST_CASE
(   "gnx::gc_content()"
,   "[utility][gc_content][gc_ratio][at_content][at_ratio]"
,   std::vector<char>
)
{   typedef TestType T;
    const auto N{10'000};
    gnx::generic_sequence<T> s(N);
    gnx::rand(s.begin(), N, "ACGT", {35, 15, 15, 35}, seed_pi);

    SECTION( "gc-content and gc-ratio" )
    {   REQUIRE_THAT(gnx::gc_content(s), Catch::Matchers::WithinAbs(30.16, 0.001));
        REQUIRE_THAT(gnx::gc_ratio(s), Catch::Matchers::WithinAbs(0.3016, 0.001));
    }

    SECTION( "at-content and at-ratio" )
    {   REQUIRE_THAT(gnx::at_content(s), Catch::Matchers::WithinAbs(69.84, 0.001));
        REQUIRE_THAT(gnx::at_ratio(s), Catch::Matchers::WithinAbs(0.6984, 0.001));
    }
}

// =============================================================================
// forward stream sequence bank tests
// =============================================================================

TEMPLATE_TEST_CASE
(   "gnx::sequence_bank<forward_stream>"
,   "[backend][forward_stream]"
,   std::vector<char>
)
{   typedef TestType T;

    SECTION( "forward_stream" )
    {   gnx::sequence_bank sb{gnx::forward_stream<gnx::generic_sequence<T>>{SAMPLE_GENOME}};
        for (const auto& s : sb)
        {   CHECK(gnx::valid(s.sequence()));
            CHECK(s.quality().empty());  // No quality scores in this test
        }
    }
}

// =============================================================================
// fixtures for virtual_vector tests
// =============================================================================

template <typename T>
struct fai_fixture
{   using SequenceType = T;
    fai_fixture()
    {   auto tmp = std::filesystem::temp_directory_path();
        tmp_fa        = (tmp / "gnx_test.fa").string();
        tmp_fai       = (tmp / "gnx_test.fa.fai").string();

        gnx::sequence_bank sb{gnx::forward_stream<SequenceType>{SAMPLE_GENOME}};
        gnx::out::fasta out(true);
        out.open(tmp_fa);
        for (const auto& s : sb)
            out.write(s());
        out.close();
    }
    ~fai_fixture()
    {   // Clean up temporary files created during tests.
        std::filesystem::remove(tmp_fa);
        std::filesystem::remove(tmp_fai);
    }

    std::string tmp_fa;
    std::string tmp_fai;
};

template <typename T>
struct gzi_fixture
{   using SequenceType = T;
    gzi_fixture()
    {   auto tmp = std::filesystem::temp_directory_path();
        tmp_fa_gz     = (tmp / "gnx_test.fa.gz").string();
        tmp_fa_gz_fai = (tmp / "gnx_test.fa.gz.fai").string();
        tmp_fa_gz_gzi = (tmp / "gnx_test.fa.gz.gzi").string();

        gnx::sequence_bank sb{gnx::forward_stream<SequenceType>{SAMPLE_GENOME}};
        gnx::out::fasta_gz out(true);
        out.open(tmp_fa_gz);
        for (const auto& s : sb)
            out.write(s());
        out.close();
    }
    ~gzi_fixture()
    {   // Clean up temporary files created during tests.
        std::filesystem::remove(tmp_fa_gz);
        std::filesystem::remove(tmp_fa_gz_fai);
        std::filesystem::remove(tmp_fa_gz_gzi);
    }

    std::string tmp_fa_gz;
    std::string tmp_fa_gz_fai;
    std::string tmp_fa_gz_gzi;
};

TEMPLATE_TEST_CASE_METHOD
(   fai_fixture
,   "gnx::sequence_bank<virtual_vector>"
,   "[backend][virtual_vector]"
,   gnx::generic_sequence<std::vector<char>>
,   gnx::packed_generic_sequence_2bit<std::vector<uint8_t>>
)
{   typedef typename fai_fixture<TestType>::SequenceType SequenceType;

    SECTION( "size and empty" )
    {   gnx::virtual_vector<SequenceType> vv(this->tmp_fa);
        CHECK(vv.size() == 2);
        CHECK_FALSE(vv.empty());
    }

    SECTION( "auto-builds .fai when missing" )
    {   std::filesystem::remove(this->tmp_fai);             // remove any existing index
        REQUIRE_FALSE(std::ifstream(this->tmp_fai).good()); // guard: no index yet
        gnx::virtual_vector<SequenceType> vv(this->tmp_fa);
        CHECK(std::ifstream(this->tmp_fai).good());         // index was created
    }

    SECTION( "name() returns correct IDs without disk I/O" )
    {   gnx::virtual_vector<SequenceType> vv(this->tmp_fa);
        CHECK(std::string(vv.name(0)) == "NC_017287.1");
        CHECK(std::string(vv.name(1)) == "NC_017288.1");
    }

    SECTION( "entry() fields match expected FAI values" )
    {   gnx::virtual_vector<SequenceType> vv(this->tmp_fa);
        const auto& e0 = vv.entry(0);
        CHECK(e0.name   == "NC_017287.1");
        CHECK(e0.length == 1171667);
        const auto& e1 = vv.entry(1);
        CHECK(e1.name   == "NC_017288.1");
        CHECK(e1.length == 7553);
    }

    SECTION( "operator[] reads correct sequence content" )
    {   gnx::virtual_vector<SequenceType> vv(this->tmp_fa);
        auto s0 = vv[0];
        CHECK(std::size(s0) == 1171667);
        CHECK("NC_017287.1" == std::any_cast<std::string>(s0["_id"]));
        CHECK(s0(0, 10) == "TATATAAATA");
        auto s1 = vv[1];
        CHECK(std::size(s1) == 7553);
        CHECK("NC_017288.1" == std::any_cast<std::string>(s1["_id"]));
        CHECK(s1(0, 10) == "TATAATTAAA");
    }

    SECTION( "at() throws std::out_of_range for invalid index" )
    {   gnx::virtual_vector<SequenceType> vv(this->tmp_fa);
        CHECK_NOTHROW(vv.at(0));
        CHECK_NOTHROW(vv.at(1));
        CHECK_THROWS_AS(vv.at(2), std::out_of_range);
    }

    SECTION( "iterator yields valid nucleotide sequences" )
    {   gnx::virtual_vector<SequenceType> vv(this->tmp_fa);
        std::size_t count = 0;
        for (const auto& s : vv)
        {   CHECK(gnx::valid_nucleotide(s));
            ++count;
        }
        CHECK(count == 2);
    }

    SECTION( "random-access iterator arithmetic" )
    {   gnx::virtual_vector<SequenceType> vv(this->tmp_fa);
        auto it = vv.begin();
        CHECK(vv.end() - vv.begin() == 2);
        CHECK(std::size(*it) == 1171667);   // seq0
        ++it;
        CHECK(std::size(*it) == 7553);      // seq1
        --it;
        CHECK(std::size(*it) == 1171667);   // back to seq0
        CHECK(std::size(it[1]) == 7553);    // subscript offset
    }

    SECTION( "reuses existing .fai on second construction" )
    {   // First construction: builds the index
        {   gnx::virtual_vector<SequenceType> vv1(this->tmp_fa);
            CHECK(vv1.size() == 2);
        }
        // Second construction: loads the existing index (size stays correct)
        gnx::virtual_vector<SequenceType> vv2(this->tmp_fa);
        CHECK(vv2.size() == 2);
        CHECK(std::string(vv2.name(0)) == "NC_017287.1");
    }

    SECTION( "custom fai_path" )
    {   std::string custom_fai = this->tmp_fa + ".custom.fai";
        std::remove(custom_fai.c_str());
        {
            gnx::virtual_vector<SequenceType> vv(this->tmp_fa, custom_fai);
            CHECK(std::ifstream(custom_fai).good());
            CHECK(vv.size() == 2);
            CHECK(std::string(vv.name(1)) == "NC_017288.1");
        }
        std::remove(custom_fai.c_str());
    }

    SECTION( "sequence_bank integration" )
    {   gnx::sequence_bank sb
        {   gnx::virtual_vector<SequenceType>{this->tmp_fa}
        };
        std::size_t count = 0;
        for (const auto& s : sb)
        {   CHECK(gnx::valid_nucleotide(s));
            ++count;
        }
        CHECK(count == 2);
    }
}

TEMPLATE_TEST_CASE_METHOD
(   gzi_fixture
,   "gnx::sequence_bank<virtual_vector> bgzip"
,   "[backend][virtual_vector][bgzip]"
,   gnx::generic_sequence<std::vector<char>>
,   gnx::packed_generic_sequence_2bit<std::vector<uint8_t>>
)
{   typedef typename gzi_fixture<TestType>::SequenceType SequenceType;

    SECTION( "size and empty" )
    {   gnx::virtual_vector<SequenceType> vv(this->tmp_fa_gz);
        CHECK(vv.size() == 2);
        CHECK_FALSE(vv.empty());
    }

    SECTION( "name() returns correct IDs" )
    {   gnx::virtual_vector<SequenceType> vv(this->tmp_fa_gz);
        CHECK(std::string(vv.name(0)) == "NC_017287.1");
        CHECK(std::string(vv.name(1)) == "NC_017288.1");
    }

    SECTION( "entry() fields match FAI values" )
    {   gnx::virtual_vector<SequenceType> vv(this->tmp_fa_gz);
        const auto& e0 = vv.entry(0);
        CHECK(e0.name   == "NC_017287.1");
        CHECK(e0.length == 1171667);
        CHECK(e0.offset == 55);
        const auto& e1 = vv.entry(1);
        CHECK(e1.name   == "NC_017288.1");
        CHECK(e1.length == 7553);
        CHECK(e1.offset == 1186439);
    }

    SECTION( "operator[] reads correct sequences" )
    {   gnx::virtual_vector<SequenceType> vv(this->tmp_fa_gz);
        auto s0 = vv[0];
        CHECK(std::size(s0) == 1171667);
        CHECK("NC_017287.1" == std::any_cast<std::string>(s0["_id"]));
        CHECK(s0(0, 10) == "TATATAAATA");
        auto s1 = vv[1];
        CHECK(std::size(s1) == 7553);
        CHECK("NC_017288.1" == std::any_cast<std::string>(s1["_id"]));
        CHECK(s1(0, 10) == "TATAATTAAA");
    }

    SECTION( "iterator yields valid nucleotide sequences" )
    {   gnx::virtual_vector<SequenceType> vv(this->tmp_fa_gz);
        std::size_t count = 0;
        for (const auto& s : vv)
        {   CHECK(gnx::valid_nucleotide(s));
            ++count;
        }
        CHECK(count == 2);
    }

    SECTION( "auto-builds .gzi when missing" )
    {   std::filesystem::remove(this->tmp_fa_gz_gzi);
        // No .gzi present — constructor must auto-generate it.
        REQUIRE_NOTHROW(gnx::virtual_vector<SequenceType>{this->tmp_fa_gz});
        CHECK(std::ifstream(this->tmp_fa_gz_gzi).good());  // index was created
        {   gnx::virtual_vector<SequenceType> vv(this->tmp_fa_gz);
            CHECK(std::size(vv[0]) == 1171667);
            CHECK(std::size(vv[1]) == 7553);
        }
    }

    // SECTION( "save()" )
    // {   gnx::virtual_vector<SequenceType> vv(this->tmp_fa_gz);
    //     vv.save("/tmp/test3.fa.gz");
    // }

}
