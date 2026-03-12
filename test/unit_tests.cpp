// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2025 Armin Sobhani
//
#include <catch2/catch_all.hpp>

#include <utility>

#include <gnx/sq.hpp>
#include <gnx/views.hpp>
#include <gnx/psq.hpp>
#include <gnx/sqb.hpp>
#include <gnx/interface/forward_stream.hpp>
#include <gnx/io/fastaqz.hpp>
#include <gnx/algorithms/valid.hpp>
#include <gnx/algorithms/random.hpp>
#include <gnx/algorithms/compare.hpp>
#include <gnx/algorithms/local_align.hpp>
#include <gnx/algorithms/count.hpp>
#include <gnx/algorithms/complement.hpp>
#include <gnx/utility/gc-content.hpp>

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
        s.save(filename, gnx::out::fasta());
        t.load(filename);
        CHECK(s == t);
        std::remove(filename.c_str());
    }
    SECTION( "save fasta.gz" )
    {   s.load(SAMPLE_GENOME, 1);
        std::string filename = "test_output.fa.gz";
        s.save(filename, gnx::out::fasta_gz());
        t.load(filename);
        CHECK(s == t);
        std::remove(filename.c_str());
    }
    SECTION( "save fastq" )
    {   s.load(SAMPLE_READS);
        std::string filename = "test_reads.fq";
        s.save(filename, gnx::out::fastq());
        t.load(filename);
        CHECK(s == t);
        std::remove(filename.c_str());
    }
    SECTION( "save fastq.gz" )
    {   s.load(SAMPLE_READS);
        std::string filename = "test_reads.fqz";
        s.save(filename, gnx::out::fastq_gz());
        t.load(filename);
        CHECK(s == t);
        std::remove(filename.c_str());
    }
}

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
    {   gnx::rand(s.begin(), 20, "ACGT", {35, 15, 15, 35}, seed_pi);
        CHECK(gnx::valid_nucleotide(s));
        CHECK(s == "TTCTTAAGTCTTTAAACACG");
        auto t = gnx::random::dna<decltype(s)>(20, 30, seed_pi);
        t[2] = 'C';
        CHECK(s == t);
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

// =============================================================================
// local_align_n() algorithm tests
// =============================================================================

TEMPLATE_TEST_CASE
(   "gnx::local_align_n"
,   "[algorithm][local_align]"
,   std::vector<char>
)
{   typedef TestType T;

// -- basic alignment ----------------------------------------------------------

    SECTION( "identical sequences" )
    {   gnx::generic_sequence<T> s1{"ACGT"};
        gnx::generic_sequence<T> s2{"ACGT"};
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 8);  // 4 matches * 2
        CHECK(result.aligned_seq1 == "ACGT");
        CHECK(result.aligned_seq2 == "ACGT");
        CHECK(result.traceback.size() == 4);
        for (const auto& dir : result.traceback)
            CHECK(dir == gnx::alignment_direction::diagonal);
    }

    SECTION( "single mismatch" )
    {   gnx::generic_sequence<T> s1{"ACGT"};
        gnx::generic_sequence<T> s2{"ACAT"};
        auto result = gnx::local_align_n(s1, s2);

        // Best local alignment should still find matching regions
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1.length() == result.aligned_seq2.length());
    }

    SECTION( "no alignment (completely different)" )
    {   gnx::generic_sequence<T> s1{"AAAA"};
        gnx::generic_sequence<T> s2{"TTTT"};
        auto result = gnx::local_align_n(s1, s2, 2, -3, -1);

        // With strong mismatch penalty, may have low or zero score
        CHECK(result.score >= 0);  // Smith-Waterman never goes negative
    }

// -- subsequence alignment ----------------------------------------------------

    SECTION( "subsequence in larger sequence" )
    {   gnx::generic_sequence<T> s1{"ACGTACGT"};
        gnx::generic_sequence<T> s2{"ACGT"};
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 8);  // Perfect match of ACGT
        CHECK(result.aligned_seq1 == "ACGT");
        CHECK(result.aligned_seq2 == "ACGT");
    }

    SECTION( "overlapping sequences" )
    {   gnx::generic_sequence<T> s1{"ACGTACGT"};
        gnx::generic_sequence<T> s2{"TACGTACG"};
        auto result = gnx::local_align_n(s1, s2);

        // Should find significant alignment
        CHECK(result.score > 10);
        CHECK(result.aligned_seq1.length() > 5);
    }

// -- gap handling -------------------------------------------------------------

    SECTION( "alignment with gap in first sequence" )
    {   gnx::generic_sequence<T> s1{"ACGT"};
        gnx::generic_sequence<T> s2{"ACGGT"};
        auto result = gnx::local_align_n(s1, s2, 2, -1, -1);

        CHECK(result.score >= 4);  // At least some matches
        // May or may not have gap depending on scoring
    }

    SECTION( "alignment with gap in second sequence" )
    {   gnx::generic_sequence<T> s1{"ACGGT"};
        gnx::generic_sequence<T> s2{"ACGT"};
        auto result = gnx::local_align_n(s1, s2, 2, -1, -1);

        CHECK(result.score >= 4);
    }

// -- custom scoring -----------------------------------------------------------

    SECTION( "custom match score" )
    {   gnx::generic_sequence<T> s1{"ACGT"};
        gnx::generic_sequence<T> s2{"ACGT"};
        auto result = gnx::local_align_n(s1, s2, 5, -1, -1);

        CHECK(result.score == 20);  // 4 matches * 5
    }

    SECTION( "custom mismatch penalty" )
    {   gnx::generic_sequence<T> s1{"ACGT"};
        gnx::generic_sequence<T> s2{"TTTT"};
        auto result = gnx::local_align_n(s1, s2, 2, -10, -1);

        // Strong mismatch penalty should result in low score
        CHECK(result.score <= 2);
    }

    SECTION( "custom gap penalty" )
    {   gnx::generic_sequence<T> s1{"ACGT"};
        gnx::generic_sequence<T> s2{"ACGGT"};
        auto result = gnx::local_align_n(s1, s2, 2, -1, -5);

        // Strong gap penalty should discourage gaps
        CHECK(result.score >= 0);
    }

// -- edge cases ---------------------------------------------------------------

    SECTION( "empty first sequence" )
    {   gnx::generic_sequence<T> s1;
        gnx::generic_sequence<T> s2{"ACGT"};
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 0);
        CHECK(result.aligned_seq1.empty());
        CHECK(result.aligned_seq2.empty());
    }

    SECTION( "empty second sequence" )
    {   gnx::generic_sequence<T> s1{"ACGT"};
        gnx::generic_sequence<T> s2;
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 0);
        CHECK(result.aligned_seq1.empty());
        CHECK(result.aligned_seq2.empty());
    }

    SECTION( "both sequences empty" )
    {   gnx::generic_sequence<T> s1;
        gnx::generic_sequence<T> s2;
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 0);
        CHECK(result.aligned_seq1.empty());
        CHECK(result.aligned_seq2.empty());
    }

    SECTION( "single character sequences" )
    {   gnx::generic_sequence<T> s1{"A"};
        gnx::generic_sequence<T> s2{"A"};
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 2);  // Default match score
        CHECK(result.aligned_seq1 == "A");
        CHECK(result.aligned_seq2 == "A");
    }

    SECTION( "single character mismatch" )
    {   gnx::generic_sequence<T> s1{"A"};
        gnx::generic_sequence<T> s2{"T"};
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 0);  // SW doesn't allow negative scores
    }

// -- case insensitivity -------------------------------------------------------

    SECTION( "lowercase sequences" )
    {   gnx::generic_sequence<T> s1{"acgt"};
        gnx::generic_sequence<T> s2{"acgt"};
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 8);
        CHECK(result.aligned_seq1 == "acgt");
        CHECK(result.aligned_seq2 == "acgt");
    }

    SECTION( "mixed case sequences" )
    {   gnx::generic_sequence<T> s1{"AcGt"};
        gnx::generic_sequence<T> s2{"aCgT"};
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 8);  // Should match regardless of case
    }

// -- gnx::generic_sequence tests --------------------------------------------------------

    SECTION( "gnx::sq alignment" )
    {   auto s1 = "ACGTACGT"_sq;
        auto s2 = "TACGT"_sq;
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score > 0);
        CHECK(result.aligned_seq1.length() > 0);
        CHECK(result.aligned_seq2.length() > 0);
    }

// -- realistic biological example ---------------------------------------------

    SECTION( "realistic DNA sequences with SNP" )
    {   // Two sequences with single nucleotide polymorphism
        gnx::generic_sequence<T> s1{"ATCGATCGATCG"};
        gnx::generic_sequence<T> s2{"ATCGCTCGATCG"};  // C instead of A at position 5
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score >= 16);  // Most bases should match
        CHECK(result.aligned_seq1.length() >= 10);
    }

    SECTION( "realistic DNA with indel" )
    {   // Sequence with insertion/deletion
        gnx::generic_sequence<T> s1{"ATCGATCGATCG"};
        gnx::generic_sequence<T> s2{"ATCGTCGATCG"};  // Missing 'A' at position 5
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score > 0);
        // Should find good alignment around the indel
    }

// -- longer sequences ---------------------------------------------------------

    SECTION( "longer sequences" )
    {   gnx::generic_sequence<T> s1{"ACGTACGTACGTACGTACGTACGTACGTACGT"};
        gnx::generic_sequence<T> s2{"ACGTACGTACGTACGTACGTACGTACGTACGT"};
        auto result = gnx::local_align_n(s1, s2);

        CHECK(result.score == 64);  // 32 matches * 2
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

    SECTION( "partially matching longer sequences" )
    {   gnx::generic_sequence<T> s1{"AAAAAAACGTACGTACGTTTTTTT"};
        gnx::generic_sequence<T> s2{"ACGTACGTACGT"};
        auto result = gnx::local_align_n(s1, s2);

        // Should find the matching middle part
        CHECK(result.score == 24);  // 12 matches * 2
        CHECK(result.aligned_seq1 == "ACGTACGTACGT");
        CHECK(result.aligned_seq2 == "ACGTACGTACGT");
    }
}

// =============================================================================
// local_align_p() algorithm with substitution matrices tests
// =============================================================================

TEMPLATE_TEST_CASE
(   "gnx::local_align_p"
,   "[algorithm][peptide][local_align][blosum][pam]"
,   std::vector<char>
)
{   typedef TestType T;

// -- BLOSUM62 tests -----------------------------------------------------------

    SECTION( "BLOSUM62 - identical peptide sequences" )
    {   gnx::generic_sequence<T> s1{"ARNDCQEGHILKMFPSTWYV"};
        gnx::generic_sequence<T> s2{"ARNDCQEGHILKMFPSTWYV"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        // Score should be sum of diagonal elements for each amino acid
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

    SECTION( "BLOSUM62 - single amino acid difference" )
    {   gnx::generic_sequence<T> s1{"ARNDCQEG"};
        gnx::generic_sequence<T> s2{"ARNDCQKG"};  // E->K substitution
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        // Should align well with one mismatch
        CHECK(result.score > 20);
        CHECK(result.aligned_seq1.length() == result.aligned_seq2.length());
    }

    SECTION( "BLOSUM62 - conservative substitution" )
    {   // Leucine (L) and Isoleucine (I) are similar hydrophobic amino acids
        gnx::generic_sequence<T> s1{"ACDEFGHIKLMNPQRSTVWY"};
        gnx::generic_sequence<T> s2{"ACDEFGHIKLMNPQRSTVWY"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        CHECK(result.score > 50);
        CHECK(result.aligned_seq1 == s1);
    }

    SECTION( "BLOSUM62 - peptide with gaps" )
    {   gnx::generic_sequence<T> s1{"ARNDCQEG"};
        gnx::generic_sequence<T> s2{"ARNDQEG"};  // C removed
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62, -8);

        CHECK(result.score > 0);
        // Should align with one gap
    }

    SECTION( "BLOSUM62 - different gap penalty" )
    {   gnx::generic_sequence<T> s1{"ACDEFG"};
        gnx::generic_sequence<T> s2{"ACDEFG"};
        auto result1 = gnx::local_align_p(s1, s2, gnx::lut::blosum62, -8);
        auto result2 = gnx::local_align_p(s1, s2, gnx::lut::blosum62, -2);

        // With identical sequences, gap penalty shouldn't matter
        CHECK(result1.score == result2.score);
    }

    SECTION( "BLOSUM62 - case insensitive" )
    {   gnx::generic_sequence<T> s1{"ARNDCQEG"};
        gnx::generic_sequence<T> s2{"arndcqeg"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        // Should match regardless of case
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1.length() == result.aligned_seq2.length());
    }

// -- BLOSUM80 tests -----------------------------------------------------------

    SECTION( "BLOSUM80 - identical sequences" )
    {   gnx::generic_sequence<T> s1{"MVHLTPEEK"};
        gnx::generic_sequence<T> s2{"MVHLTPEEK"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum80);

        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

    SECTION( "BLOSUM80 vs BLOSUM62 comparison" )
    {   // BLOSUM80 is more stringent for closely related sequences
        gnx::generic_sequence<T> s1{"ACDEFG"};
        gnx::generic_sequence<T> s2{"ACDEFG"};
        auto result62 = gnx::local_align_p(s1, s2, gnx::lut::blosum62);
        auto result80 = gnx::local_align_p(s1, s2, gnx::lut::blosum80);

        // BLOSUM80 typically gives higher scores for identical sequences
        CHECK(result80.score >= result62.score);
    }

// -- BLOSUM45 tests -----------------------------------------------------------

    SECTION( "BLOSUM45 - distantly related sequences" )
    {   gnx::generic_sequence<T> s1{"ARNDCQEG"};
        gnx::generic_sequence<T> s2{"ARNDCQEG"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum45);

        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

// -- PAM250 tests -------------------------------------------------------------

    SECTION( "PAM250 - identical sequences" )
    {   gnx::generic_sequence<T> s1{"MVHLTPEEK"};
        gnx::generic_sequence<T> s2{"MVHLTPEEK"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::pam250);

        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

    SECTION( "PAM250 - with mismatches" )
    {   gnx::generic_sequence<T> s1{"ARNDCQEG"};
        gnx::generic_sequence<T> s2{"ARNDCQKG"};  // E->K substitution
        auto result = gnx::local_align_p(s1, s2, gnx::lut::pam250);

        CHECK(result.score > 0);
        CHECK(result.aligned_seq1.length() == result.aligned_seq2.length());
    }

// -- PAM120 tests -------------------------------------------------------------

    SECTION( "PAM120 - closely related sequences" )
    {   gnx::generic_sequence<T> s1{"ACDEFGHIKL"};
        gnx::generic_sequence<T> s2{"ACDEFGHIKL"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::pam120);

        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

    SECTION( "PAM120 vs PAM250 comparison" )
    {   // Different PAM matrices for different evolutionary distances
        gnx::generic_sequence<T> s1{"ACDEFG"};
        gnx::generic_sequence<T> s2{"ACDEFG"};
        auto result120 = gnx::local_align_p(s1, s2, gnx::lut::pam120);
        auto result250 = gnx::local_align_p(s1, s2, gnx::lut::pam250);

        // Both should align perfectly
        CHECK(result120.score > 0);
        CHECK(result250.score > 0);
    }

// -- PAM30 tests --------------------------------------------------------------

    SECTION( "PAM30 - very closely related sequences" )
    {   gnx::generic_sequence<T> s1{"MVHLTPEEK"};
        gnx::generic_sequence<T> s2{"MVHLTPEEK"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::pam30);

        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

// -- Realistic peptide alignment examples -------------------------------------

    SECTION( "realistic - human vs mouse hemoglobin fragment" )
    {   // Simplified example of conserved protein region
        gnx::generic_sequence<T> human{"VLSPADKTNVKAAW"};
        gnx::generic_sequence<T> mouse{"VLSAADKTNVKAAW"};  // P->A substitution
        auto result = gnx::local_align_p(human, mouse, gnx::lut::blosum62);

        // Should find good alignment despite one difference
        CHECK(result.score > 40);
        CHECK(result.aligned_seq1.length() >= 10);
    }

    SECTION( "realistic - enzyme active site comparison" )
    {   // Catalytic triad-like sequence
        gnx::generic_sequence<T> enzyme1{"HDSGICN"};
        gnx::generic_sequence<T> enzyme2{"HDSGVCN"};  // I->V conservative substitution
        auto result = gnx::local_align_p(enzyme1, enzyme2, gnx::lut::blosum62);

        // Conservative substitution should still score well
        CHECK(result.score > 20);
    }

    SECTION( "realistic - signal peptide vs mature protein" )
    {   gnx::generic_sequence<T> full_seq{"MKTIIALSYIFCLVFAACDEFGHIKL"};
        gnx::generic_sequence<T> mature{"ACDEFGHIKL"};  // After signal peptide cleavage
        auto result = gnx::local_align_p(full_seq, mature, gnx::lut::blosum62);

        // Should find the mature protein region
        CHECK(result.score > 30);
        CHECK(result.aligned_seq2 == mature);
    }

// -- ambiguous amino acids ----------------------------------------------------

    SECTION( "ambiguous amino acids - B (D or N)" )
    {   gnx::generic_sequence<T> s1{"ACDEFG"};
        gnx::generic_sequence<T> s2{"ACBEFG"};  // D->B
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        // B should have reasonable score with D
        CHECK(result.score > 0);
    }

    SECTION( "ambiguous amino acids - Z (E or Q)" )
    {   gnx::generic_sequence<T> s1{"ACDEFG"};
        gnx::generic_sequence<T> s2{"ACDQFG"};  // E->Q
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        CHECK(result.score > 0);
    }

    SECTION( "ambiguous amino acids - X (any)" )
    {   gnx::generic_sequence<T> s1{"ACDEFG"};
        gnx::generic_sequence<T> s2{"ACXEFG"};  // D->X
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        // X should have neutral or small penalty
        CHECK(result.score >= 0);
    }

// -- stop codon handling ------------------------------------------------------

    SECTION( "stop codon in sequence" )
    {   gnx::generic_sequence<T> s1{"ACDEFG*"};  // * represents stop codon
        gnx::generic_sequence<T> s2{"ACDEFG*"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        // Should handle stop codon
        CHECK(result.score > 0);
    }

// -- edge cases with matrices -------------------------------------------------

    SECTION( "empty sequences with matrix" )
    {   gnx::generic_sequence<T> s1;
        gnx::generic_sequence<T> s2{"ACDEFG"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        CHECK(result.score == 0);
        CHECK(result.aligned_seq1.empty());
        CHECK(result.aligned_seq2.empty());
    }

    SECTION( "single amino acid with matrix" )
    {   gnx::generic_sequence<T> s1{"A"};
        gnx::generic_sequence<T> s2{"A"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        // BLOSUM62[A][A] = 4
        CHECK(result.score == 4);
        CHECK(result.aligned_seq1 == "A");
        CHECK(result.aligned_seq2 == "A");
    }

// -- gnx::sq with matrices ----------------------------------------------------

    SECTION( "gnx::sq with BLOSUM62" )
    {   auto s1 = "MVHLTPEEK"_sq;
        auto s2 = "MVHLTPEEK"_sq;
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);
        
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == "MVHLTPEEK");
        CHECK(result.aligned_seq2 == "MVHLTPEEK");
    }

    SECTION( "gnx::sq with PAM250" )
    {   auto s1 = "ACDEFGHIKL"_sq;
        auto s2 = "ACDEFGHIKL"_sq;
        auto result = gnx::local_align_p(s1, s2, gnx::lut::pam250);

        CHECK(result.score > 0);
        CHECK(result.aligned_seq1.length() > 0);
    }

// -- performance test with longer sequences -----------------------------------

    SECTION( "longer peptide sequences with BLOSUM62" )
    {   gnx::generic_sequence<T> s1{"MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"};
        gnx::generic_sequence<T> s2{"MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"};
        auto result = gnx::local_align_p(s1, s2, gnx::lut::blosum62);

        // Human beta-globin, should align perfectly with itself
        CHECK(result.score > 500);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }
}

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
(   "gnx::sequence_bank"
,   "[interface][forward_stream]"
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
