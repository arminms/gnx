// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#include <catch2/catch_all.hpp>

#include <gnx/psq.hpp>
#include <gnx/algorithms/random.hpp>

const uint64_t seed_pi{3141592654};

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
