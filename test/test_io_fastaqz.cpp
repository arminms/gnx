// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#include <catch2/catch_all.hpp>

#include <gnx/sq.hpp>
#include <gnx/psq.hpp>
#include <gnx/io/fastaqz.hpp>

// =============================================================================
// I/O fastaqz tests
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
        CHECK("NC_017288.1" == std::get<std::string>(s["_id"]));
        CHECK(desc == std::get<std::string>(s["_desc"]));
    }
    SECTION( "load with id" )
    {   s.load(SAMPLE_GENOME, "NC_017288.1");
        CHECK(7553 == std::size(s));
        CHECK(s(0, 10) == "TATAATTAAA");
        CHECK(s( 7543) == "TCCAATTCTA");
        CHECK("NC_017288.1" == std::get<std::string>(s["_id"]));
        CHECK(desc == std::get<std::string>(s["_desc"]));
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
