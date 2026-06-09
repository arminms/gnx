// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include <gnx/sq.hpp>
#include <gnx/psq.hpp>
#include <gnx/sqb.hpp>
#include <gnx/algorithms/valid.hpp>
#include <gnx/backend/forward_stream.hpp>
#include <gnx/backend/virtual_vector.hpp>

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
