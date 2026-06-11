// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#include <catch2/catch_all.hpp>

#include <gnx/sq.hpp>
#include <gnx/utility/wget.hpp>

const uint64_t seed_pi{3141592654};

// =============================================================================
// wget() utility tests
// =============================================================================

TEMPLATE_TEST_CASE
(   "gnx::wget()"
,   "[utility][wget]"
,   gnx::sq
)
{   typedef TestType T;
    const auto url{"ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/204/255/GCF_000204255.1_ASM20425v1/GCF_000204255.1_ASM20425v1_genomic.fna.gz"};
    auto result = gnx::wget(url);

    SECTION( "result.temp_file_path" )
    {   REQUIRE_FALSE(result.temp_file_path.empty());
        REQUIRE(result.temp_file_path.filename() == "GCF_000204255.1_ASM20425v1_genomic.fna.gz");
        REQUIRE(std::filesystem::exists(result.temp_file_path));
    }

    SECTION( "file content" )
    {   auto file_size = std::filesystem::file_size(result.temp_file_path);
        REQUIRE(file_size == 351022);
        T seq{result.temp_file_path.string()};
        REQUIRE_FALSE(seq.empty());
        REQUIRE(seq.size() == 1171667);
        CHECK("NC_017287.1" == std::any_cast<std::string>(seq["_id"]));
        CHECK(seq(0, 10) == "TATATAAATA");
    }
}
