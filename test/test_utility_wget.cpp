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
    {   REQUIRE_FALSE(result.file_path().empty());
        REQUIRE_FALSE(result().empty());
        REQUIRE(result.file_path().filename() == "GCF_000204255.1_ASM20425v1_genomic.fna.gz");
        REQUIRE(std::filesystem::exists(result.file_path()));
    }

    SECTION( "file content" )
    {   auto file_size = std::filesystem::file_size(result.file_path());
        REQUIRE(file_size == 351022);
        T seq{result()};
        REQUIRE_FALSE(seq.empty());
        REQUIRE(seq.size() == 1171667);
        CHECK("NC_017287.1" == std::get<std::string>(seq["_id"]));
        CHECK(seq(0, 10) == "TATATAAATA");
    }

    SECTION( "invalid URL" )
    {   REQUIRE_THROWS_AS(gnx::wget("invalid_url"), std::invalid_argument);
    }

    SECTION( "unsupported URL scheme" )
    {   REQUIRE_THROWS_AS(gnx::wget("ftpz://example.com/file.txt"), std::invalid_argument);
    }

    SECTION( "non-existent URL" )
    {   REQUIRE_THROWS_AS(gnx::wget("ftp://ftp.ncbi.nlm.nih.gov/test.html"), std::runtime_error);
    }

    SECTION( "temporary file cleanup" )
    {   auto temp_file_path = result.file_path();
        REQUIRE(std::filesystem::exists(temp_file_path));
        result.close();
        REQUIRE_FALSE(std::filesystem::exists(temp_file_path));
    }

    SECTION( "genome URL construction" )
    {   auto genome_url = "genome://GCF_000204255.1_ASM20425v1";
        auto constructed_url = gnx::detail::construct_genome_url(genome_url);
        REQUIRE(constructed_url == "ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/204/255/GCF_000204255.1_ASM20425v1/GCF_000204255.1_ASM20425v1_genomic.fna.gz");
    }

    SECTION( "genome URL download" )
    {   auto genome_url = "genome://GCF_000204255.1_ASM20425v1";
        auto result = gnx::wget(genome_url);
        REQUIRE_FALSE(result.file_path().empty());
        REQUIRE_FALSE(result().empty());
        REQUIRE(result.file_path().filename() == "GCF_000204255.1_ASM20425v1_genomic.fna.gz");
        REQUIRE(std::filesystem::exists(result.file_path()));
    }

#if defined(GNX_USE_ASIO)
    SECTION( "HTTP download (redirects to HTTPS)" )
    {   const auto http_url{"http://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/204/255/GCF_000204255.1_ASM20425v1/GCF_000204255.1_ASM20425v1_genomic.fna.gz"};
        auto result = gnx::wget(http_url);
        REQUIRE_FALSE(result.file_path().empty());
        REQUIRE_FALSE(result().empty());
        REQUIRE(result.file_path().filename() == "GCF_000204255.1_ASM20425v1_genomic.fna.gz");
        REQUIRE(std::filesystem::exists(result.file_path()));
        auto file_size = std::filesystem::file_size(result.file_path());
        REQUIRE(file_size == 351022);
        T seq{result()};
        REQUIRE(seq.size() == 1171667);
        CHECK("NC_017287.1" == std::get<std::string>(seq["_id"]));
    }

    SECTION( "HTTPS download" )
    {   const auto https_url{"https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/204/255/GCF_000204255.1_ASM20425v1/GCF_000204255.1_ASM20425v1_genomic.fna.gz"};
        auto result = gnx::wget(https_url);
        REQUIRE_FALSE(result.file_path().empty());
        REQUIRE_FALSE(result().empty());
        REQUIRE(result.file_path().filename() == "GCF_000204255.1_ASM20425v1_genomic.fna.gz");
        REQUIRE(std::filesystem::exists(result.file_path()));
        auto file_size = std::filesystem::file_size(result.file_path());
        REQUIRE(file_size == 351022);
        T seq{result()};
        REQUIRE(seq.size() == 1171667);
        CHECK("NC_017287.1" == std::get<std::string>(seq["_id"]));
    }
#endif

    SECTION( "SRA URL construction 9" )
    {   auto constructed_url = gnx::detail::construct_sra_url("sra://SRR123456");
        REQUIRE(constructed_url == "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR123/SRR123456/SRR123456.fastq.gz");
        constructed_url = gnx::detail::construct_sra_url("sra://SRR123456_1");
        REQUIRE(constructed_url == "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR123/SRR123456/SRR123456_1.fastq.gz");
    }

    SECTION( "SRA URL construction 10" )
    {   auto constructed_url = gnx::detail::construct_sra_url("sra://SRR1234567");
        REQUIRE(constructed_url == "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR123/007/SRR1234567/SRR1234567.fastq.gz");
        constructed_url = gnx::detail::construct_sra_url("sra://SRR1234567_1");
        REQUIRE(constructed_url == "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR123/007/SRR1234567/SRR1234567_1.fastq.gz");
    }

    SECTION( "SRA URL construction 11" )
    {   auto constructed_url = gnx::detail::construct_sra_url("sra://SRR12345678");
        REQUIRE(constructed_url == "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR123/078/SRR12345678/SRR12345678.fastq.gz");
        constructed_url = gnx::detail::construct_sra_url("sra://SRR12345678_1");
        REQUIRE(constructed_url == "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR123/078/SRR12345678/SRR12345678_1.fastq.gz");
    }

    SECTION( "SRA URL construction 12" )
    {   auto constructed_url = gnx::detail::construct_sra_url("sra://SRR123456789");
        REQUIRE(constructed_url == "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR123/789/SRR123456789/SRR123456789.fastq.gz");
        constructed_url = gnx::detail::construct_sra_url("sra://SRR123456789_1");
        REQUIRE(constructed_url == "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR123/789/SRR123456789/SRR123456789_1.fastq.gz");
    }

    SECTION( "SRA URL download" )
    {   auto sra_url = "sra://SRR10190173_1";
        auto result = gnx::wget(sra_url);
        REQUIRE_FALSE(result.file_path().empty());
        REQUIRE_FALSE(result().empty());
        REQUIRE(result.file_path().filename() == "SRR10190173_1.fastq.gz");
        REQUIRE(std::filesystem::exists(result.file_path()));
    }
}
