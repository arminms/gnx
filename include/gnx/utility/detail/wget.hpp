// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <gnx/utility/detail/knetfile.h>

#include <fmt/base.h>
#include <fmt/format.h>

#include <string_view>
#include <filesystem>
#include <stdexcept>

namespace gnx::detail
{

struct wget_result
{   wget_result() = default;
    wget_result(wget_result const&) = delete;
    wget_result& operator=(wget_result const&) = delete;
    wget_result(wget_result&&) = default;
    wget_result& operator=(wget_result&&) = default;
    wget_result(std::filesystem::path const& temp_file_path)
    :   _temp_file_path(temp_file_path)
    ,   _file_path(temp_file_path.string())
    {}
    ~wget_result()
    {   if (std::filesystem::exists(_temp_file_path))
            std::filesystem::remove(_temp_file_path);
    }
    std::string_view operator() () const noexcept
    {   return _file_path;
    }
    std::filesystem::path file_path() const noexcept
    {   return _temp_file_path;
    }
    void close() noexcept
    {   if (std::filesystem::exists(_temp_file_path))
            std::filesystem::remove(_temp_file_path);
        _file_path.clear();
    }

private:
    std::filesystem::path _temp_file_path;
    std::string _file_path;
};

[[nodiscard]]
inline bool is_valid_url(std::string_view url)
{   return
    (   url.starts_with("ftp://")
    ||  url.starts_with("http://")
    ||  url.starts_with("https://")
    ||  url.starts_with("genome://")
    ||  url.starts_with("sra://")
    );  
}

[[nodiscard]]
inline std::string construct_genome_url(std::string_view genome_acn_n_assembly)
{   genome_acn_n_assembly.remove_prefix(9); // remove "genome://"
    return fmt::format
    (   "ftp://ftp.ncbi.nlm.nih.gov/genomes/all/{}/{}/{}/{}/{}/{}_genomic.fna.gz"
    ,   genome_acn_n_assembly.substr(0, 3)
    ,   genome_acn_n_assembly.substr(4, 3)
    ,   genome_acn_n_assembly.substr(7, 3)
    ,   genome_acn_n_assembly.substr(10, 3)
    ,   genome_acn_n_assembly
    ,   genome_acn_n_assembly
    );
}

[[nodiscard]]
inline std::string construct_sra_url(std::string_view run_acn)
{   std::string result, base{"ftp://ftp.sra.ebi.ac.uk/vol1/fastq"};
    run_acn.remove_prefix(6); // remove "sra://"
    auto pos = run_acn.rfind('_');
    auto accession
    =   pos == std::string_view::npos
    ?   std::string(run_acn)
    :   std::string(run_acn.substr(0, pos));
    if (accession.size() == 9)
        result = fmt::format
        (   "{}/{}/{}/{}.fastq.gz"
        ,   base
        ,   accession.substr(0, 6)
        ,   accession
        ,   run_acn
        );
    else if (accession.size() == 10)
        result = fmt::format
        (   "{}/{}/{:0>3}/{}/{}.fastq.gz"
        ,   base
        ,   accession.substr(0, 6)
        ,   accession.substr(9, 1)
        ,   accession
        ,   run_acn
        );
    else if (accession.size() == 11)
        result = fmt::format
        (   "{}/{}/{:0>3}/{}/{}.fastq.gz"
        ,   base
        ,   accession.substr(0, 6)
        ,   accession.substr(9, 2)
        ,   accession
        ,   run_acn
        );
    else if (accession.size() >= 12)
        result = fmt::format
        (   "{}/{}/{}/{}/{}.fastq.gz"
        ,   base
        ,   accession.substr(0, 6)
        ,   accession.substr(9, 3)
        ,   accession
        ,   run_acn
        );
    else
        throw std::invalid_argument
        (   fmt::format
            (   "Invalid SRA accession in '{}'. Expected format: "
                "'sra://SRR12345678' or 'sra://SRR12345678_1'"
            ,   run_acn
            )
        );
    
    return result;
}

std::string human_readable_size(double size)
{   int suffix_index{0};
    for (; size >= 1024.; size /= 1024., ++suffix_index);
    return fmt::format
    (   "{:.1f} {}B"
    ,   std::ceil(size * 10.) / 10.
    ,   "BKMGTPE"[suffix_index]
    );
};


} // namespace gnx::detail