// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <filesystem>

#include <fmt/core.h>
#include <fmt/format.h>

#include <gnx/sq.hpp>
#include <gnx/lut/ansi.hpp>
#include <gnx/lut/color_schemes.hpp>
#include <gnx/backend/forward_stream.hpp>
#include <gnx/backend/virtual_vector.hpp>
#include <gnx/algorithms/is_peptide.hpp>
#include <gnx/algorithms/count.hpp>

namespace gnx::detail {

// human-readable size formatter
std::string hrs(std::size_t size, bool peptide = false)
{   if (size < 1000)
    {   return fmt::format
        (   "{}{:3}{} {}"
        ,   gnx::ansi::ESC[style::bold]
        ,   size
        ,   gnx::ansi::ESC[style::reset]
        ,   peptide ? "aa" : "bp"
        );
    }
    else
    {   int suffix_index{0};
        double human_size = static_cast<double>(size);
        while (human_size >= 1000 && suffix_index < 4)
        {   human_size /= 1000;
            ++suffix_index;
        }
        return fmt::format
        (   "{}{:.1f}{} {}{} ({}{:L}{})"
        ,   gnx::ansi::ESC[style::bold]
        ,   human_size
        ,   gnx::ansi::ESC[style::reset]
        ,   " KMGT"[suffix_index]
        ,   peptide ? "aa" : "bp"
        ,   gnx::ansi::ESC[style::bold]
        ,   size
        ,   gnx::ansi::ESC[style::reset]
        );
    }
};

inline void describe_vv(std::string_view filename)
{   gnx::virtual_vector<gnx::sq> vv{filename};

    if (vv.empty())
    {   std::cerr
        <<  fmt::format
            (   "gnx::describe(): {} is not a supported file format\n"
            ,   filename
            );
        return;
    }

    std::size_t count{0}, total_length{0};
    fmt::memory_buffer buf;
    fmt::format_to(std::back_inserter(buf)
    ,   "▼ {}{}{}\n"
    ,   gnx::ansi::ESC[fg::bright_green]
    ,   std::filesystem::path(filename).filename().string()
    ,   gnx::ansi::ESC[fg::reset]
    );
    for (std::size_t i = 0; i < vv.size(); ++i)
    {   fmt::format_to(std::back_inserter(buf)
        ,   "{}├── ▼ {}{}{}\n"
        ,   gnx::ansi::vga::fg::ESC[250]
        ,   gnx::ansi::ESC[fg::bright_magenta]
        ,   vv.entry(i).name
        ,   gnx::ansi::ESC[fg::reset]
        );
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "{}│   └──{} {}{}{}\n"
        ,   gnx::ansi::vga::fg::ESC[250]
        ,   gnx::ansi::ESC[fg::reset]
        ,   gnx::ansi::ESC[style::bold]
        ,   hrs(vv.entry(i).length)
        ,   gnx::ansi::ESC[style::reset]
        );
        total_length += vv.entry(i).length;
    }
    fmt::format_to
    (   std::back_inserter(buf)
    ,   "{}└──{} {}{}{} {}, {}{}{}\n"
    ,   gnx::ansi::vga::fg::ESC[250]
    ,   gnx::ansi::ESC[fg::reset]
    ,   gnx::ansi::ESC[style::bold]
    ,   vv.size()
    ,   gnx::ansi::ESC[style::reset]
    ,   (vv.size() > 1) ? "sequences" : "sequence"
    ,   gnx::ansi::ESC[style::bold]
    ,   hrs(total_length)
    ,   gnx::ansi::ESC[style::reset]
    );
    std::cout.write(buf.data(), buf.size());
}

inline void describe_fs(std::string_view filename)
{   gnx::forward_stream<gnx::sq> stream{filename};
    auto it     = stream.begin();
    auto end_it = stream.end();

    if (it == end_it || it->sequence().empty())
    {   std::cerr
        <<  fmt::format
            (   "gnx::describe(): {} is not a supported file format\n"
            ,   filename
            );
        return;
    }

    bool is_fastq = !stream.quality().empty();
    bool peptide
    =   is_fastq
    ?   false
    :   is_peptide(std::begin(it->sequence()), std::end(it->sequence()));

    if (is_fastq)
    {   std::size_t count{0}, total_length{0}, min_length{0}, max_length{0};
        bool phred33{false};
        fmt::memory_buffer buf;
        fmt::format_to(std::back_inserter(buf)
        ,   "{}▼ {}{}{}\n"
        ,   gnx::ansi::vga::fg::ESC[250]
        ,   gnx::ansi::ESC[fg::bright_green]
        ,   std::filesystem::path(filename).filename().string()
        ,   gnx::ansi::ESC[fg::reset]
        );
        for (; it != end_it; ++it, ++count)
        {   min_length
            =   (count == 0)
            ?   it->sequence().size()
            :   std::min(min_length, it->sequence().size());
            max_length
            =   (count == 0)
            ?   it->sequence().size()
            :   std::max(max_length, it->sequence().size());
            if (phred33 == false && count < 10000)
            {   for (char c : stream.quality())
                {   if (c < 58 || c > 1)
                    {   phred33 = true;
                        break;
                    }
                }
            }
            total_length += it->sequence().size();
        }
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "{}├──{} Encoding: {}{}{}\n"
        ,   gnx::ansi::vga::fg::ESC[250]
        ,   gnx::ansi::ESC[fg::reset]
        ,   gnx::ansi::ESC[fg::bright_cyan]
        +   gnx::ansi::ESC[style::italic]
        ,   phred33
            ?   min_length > 1000
                ?   "PacBio / Nanopore"
                :   "Sanger / Illumina 1.8+"
            :   "Illumina 1.3 - 1.7"
        ,   gnx::ansi::ESC[style::reset]
        );
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "{}├──{} Total reads: {}{}{}\n"
        ,   gnx::ansi::vga::fg::ESC[250]
        ,   gnx::ansi::ESC[fg::reset]
        ,   gnx::ansi::ESC[style::bold]
        ,   count
        ,   gnx::ansi::ESC[style::reset]
        );
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "{}├──{} Total bases: {}{}{}\n"
        ,   gnx::ansi::vga::fg::ESC[250]
        ,   gnx::ansi::ESC[fg::reset]
        ,   gnx::ansi::ESC[style::bold]
        ,   hrs(total_length, peptide)
        ,   gnx::ansi::ESC[style::reset]
        );
        if (max_length > min_length)
            fmt::format_to
            (   std::back_inserter(buf)
            ,   "{}└──{} Read length: {}{}-{}{}\n"
            ,   gnx::ansi::vga::fg::ESC[250]
            ,   gnx::ansi::ESC[fg::reset]
            ,   gnx::ansi::ESC[style::bold]
            ,   min_length
            ,   max_length
            ,   gnx::ansi::ESC[style::reset]
            );
        else
            fmt::format_to
            (   std::back_inserter(buf)
            ,   "{}└──{} Read length: {}{}{} bp\n"
            ,   gnx::ansi::vga::fg::ESC[250]
            ,   gnx::ansi::ESC[fg::reset]
            ,   gnx::ansi::ESC[style::bold]
            ,   min_length
            ,   gnx::ansi::ESC[style::reset]
            );
        std::cout.write(buf.data(), buf.size());
    }
    else  // fastq format containing reads
    {   std::size_t count{0}, total_length{0};
        fmt::memory_buffer buf;
        fmt::format_to(std::back_inserter(buf)
        ,   "{}▼ {}{}{}\n"
        ,   gnx::ansi::vga::fg::ESC[250]
        ,   gnx::ansi::ESC[fg::bright_green]
        ,   std::filesystem::path(filename).filename().string()
        ,   gnx::ansi::ESC[fg::reset]
        );
        for (; it != end_it; ++it, ++count)
        {   fmt::format_to(std::back_inserter(buf)
            ,   "{}├── ▼ {}{}{}\n"
            ,   gnx::ansi::vga::fg::ESC[250]
            ,   gnx::ansi::ESC[fg::bright_magenta]
            ,   it->id()
            ,   gnx::ansi::ESC[fg::reset]
            );
            if (it->description().size() > 0)
                fmt::format_to
                (   std::back_inserter(buf)
                ,   "{}│   ├── {}{}{}\n"
                ,   gnx::ansi::vga::fg::ESC[250]
                ,   gnx::ansi::ESC[fg::bright_cyan]
                +   gnx::ansi::ESC[style::italic]
                ,   it->description()
                ,   gnx::ansi::ESC[style::reset]
                );
            fmt::format_to
            (   std::back_inserter(buf)
            ,   "{}│   └──{} {}{}{}\n"
            ,   gnx::ansi::vga::fg::ESC[250]
            ,   gnx::ansi::ESC[fg::reset]
            ,   gnx::ansi::ESC[style::bold]
            ,   hrs(it->sequence().size(), peptide)
            ,   gnx::ansi::ESC[style::reset]
            );
            total_length += it->sequence().size();
        }
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "{}└──{} {}{}{} {}, {}{}{}\n"
        ,   gnx::ansi::vga::fg::ESC[250]
        ,   gnx::ansi::ESC[fg::reset]
        ,   gnx::ansi::ESC[style::bold]
        ,   count
        ,   gnx::ansi::ESC[style::reset]
        ,   (count > 1) ? "sequences" : "sequence"
        ,   gnx::ansi::ESC[style::bold]
        ,   hrs(total_length, peptide)
        ,   gnx::ansi::ESC[style::reset]
        );
        std::cout.write(buf.data(), buf.size());
    }
}

template<typename Iterator>
void describe(Iterator begin, Iterator end)
{   bool peptide = is_peptide(begin, end);
    auto total_length = std::distance(begin, end);
    auto counts = gnx::count(begin, end);
    fmt::memory_buffer buf;
    for (const auto& [symbol, count] : counts)
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "{}├──{} [{}{}] {}\n"
        ,   gnx::ansi::vga::fg::ESC[250]
        ,   gnx::ansi::ESC[fg::reset]
        ,   peptide
            ?   color_scheme::aa_clustal[static_cast<uint8_t>(symbol)]
            :   color_scheme::na[static_cast<uint8_t>(symbol)]
        ,   gnx::ansi::ESC[style::reset]
        ,   hrs(count, peptide)
        );
    fmt::format_to
    (   std::back_inserter(buf)
    ,   "{}└──{} [Σ] {}\n"
    ,   gnx::ansi::vga::fg::ESC[250]
    ,   gnx::ansi::ESC[fg::reset]
    ,   hrs(total_length, peptide)
    );
    std::cout.write(buf.data(), buf.size());
}

}  // namespace gnx::detail