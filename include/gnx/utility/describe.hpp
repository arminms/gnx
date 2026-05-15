// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <filesystem>

#include <fmt/core.h>
#include <fmt/format.h>

#include <gnx/sq.hpp>
#include <gnx/lut/ansi.hpp>
#include <gnx/lut/peptide.hpp>
#include <gnx/backend/forward_stream.hpp>

namespace gnx {

inline void describe(std::string_view filename)
{   gnx::forward_stream<sq> stream{filename};
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

    // human-readable size formatter
    auto hrs = [](std::size_t size, bool peptide) -> std::string
    {   if (size < 1000)
        {   return fmt::format
            (   "{} {}"
            ,   size
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
            (   "{:.1f} {}{} ({:L})"
            ,   human_size
            ,   " KMGT"[suffix_index]
            ,   peptide ? "aa" : "bp"
            ,   size
            );
        }
    };

    bool is_fastq = !stream.quality().empty();
    bool is_peptide
    =   is_fastq
    ?   false
    :   std::any_of
        (   it->sequence().begin()
        ,   it->sequence().end()
        ,   [](char c)
            {   return gnx::lut::is_peptide[static_cast<uint8_t>(c)];
            }
        );

    if (is_fastq)
    {   std::size_t count{0}, total_length{0}, min_length{0}, max_length{0};
        bool phred33{false};
        fmt::memory_buffer buf;
        fmt::format_to(std::back_inserter(buf)
        ,   "▼ {}{}{}\n"
        ,   gnx::ansi::escape[fg::bright_green]
        ,   std::filesystem::path(filename).stem().string()
        ,   gnx::ansi::escape[fg::reset]
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
        ,   "├── Encoding: {}{}{}\n"
        ,   gnx::ansi::escape[fg::bright_cyan]
        +   gnx::ansi::escape[style::italic]
        ,   phred33
            ?   min_length > 1000
                ?   "PacBio / Nanopore"
                :   "Sanger / Illumina 1.8+"
            :   "Illumina 1.3 - 1.7"
        ,   gnx::ansi::escape[style::reset]
        );
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "├── Total reads: {}{}{}\n"
        ,   gnx::ansi::escape[fg::bright_yellow]
        ,   count
        ,   gnx::ansi::escape[fg::reset]
        );
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "├── Total bases: {}{}{}\n"
        ,   gnx::ansi::escape[fg::bright_yellow]
        ,   hrs(total_length, is_peptide)
        ,   gnx::ansi::escape[fg::reset]
        );
        if (max_length > min_length)
            fmt::format_to
            (   std::back_inserter(buf)
            ,   "└── Read length: {}{}-{}{}\n"
            ,   gnx::ansi::escape[fg::bright_yellow]
            ,   min_length
            ,   max_length
            ,   gnx::ansi::escape[fg::reset]
            );
        else
            fmt::format_to
            (   std::back_inserter(buf)
            ,   "└── Read length: {}{}{} bp\n"
            ,   gnx::ansi::escape[fg::bright_yellow]
            ,   min_length
            ,   gnx::ansi::escape[fg::reset]
            );
        std::cout.write(buf.data(), buf.size());
    }
    else  // fastq format containing reads
    {   std::size_t count{0}, total_length{0};
        fmt::memory_buffer buf;
        fmt::format_to(std::back_inserter(buf)
        ,   "▼ {}{}{}\n"
        ,   gnx::ansi::escape[fg::bright_green]
        ,   std::filesystem::path(filename).stem().string()
        ,   gnx::ansi::escape[fg::reset]
        );
        for (; it != end_it; ++it, ++count)
        {   fmt::format_to(std::back_inserter(buf)
            ,   "├── ▼ {}{}{}\n"
            ,   gnx::ansi::escape[fg::bright_magenta]
            ,   it->id()
            ,   gnx::ansi::escape[fg::reset]
            );
            if (it->description().size() > 0)
                fmt::format_to
                (   std::back_inserter(buf)
                ,   "│   ├── {}{}{}\n"
                ,   gnx::ansi::escape[fg::bright_cyan]
                +   gnx::ansi::escape[style::italic]
                ,   it->description()
                ,   gnx::ansi::escape[style::reset]
                );
            fmt::format_to
            (   std::back_inserter(buf)
            ,   "│   └── {}{}{}\n"
            ,   gnx::ansi::escape[fg::bright_yellow]
            ,   hrs(it->sequence().size(), is_peptide)
            ,   gnx::ansi::escape[fg::reset]
            );
            total_length += it->sequence().size();
        }
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "└── {}{}{} {}, {}{}{}\n"
        ,   gnx::ansi::escape[fg::bright_yellow]
        ,   count
        ,   gnx::ansi::escape[fg::reset]
        ,   (count > 1) ? "sequences" : "sequence"
        ,   gnx::ansi::escape[fg::bright_yellow]
        ,   hrs(total_length, is_peptide)
        ,   gnx::ansi::escape[fg::reset]
        );
        std::cout.write(buf.data(), buf.size());
    }
}

}   // namespace gnx

