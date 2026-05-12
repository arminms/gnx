// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <filesystem>

#include <fmt/core.h>
#include <fmt/format.h>

#include <gnx/sq.hpp>
#include <gnx/lut/ansi.hpp>
#include <gnx/backend/forward_stream.hpp>

namespace gnx {

inline void describe(std::string_view filename)
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

    // bool is_fastq = !stream.quality().empty();

    std::size_t count{0}, total_length{0};
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
        ,   "│   └── {}{}{} bp\n"
        ,   gnx::ansi::escape[fg::bright_yellow]
        ,   it->sequence().size()
        ,   gnx::ansi::escape[fg::reset]
        );
        total_length += it->sequence().size();
    }
    fmt::format_to
    (   std::back_inserter(buf)
    ,   "└── {}{}{} {}, {}{}{} bp\n"
    ,   gnx::ansi::escape[fg::bright_yellow]
    ,   count
    ,   gnx::ansi::escape[fg::reset]
    ,   (count > 1) ? "sequences" : "sequence"
    ,   gnx::ansi::escape[fg::bright_yellow]
    ,   total_length
    ,   gnx::ansi::escape[fg::reset]
    );
    std::cout.write(buf.data(), buf.size());
}

}   // namespace gnx