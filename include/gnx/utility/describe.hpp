// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <iterator>
#include <ranges>
#include <type_traits>

#include <gnx/utility/detail/describe.hpp>

namespace gnx {

template<std::ranges::input_range Range>
inline void describe(const Range& range)
{   if constexpr (requires { typename Range::map_type; })
    {   fmt::memory_buffer buf;
        if (range.has("_id"))
            fmt::format_to(std::back_inserter(buf)
            ,   "{}▼ {}{}{}\n"
            ,   ansi::vga::fg::ESC[250]
            ,   ansi::ESC[fg::bright_magenta]
            ,   std::any_cast<std::string>(range["_id"])
            ,   ansi::ESC[fg::reset]
            );
        else
            fmt::format_to(std::back_inserter(buf)
            ,   "{}▼ {}{}{}\n"
            ,   ansi::vga::fg::ESC[250]
            ,   ansi::vga::fg::ESC[10]
            ,   is_peptide(std::begin(range), std::end(range))
                ?   "Protein sequence"
                :   "DNA/RNA sequence"
            ,   ansi::ESC[fg::reset]
            );
        if (range.has("_desc"))
        fmt::format_to(std::back_inserter(buf)
        ,   "{}├── {}{}{}\n"
        ,   ansi::vga::fg::ESC[250]
        ,   ansi::ESC[fg::bright_cyan]
        ,   std::any_cast<std::string>(range["_desc"])
        ,   ansi::ESC[fg::reset]
        );
        std::cout.write(buf.data(), buf.size());
        detail::describe(std::begin(range), std::end(range));
    }
    else if constexpr (std::is_convertible_v<Range, std::filesystem::path>)
    {   // if the range is a view over a file, attempt to describe it as such
        if (std::filesystem::exists(std::string(range) + ".fai"))
            detail::describe_vv(range);
        else
            detail::describe_fs(range);
    }
    else
    {   fmt::memory_buffer buf;
        fmt::format_to(std::back_inserter(buf)
        ,   "{}▼ {}View over a {} sequence{}\n"
        ,   ansi::vga::fg::ESC[250]
        ,   ansi::vga::fg::ESC[10]
        ,   is_peptide(std::begin(range), std::end(range))
            ?   "protein"
            :   "DNA/RNA"
        ,   ansi::ESC[fg::reset]
        );
        std::cout.write(buf.data(), buf.size());
        detail::describe(std::begin(range), std::end(range));
    }
}

}   // namespace gnx

