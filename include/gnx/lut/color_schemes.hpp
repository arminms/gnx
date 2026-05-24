// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
// Implementation of the ANSI color codes for terminal output
//
#pragma once

#include <gnx/lut/ansi.hpp>

namespace gnx::color_scheme {

std::array<std::string, 256> create_na_cs_256() noexcept
{   std::array<std::string, 256> table{};
    for (size_t i = 0; i < 256; ++i)
    {   table[i]
        =   fmt::format
        (   "{}{}{}{}"
        ,   gnx::ansi::vga::ESC_BG[196]
        ,   gnx::ansi::vga::ESC_FG[16]
        ,   static_cast<char>(i)
        ,   gnx::ansi::ESC[style::reset]
        );
    }

    table['A'] = fmt::format("{}A", gnx::ansi::vga::ESC_FG[40]);
    table['C'] = fmt::format("{}C", gnx::ansi::vga::ESC_FG[33]);
    table['G'] = "G";
    table['T'] = fmt::format("{}T", gnx::ansi::vga::ESC_FG[160]);
    table['U'] = fmt::format("{}U", gnx::ansi::vga::ESC_FG[160]);

    table['N']
    =   fmt::format
    (   "{}{}N{}"
    ,   gnx::ansi::vga::ESC_BG[226]
    ,   gnx::ansi::vga::ESC_FG[16]
    ,   gnx::ansi::ESC[style::reset]
    );

    table['a'] = fmt::format("{}a", gnx::ansi::vga::ESC_FG[40]);
    table['c'] = fmt::format("{}c", gnx::ansi::vga::ESC_FG[33]);
    table['g'] = "g";
    table['t'] = fmt::format("{}t", gnx::ansi::vga::ESC_FG[160]);
    table['u'] = fmt::format("{}u", gnx::ansi::vga::ESC_FG[160]);

    table['n']
    =   fmt::format
    (   "{}{}n{}"
    ,   gnx::ansi::vga::ESC_BG[226]
    ,   gnx::ansi::vga::ESC_FG[16]
    ,   gnx::ansi::ESC[style::reset]
    );

    return table;
}

thread_local static const auto na = create_na_cs_256();

std::array<std::string, 256> create_na_warn_cs() noexcept
{   std::array<std::string, 256> table{};
    for (size_t i = 0; i < 256; ++i)
    {   table[i]
        =   fmt::format
        (   "{}{}{}{}"
        ,   gnx::ansi::vga::ESC_BG[196]
        ,   gnx::ansi::vga::ESC_FG[16]
        ,   static_cast<char>(i)
        ,   gnx::ansi::ESC[style::reset]
        );
    }

    table['N']
    =   fmt::format
    (   "{}{}N{}"
    ,   gnx::ansi::vga::ESC_BG[226]
    ,   gnx::ansi::vga::ESC_FG[16]
    ,   gnx::ansi::ESC[style::reset]
    );
    table['n']
    =   fmt::format
    (   "{}{}n{}"
    ,   gnx::ansi::vga::ESC_BG[226]
    ,   gnx::ansi::vga::ESC_FG[16]
    ,   gnx::ansi::ESC[style::reset]
    );

    table['A'] = "A";
    table['C'] = "C";
    table['G'] = "G";
    table['T'] = "T";
    table['U'] = "U";

    table['a'] = "a";
    table['c'] = "c";
    table['g'] = "g";
    table['t'] = "t";
    table['u'] = "u";

    return table;
}

thread_local static const auto na_warn = create_na_warn_cs();

std::array<std::string, 256> create_aa_clustal_cs() noexcept
{   std::array<std::string, 256> table{};
    for (size_t i = 0; i < 256; ++i)
    {   table[i]
        =   fmt::format
        (   "{}{}{}{}"
        ,   gnx::ansi::vga::ESC_BG[160]
        ,   gnx::ansi::vga::ESC_FG[226]
        ,   static_cast<char>(i)
        ,   gnx::ansi::ESC[style::reset]
        );
    }

    auto assign = [&table](std::string_view chars, std::string_view color)
    {   for (char c : chars)
        {   table[static_cast<uint8_t>(std::toupper(c))]
            =   fmt::format("{}{}", color, c);
            table[static_cast<uint8_t>(std::tolower(c))]
            =   fmt::format("{}{}", color, c);
        }
    };

    assign("AILMV", gnx::ansi::vga::ESC_FG[33]); // hydrophobic
    assign("KR", gnx::ansi::vga::ESC_FG[196]);   // basic
    assign("DE", gnx::ansi::vga::ESC_FG[201]);   // acidic
    assign("NQST", gnx::ansi::vga::ESC_FG[46]);  // polar
    assign("C", gnx::ansi::vga::ESC_FG[213]);    // cysteine
    assign("G", gnx::ansi::vga::ESC_FG[214]);    // glycine
    assign("P", gnx::ansi::vga::ESC_FG[226]);    // proline
    assign("FHWY", gnx::ansi::vga::ESC_FG[51]);  // aromatic

    return table;
}

thread_local static const auto aa_clustal = create_aa_clustal_cs();

} // namespace gnx::color_scheme