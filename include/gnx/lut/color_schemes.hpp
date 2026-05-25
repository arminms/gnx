// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
// Implementation of the ANSI color codes for terminal output
//
#pragma once

#include <gnx/lut/ansi.hpp>

namespace gnx::color_scheme {

namespace detail {

void assign_color
(   std::array<std::string, 256> &table
,   std::string_view chars
,   std::string_view color
)
{   for (char c : chars)
    {   table[static_cast<uint8_t>(std::toupper(c))]
        =   fmt::format("{}{}", color, (char)std::toupper(c));
        table[static_cast<uint8_t>(std::tolower(c))]
        =   fmt::format("{}{}", color, (char)std::tolower(c));
    }
}

void assign_color
(   std::array<std::string, 256> &table
,   std::string_view chars
,   std::string_view fg_color
,   std::string_view bg_color
)
{   for (char c : chars)
    {   table[static_cast<uint8_t>(std::toupper(c))]
        =   fmt::format("{}{}{}", fg_color, bg_color, (char)std::toupper(c));
        table[static_cast<uint8_t>(std::tolower(c))]
        =   fmt::format("{}{}{}", fg_color, bg_color, (char)std::tolower(c));
    }
}

void assign_color_reset
(   std::array<std::string, 256> &table
,   std::string_view chars
,   std::string_view fg_color
,   std::string_view bg_color
)
{   for (char c : chars)
    {   table[static_cast<uint8_t>(std::toupper(c))]
        =   fmt::format
            (   "{}{}{}{}"
            ,   fg_color
            ,   bg_color
            ,   (char)std::toupper(c)
            ,   gnx::ansi::ESC[style::reset]
            );
        table[static_cast<uint8_t>(std::tolower(c))]
        =   fmt::format
            (   "{}{}{}{}"
            ,   fg_color
            ,   bg_color
            ,   (char)std::tolower(c)
            ,   gnx::ansi::ESC[style::reset]
            );
    }
}

} // namespace detail

std::array<std::string, 256> create_mono_cs() noexcept
{   std::array<std::string, 256> table{};
    for (size_t i = 0; i < 256; ++i)
        table[i] = static_cast<char>(i);
    return table;
}

thread_local static const auto mono = create_mono_cs();

std::array<std::string, 256> create_na_cs() noexcept
{   std::array<std::string, 256> table{};
    for (size_t i = 0; i < 256; ++i)
    {   table[i]
        =   fmt::format
        (   "{}{}{}{}"
        ,   gnx::ansi::vga::ESC_FG[226]
        ,   gnx::ansi::vga::ESC_BG[160]
        ,   static_cast<char>(i)
        ,   gnx::ansi::ESC[style::reset]
        );
    }

    detail::assign_color(table, "A", gnx::ansi::vga::ESC_FG[40]);
    detail::assign_color(table, "C", gnx::ansi::vga::ESC_FG[33]);
    detail::assign_color(table, "G", gnx::ansi::vga::ESC_FG[240]);
    detail::assign_color(table, "T", gnx::ansi::vga::ESC_FG[160]);
    detail::assign_color(table, "U", gnx::ansi::vga::ESC_FG[160]);
    detail::assign_color_reset
    (   table
    ,   "N"
    ,   gnx::ansi::vga::ESC_FG[16]
    ,   gnx::ansi::vga::ESC_BG[226]
    );
    detail::assign_color_reset
    (   table
    ,   "RYWSKMBDHV"
    ,   gnx::ansi::vga::ESC_FG[16]
    ,   gnx::ansi::vga::ESC_BG[207]
    );

    return table;
}

thread_local static const auto na = create_na_cs();

std::array<std::string, 256> create_na_inverted_cs() noexcept
{   std::array<std::string, 256> table{};
    for (size_t i = 0; i < 256; ++i)
    {   table[i]
        =   fmt::format
        (   "{}{}{}"
        ,   gnx::ansi::vga::ESC_FG[226]
        ,   gnx::ansi::vga::ESC_BG[160]
        ,   static_cast<char>(i)
        );
    }

    detail::assign_color(table, "A", gnx::ansi::vga::ESC_FG[16], gnx::ansi::vga::ESC_BG[40]);
    detail::assign_color(table, "C", gnx::ansi::vga::ESC_FG[16], gnx::ansi::vga::ESC_BG[33]);
    detail::assign_color(table, "G", gnx::ansi::vga::ESC_FG[15], gnx::ansi::vga::ESC_BG[240]);
    detail::assign_color(table, "T", gnx::ansi::vga::ESC_FG[16], gnx::ansi::vga::ESC_BG[160]);
    detail::assign_color(table, "U", gnx::ansi::vga::ESC_FG[16], gnx::ansi::vga::ESC_BG[160]);
    detail::assign_color(table, "N", gnx::ansi::vga::ESC_FG[16], gnx::ansi::vga::ESC_BG[226]);
    detail::assign_color
    (   table
    ,   "RYWSKMBDHV"
    ,   gnx::ansi::vga::ESC_FG[16]
    ,   gnx::ansi::vga::ESC_BG[207]
    );

    return table;
}

thread_local static const auto na_inverted = create_na_inverted_cs();

std::array<std::string, 256> create_na_warn_cs() noexcept
{   std::array<std::string, 256> table{};
    for (size_t i = 0; i < 256; ++i)
    {   table[i]
        =   fmt::format
        (   "{}{}{}{}"
        ,   gnx::ansi::vga::ESC_FG[226]
        ,   gnx::ansi::vga::ESC_BG[160]
        ,   static_cast<char>(i)
        ,   gnx::ansi::ESC[style::reset]
        );
    }

    detail::assign_color(table, "ACGTU", "");
    detail::assign_color_reset
    (   table
    ,   "N"
    ,   gnx::ansi::vga::ESC_FG[16]
    ,   gnx::ansi::vga::ESC_BG[226]
    );
    detail::assign_color_reset
    (   table
    ,   "RYWSKMBDHV"
    ,   gnx::ansi::vga::ESC_FG[16]
    ,   gnx::ansi::vga::ESC_BG[207]
    );

    return table;
}

thread_local static const auto na_warn = create_na_warn_cs();

std::array<std::string, 256> create_aa_warn_cs() noexcept
{   std::array<std::string, 256> table{};
    for (size_t i = 0; i < 256; ++i)
    {   table[i]
        =   fmt::format
        (   "{}{}{}{}"
        ,   gnx::ansi::vga::ESC_FG[226]
        ,   gnx::ansi::vga::ESC_BG[160]
        ,   static_cast<char>(i)
        ,   gnx::ansi::ESC[style::reset]
        );
    }

    detail::assign_color(table, "ACDEFGHIKLMNPQRSTVWY", "");
    detail::assign_color_reset
    (   table
    ,   "*"
    ,   gnx::ansi::vga::ESC_FG[16]
    ,   gnx::ansi::vga::ESC_BG[226]
    );
    detail::assign_color_reset
    (   table
    ,   "BZXUOJ"
    ,   gnx::ansi::vga::ESC_FG[16]
    ,   gnx::ansi::vga::ESC_BG[207]
    );

    return table;
}

thread_local static const auto aa_warn = create_aa_warn_cs();

std::array<std::string, 256> create_aa_clustal_cs() noexcept
{   std::array<std::string, 256> table{};
    for (size_t i = 0; i < 256; ++i)
    {   table[i]
        =   fmt::format
        (   "{}{}{}{}"
        ,   gnx::ansi::vga::ESC_FG[226]
        ,   gnx::ansi::vga::ESC_BG[160]
        ,   static_cast<char>(i)
        ,   gnx::ansi::ESC[style::reset]
        );
    }

    detail::assign_color(table, "AILMV", gnx::ansi::vga::ESC_FG[33]); // hydrophobic (blue)
    detail::assign_color(table, "KR", gnx::ansi::vga::ESC_FG[160]);   // basic (red)
    detail::assign_color(table, "DE", gnx::ansi::vga::ESC_FG[201]);   // acidic (magenta)
    detail::assign_color(table, "NQST", gnx::ansi::vga::ESC_FG[40]);  // polar (green)
    detail::assign_color(table, "C", gnx::ansi::vga::ESC_FG[213]);    // cysteine (pink)
    detail::assign_color(table, "G", gnx::ansi::vga::ESC_FG[208]);    // glycine (orange)
    detail::assign_color(table, "P", gnx::ansi::vga::ESC_FG[220]);    // proline (yellow)
    detail::assign_color(table, "FHWY", gnx::ansi::vga::ESC_FG[45]);  // aromatic (cyan)

    return table;
}

thread_local static const auto aa_clustal = create_aa_clustal_cs();

std::array<std::string, 256> create_aa_clustal_inverted_cs() noexcept
{   std::array<std::string, 256> table{};
    for (size_t i = 0; i < 256; ++i)
    {   table[i]
        =   fmt::format
        (   "{}{}{}"
        ,   gnx::ansi::vga::ESC_FG[226]
        ,   gnx::ansi::vga::ESC_BG[160]
        ,   static_cast<char>(i)
        );
    }

    detail::assign_color(table, "AILMV", gnx::ansi::vga::ESC_FG[16], gnx::ansi::vga::ESC_BG[33]); // hydrophobic (blue)
    detail::assign_color(table, "KR", gnx::ansi::vga::ESC_FG[16], gnx::ansi::vga::ESC_BG[160]);   // basic (red)
    detail::assign_color(table, "DE", gnx::ansi::vga::ESC_FG[16], gnx::ansi::vga::ESC_BG[201]);   // acidic (magenta)
    detail::assign_color(table, "NQST", gnx::ansi::vga::ESC_FG[16], gnx::ansi::vga::ESC_BG[40]);  // polar (green)
    detail::assign_color(table, "C", gnx::ansi::vga::ESC_FG[16], gnx::ansi::vga::ESC_BG[213]);    // cysteine (pink)
    detail::assign_color(table, "G", gnx::ansi::vga::ESC_FG[16], gnx::ansi::vga::ESC_BG[208]);    // glycine (orange)
    detail::assign_color(table, "P", gnx::ansi::vga::ESC_FG[16], gnx::ansi::vga::ESC_BG[220]);    // proline (yellow)
    detail::assign_color(table, "FHWY", gnx::ansi::vga::ESC_FG[16], gnx::ansi::vga::ESC_BG[45]);  // aromatic (cyan)

    return table;
}

thread_local static const auto aa_clustal_inverted = create_aa_clustal_inverted_cs();

} // namespace gnx::color_scheme