// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
// Implementation of the ANSI color codes for terminal output
//
#pragma once

#include <gnx/lut/ansi.hpp>

namespace gnx::color_scheme {

std::array<std::string, 256> create_na_cs() noexcept
{   std::array<std::string, 256> table{};
    for (size_t i = 0; i < 256; ++i)
    {   table[i]
        =   gnx::ansi::vga::ESC_BG[196] + gnx::ansi::vga::ESC_FG[16]
        +   static_cast<char>(i)
        +   gnx::ansi::ESC[style::reset]
        ;
    }

    table['A'] = gnx::ansi::vga::ESC_FG[40] + "A" + gnx::ansi::ESC[fg::reset];
    table['C'] = gnx::ansi::vga::ESC_FG[33] + "C" + gnx::ansi::ESC[fg::reset];
    table['G'] = "G";
    table['T'] = gnx::ansi::vga::ESC_FG[160] + "T" + gnx::ansi::ESC[fg::reset];
    table['U'] = gnx::ansi::vga::ESC_FG[160] + "U" + gnx::ansi::ESC[fg::reset];

    table['N']
    =   gnx::ansi::vga::ESC_BG[226] + gnx::ansi::vga::ESC_FG[16]
    +   "N"
    +   gnx::ansi::ESC[style::reset];

    table['a'] = gnx::ansi::vga::ESC_FG[40] + "a" + gnx::ansi::ESC[fg::reset];
    table['c'] = gnx::ansi::vga::ESC_FG[33] + "c" + gnx::ansi::ESC[fg::reset];
    table['g'] = "g";
    table['t'] = gnx::ansi::vga::ESC_FG[160] + "t" + gnx::ansi::ESC[fg::reset];
    table['u'] = gnx::ansi::vga::ESC_FG[160] + "u" + gnx::ansi::ESC[fg::reset];

    table['n']
    =   gnx::ansi::vga::ESC_BG[226] + gnx::ansi::vga::ESC_FG[16]
    +   "n"
    +   gnx::ansi::ESC[style::reset];

    return table;
}

thread_local static const auto na = create_na_cs();

std::array<std::string, 256> create_na_warn_cs() noexcept
{   std::array<std::string, 256> table{};
    for (size_t i = 0; i < 256; ++i)
    {   table[i]
        =   gnx::ansi::vga::ESC_BG[196] + gnx::ansi::vga::ESC_FG[16]
        +   static_cast<char>(i)
        +   gnx::ansi::ESC[style::reset]
        ;
    }

    table['N']
    =   gnx::ansi::vga::ESC_BG[226] + gnx::ansi::vga::ESC_FG[16]
    +   "N"
    +   gnx::ansi::ESC[style::reset];
    table['n']
    =   gnx::ansi::vga::ESC_BG[226] + gnx::ansi::vga::ESC_FG[16]
    +   "n"
    +   gnx::ansi::ESC[style::reset];

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
        =   gnx::ansi::vga::ESC_BG[160] + gnx::ansi::vga::ESC_FG[226]
        +   static_cast<char>(i)
        +   gnx::ansi::ESC[bg::reset]
        ;
    }

    // hydrophobic
    table['A']
    =   gnx::ansi::vga::ESC_FG[33] //+ gnx::ansi::vga::ESC_FG[16]
    +   "A"
    +   gnx::ansi::ESC[style::reset]
    ;
    table['I']
    =   gnx::ansi::vga::ESC_FG[33] //+ gnx::ansi::vga::ESC_FG[16]
    +   "I"
    +   gnx::ansi::ESC[style::reset]
    ;
    table['L']
    =   gnx::ansi::vga::ESC_FG[33] //+ gnx::ansi::vga::ESC_FG[16]
    +   "L"
    +   gnx::ansi::ESC[style::reset]
    ;
    table['M']
    =   gnx::ansi::vga::ESC_FG[33] //+ gnx::ansi::vga::ESC_FG[16]
    +   "M"
    +   gnx::ansi::ESC[style::reset]
    ;
    table['V']
    =   gnx::ansi::vga::ESC_FG[33] //+ gnx::ansi::vga::ESC_FG[16]
    +   "V"
    +   gnx::ansi::ESC[style::reset]
    ;

    // Basic (Positive)
    table['K']
    =   gnx::ansi::vga::ESC_FG[196] //+ gnx::ansi::vga::ESC_FG[16]
    +   "K"
    +   gnx::ansi::ESC[style::reset]
    ;
    table['R']
    =   gnx::ansi::vga::ESC_FG[196] //+ gnx::ansi::vga::ESC_FG[16]
    +   "R"
    +   gnx::ansi::ESC[style::reset]
    ;

    // Acidic (Negative)
    table['D']
    =   gnx::ansi::vga::ESC_FG[201] //+ gnx::ansi::vga::ESC_FG[16]
    +   "D"
    +   gnx::ansi::ESC[style::reset]
    ;
    table['E']
    =   gnx::ansi::vga::ESC_FG[201] //+ gnx::ansi::vga::ESC_FG[16]
    +   "E"
    +   gnx::ansi::ESC[style::reset]
    ;

    // Polar
    table['N']
    =   gnx::ansi::vga::ESC_FG[46] //+ gnx::ansi::vga::ESC_FG[16]
    +   "N"
    +   gnx::ansi::ESC[style::reset]
    ;
    table['Q']
    =   gnx::ansi::vga::ESC_FG[46] //+ gnx::ansi::vga::ESC_FG[16]
    +   "Q"
    +   gnx::ansi::ESC[style::reset]
    ;
    table['S']
    =   gnx::ansi::vga::ESC_FG[46] //+ gnx::ansi::vga::ESC_FG[16]
    +   "S"
    +   gnx::ansi::ESC[style::reset]
    ;
    table['T']
    =   gnx::ansi::vga::ESC_FG[46] //+ gnx::ansi::vga::ESC_FG[16]
    +   "T"
    +   gnx::ansi::ESC[style::reset]
    ;

    // Cysteine
    table['C']
    =   gnx::ansi::vga::ESC_FG[213] //+ gnx::ansi::vga::ESC_FG[16]
    +   "C"
    +   gnx::ansi::ESC[style::reset]
    ;

    // Glycine
    table['G']
    =   gnx::ansi::vga::ESC_FG[214] //+ gnx::ansi::vga::ESC_FG[16]
    +   "G"
    +   gnx::ansi::ESC[style::reset]
    ;

    // Proline
    table['P']
    =   gnx::ansi::vga::ESC_FG[226] //+ gnx::ansi::vga::ESC_FG[16]
    +   "P"
    +   gnx::ansi::ESC[style::reset]
    ;

    // Aromatic
    table['F']
    =   gnx::ansi::vga::ESC_FG[51] //+ gnx::ansi::vga::ESC_FG[16]
    +   "F"
    +   gnx::ansi::ESC[style::reset]
    ;
    table['H']
    =   gnx::ansi::vga::ESC_FG[51] //+ gnx::ansi::vga::ESC_FG[16]
    +   "H"
    +   gnx::ansi::ESC[style::reset]
    ;
    table['W']
    =   gnx::ansi::vga::ESC_FG[51] //+ gnx::ansi::vga::ESC_FG[16]
    +   "W"
    +   gnx::ansi::ESC[style::reset]
    ;
    table['Y']
    =   gnx::ansi::vga::ESC_FG[51] //+ gnx::ansi::vga::ESC_FG[16]
    +   "Y"
    +   gnx::ansi::ESC[style::reset]
    ;

    return table;
}

thread_local static const auto aa_clustal = create_aa_clustal_cs();

} // namespace gnx::color_scheme