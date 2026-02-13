// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <array>
#include <cstdint>

#include <gnx/lut/lut_commons.hpp>

namespace gnx::lut {

/// @brief Compile-time generated lookup table for nucleotide complements.
/// Maps each nucleotide to its Watson-Crick complement.
/// Ambiguous bases are mapped to their appropriate complements based on IUPAC codes.
constexpr std::array<char, 256> create_complement_table()
{   std::array<char, 256> table{};
    // Initialize all characters to themselves (no complement)
    for (std::size_t i = 0; i < 256; ++i)
        table[i] = static_cast<char>(i);

    // Watson-Crick base pairs (uppercase)
    table['A'] = 'T';
    table['T'] = 'A';
    table['G'] = 'C';
    table['C'] = 'G';
    table['U'] = 'A'; // RNA: U complements to A

    // Watson-Crick base pairs (lowercase)
    table['a'] = 't';
    table['t'] = 'a';
    table['g'] = 'c';
    table['c'] = 'g';
    table['u'] = 'a';

    // IUPAC ambiguity codes (uppercase)
    table['R'] = 'Y'; // A or G -> C or T (puRine -> pYrimidine)
    table['Y'] = 'R'; // C or T -> A or G (pYrimidine -> puRine)
    table['S'] = 'S'; // G or C -> C or G (Strong, self-complementary)
    table['W'] = 'W'; // A or T -> T or A (Weak, self-complementary)
    table['K'] = 'M'; // G or T -> C or A (Keto -> aMino)
    table['M'] = 'K'; // A or C -> T or G (aMino -> Keto)
    table['B'] = 'V'; // C, G or T -> G, C or A (not A -> not T)
    table['D'] = 'H'; // A, G or T -> T, C or A (not C -> not G)
    table['H'] = 'D'; // A, C or T -> T, G or A (not G -> not C)
    table['V'] = 'B'; // A, C or G -> T, G or C (not T -> not A)
    table['N'] = 'N'; // Any base -> Any base (unknown, self-complementary)

    // IUPAC ambiguity codes (lowercase)
    table['r'] = 'y';
    table['y'] = 'r';
    table['s'] = 's';
    table['w'] = 'w';
    table['k'] = 'm';
    table['m'] = 'k';
    table['b'] = 'v';
    table['d'] = 'h';
    table['h'] = 'd';
    table['v'] = 'b';
    table['n'] = 'n';

    return table;
}

/// @brief Lookup table instance for nucleotide complements
constexpr auto complement = create_complement_table();

} // end gnx::lut namespace

