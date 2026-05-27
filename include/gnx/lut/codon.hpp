// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <array>
#include <cstdint>

#include <gnx/lut/lut_commons.hpp>

namespace gnx::lut {

/// @brief Compile-time base encoding for codon index computation.
///
/// Maps each nucleotide character to a 2-bit index using the NCBI standard
/// codon table ordering: T/U=0, C=1, A=2, G=3.
/// All other characters map to 0xFF (invalid marker).
constexpr std::array<uint8_t, 256> create_base_enc_table()
{   std::array<uint8_t, 256> table{};
    for (std::size_t i = 0; i < 256; ++i)
        table[i] = 0xFF; // invalid

    // Uppercase
    table['T'] = 0; table['U'] = 0; // T/U → 0
    table['C'] = 1;                  // C   → 1
    table['A'] = 2;                  // A   → 2
    table['G'] = 3;                  // G   → 3

    // Lowercase
    table['t'] = 0; table['u'] = 0;
    table['c'] = 1;
    table['a'] = 2;
    table['g'] = 3;

    return table;
}

/// @brief Base encoding lookup table (char → 2-bit NCBI index; 0xFF = invalid).
constexpr auto base_enc = create_base_enc_table();

/// @brief NCBI standard genetic code (codon table 1).
///
/// Indexed as `codon_table[b1*16 + b2*4 + b3]` where each base is encoded
/// via `base_enc`. The ordering T=0, C=1, A=2, G=3 matches the NCBI standard.
/// b3 is the fastest-cycling index. Stop codons are represented as '*'.
///
/// Equivalent to the NCBI translation string:
///   "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
///
/// Verification (selected entries):
///   index  0 (TTT) → F     index 35 (ATG) → M (start)
///   index 10 (TAA) → *     index 63 (GGG) → G
constexpr std::array<char, 64> codon_table =
{{  //      b3=T  b3=C  b3=A  b3=G    (b1,b2) codon prefix
    /* TT */ 'F',  'F',  'L',  'L',  //  0- 3
    /* TC */ 'S',  'S',  'S',  'S',  //  4- 7
    /* TA */ 'Y',  'Y',  '*',  '*',  //  8-11
    /* TG */ 'C',  'C',  '*',  'W',  // 12-15
    /* CT */ 'L',  'L',  'L',  'L',  // 16-19
    /* CC */ 'P',  'P',  'P',  'P',  // 20-23
    /* CA */ 'H',  'H',  'Q',  'Q',  // 24-27
    /* CG */ 'R',  'R',  'R',  'R',  // 28-31
    /* AT */ 'I',  'I',  'I',  'M',  // 32-35
    /* AC */ 'T',  'T',  'T',  'T',  // 36-39
    /* AA */ 'N',  'N',  'K',  'K',  // 40-43
    /* AG */ 'S',  'S',  'R',  'R',  // 44-47
    /* GT */ 'V',  'V',  'V',  'V',  // 48-51
    /* GC */ 'A',  'A',  'A',  'A',  // 52-55
    /* GA */ 'D',  'D',  'E',  'E',  // 56-59
    /* GG */ 'G',  'G',  'G',  'G',  // 60-63
}};

} // end gnx::lut namespace
