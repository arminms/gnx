//
// Copyright (c) 2025 Armin Sobhani
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
#ifndef _GYNX_LUT_VALID_HPP_
#define _GYNX_LUT_VALID_HPP_

#include <array>
#include <cstdint>

namespace gynx::lut {

/// @brief Compile-time generated lookup table for valid nucleotide characters.
/// Valid characters include: A, C, G, T, U, N (both uppercase and lowercase)
constexpr std::array<bool, 256> create_valid_nucleotide_table()
{   std::array<bool, 256> table{};
    table.fill(false);
    
    // Valid nucleotide characters (uppercase)
    table['A'] = true;
    table['C'] = true;
    table['G'] = true;
    table['T'] = true;
    table['U'] = true;
    table['N'] = true; // ambiguous/unknown base
    
    // Valid nucleotide characters (lowercase)
    table['a'] = true;
    table['c'] = true;
    table['g'] = true;
    table['t'] = true;
    table['u'] = true;
    table['n'] = true;
    
    // IUPAC ambiguity codes (uppercase)
    table['R'] = true; // A or G (puRine)
    table['Y'] = true; // C or T (pYrimidine)
    table['S'] = true; // G or C (Strong)
    table['W'] = true; // A or T (Weak)
    table['K'] = true; // G or T (Keto)
    table['M'] = true; // A or C (aMino)
    table['B'] = true; // C, G or T (not A)
    table['D'] = true; // A, G or T (not C)
    table['H'] = true; // A, C or T (not G)
    table['V'] = true; // A, C or G (not T)
    
    // IUPAC ambiguity codes (lowercase)
    table['r'] = true;
    table['y'] = true;
    table['s'] = true;
    table['w'] = true;
    table['k'] = true;
    table['m'] = true;
    table['b'] = true;
    table['d'] = true;
    table['h'] = true;
    table['v'] = true;
    
    return table;
}

/// @brief Compile-time generated lookup table for valid peptide (amino acid) characters.
/// Valid characters include: 20 standard amino acids + ambiguous codes (both uppercase and lowercase)
constexpr std::array<bool, 256> create_valid_peptide_table()
{   std::array<bool, 256> table{};
    table.fill(false);
    
    // 20 standard amino acids (uppercase)
    table['A'] = true; // Alanine
    table['C'] = true; // Cysteine
    table['D'] = true; // Aspartic acid
    table['E'] = true; // Glutamic acid
    table['F'] = true; // Phenylalanine
    table['G'] = true; // Glycine
    table['H'] = true; // Histidine
    table['I'] = true; // Isoleucine
    table['K'] = true; // Lysine
    table['L'] = true; // Leucine
    table['M'] = true; // Methionine
    table['N'] = true; // Asparagine
    table['P'] = true; // Proline
    table['Q'] = true; // Glutamine
    table['R'] = true; // Arginine
    table['S'] = true; // Serine
    table['T'] = true; // Threonine
    table['V'] = true; // Valine
    table['W'] = true; // Tryptophan
    table['Y'] = true; // Tyrosine
    
    // 20 standard amino acids (lowercase)
    table['a'] = true;
    table['c'] = true;
    table['d'] = true;
    table['e'] = true;
    table['f'] = true;
    table['g'] = true;
    table['h'] = true;
    table['i'] = true;
    table['k'] = true;
    table['l'] = true;
    table['m'] = true;
    table['n'] = true;
    table['p'] = true;
    table['q'] = true;
    table['r'] = true;
    table['s'] = true;
    table['t'] = true;
    table['v'] = true;
    table['w'] = true;
    table['y'] = true;
    
    // Ambiguous/special codes (uppercase)
    table['B'] = true; // Aspartic acid or Asparagine
    table['Z'] = true; // Glutamic acid or Glutamine
    table['X'] = true; // Unknown or any amino acid
    table['*'] = true; // Stop codon
    table['U'] = true; // Selenocysteine
    table['O'] = true; // Pyrrolysine
    table['J'] = true; // Leucine or Isoleucine
    
    // Ambiguous/special codes (lowercase)
    table['b'] = true;
    table['z'] = true;
    table['x'] = true;
    table['u'] = true;
    table['o'] = true;
    table['j'] = true;
    
    return table;
}

/// @brief Instantiate the nucleotide validation table in static memory.
/// Example: bool is_valid = gynx::lut::valid_nucleotide[static_cast<uint8_t>(ch)];
inline constexpr auto valid_nucleotide = create_valid_nucleotide_table();

/// @brief Instantiate the peptide validation table in static memory.
/// Example: bool is_valid = gynx::lut::valid_peptide[static_cast<uint8_t>(ch)];
inline constexpr auto valid_peptide = create_valid_peptide_table();

} // namespace gynx::lut

#endif  // _GYNX_LUT_VALID_HPP_
