// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
#pragma once

#include <array>
#include <cstdint>

#include <gnx/lut/lut_commons.hpp>

namespace gnx::lut {

constexpr std::array<bool, 256> create_peptide_table()
{   std::array<bool, 256> table{};
    table.fill(false);

    // The "E, F, I, P, Q, Z" Rule
    table['E'] = true;
    table['F'] = true;
    table['I'] = true;
    table['P'] = true;
    table['Q'] = true;
    table['Z'] = true;

    // lowercase versions
    table['e'] = true;
    table['f'] = true;
    table['i'] = true;
    table['p'] = true;
    table['q'] = true;
    table['z'] = true;

    return table;
}

/// @brief Compile-time generated lookup table for valid peptide characters.
/// Always cast your input char to uint8_t when indexing into this table
/// to avoid negative indices due to sign extension.
/// Example: bool peptide = gnx::lut::is_peptide[static_cast<uint8_t>(ch)];
inline constexpr auto is_peptide = create_peptide_table();

} // namespace gnx::lut