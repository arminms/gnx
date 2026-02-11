// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <array>
#include <cstdint>

namespace gnx::lut {

/// @brief Compile-time generated lookup table for case-insensitive character normalization.
/// Maps both uppercase and lowercase characters to their uppercase equivalents.
/// This enables case-insensitive counting and comparison operations.
constexpr std::array<char, 256> create_case_fold_table()
{   std::array<char, 256> table{};

    // Initialize with identity mapping
    for (int i = 0; i < 256; ++i)
        table[i] = static_cast<char>(i);

    // Map lowercase ASCII letters to uppercase
    for (char c = 'a'; c <= 'z'; ++c)
        table[static_cast<uint8_t>(c)] = c - 32;

    return table;
}

/// @brief Instantiate the case-folding table in static memory.
/// Example: char normalized = gnx::lut::case_fold[static_cast<uint8_t>(ch)];
inline constexpr auto case_fold = create_case_fold_table();

} // namespace gnx::lut
