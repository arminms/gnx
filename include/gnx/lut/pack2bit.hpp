// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
#pragma once

#include <array>
#include <stdint.h>

namespace gnx::lut {

// Generate the Lookup Table at compile time
constexpr std::array<uint8_t, 256> create_2bit_encode_table()
{   std::array<uint8_t, 256> table{};

    table.fill(0b00u);
    table[static_cast<uint8_t>('C')] = 0b01u;
    table[static_cast<uint8_t>('c')] = 0b01u;
    table[static_cast<uint8_t>('G')] = 0b10u;
    table[static_cast<uint8_t>('g')] = 0b10u;
    table[static_cast<uint8_t>('T')] = 0b11u;
    table[static_cast<uint8_t>('t')] = 0b11u;

    return table;
}

// Instantiate the table in static memory (read-only, hot cache).
// Always cast your input char to uint8_t when indexing into this table
// to avoid negative indices due to sign extension.
// Example: auto p = gnx::lut::encode_2bit[static_cast<uint8_t>(ch)];
thread_local static constexpr auto encode_2bit = create_2bit_encode_table();

// Decode table for 2-bit values (0-3) to nucleotide characters.
// Example: char base = gnx::lut::decode_2bit[bits & 0x03u];
thread_local static constexpr auto decode_2bit
=   std::array<char, 4>{'A', 'C', 'G', 'T'};

} // namespace gnx::lut
