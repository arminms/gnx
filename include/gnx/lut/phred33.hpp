// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
#pragma once

#include <array>
#include <cmath>

namespace gnx::lut {

// Generate the Lookup Table at Compile Time
// This maps every ASCII character to Error Probability equivalent.
constexpr std::array<double, 256> create_phred33_table()
{   std::array<double, 256> table{};
    table.fill(1.0); // C++20 feature
    // for (size_t i = 0; i < 256; ++i)
    //     table[i] = 1.0; 
    // initialize ASCII 33 (!) to 126 (~)
    for (int c = 33; c < 127; ++c)
    {   int q_score = c - 33;
        // P = 10 ^ (-Q/10)
        table[c] = std::pow(10.0, -q_score / 10.0);
    }
    return table;
}

// Instantiate the table in static memory (read-only, hot cache).
// Always cast your input char to uint8_t when indexing into this table
// to avoid negative indices due to sign extension.
// Example: double p = gnx::lut::phred33[static_cast<uint8_t>(ch)];
thread_local static constexpr auto phred33 = create_phred33_table();

} // namespace gnx::lut
