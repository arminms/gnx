// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <gnx/algorithms/count.hpp>

namespace gnx {

/// @brief Calculate the AT ratio of a DNA sequence range.
/// @tparam Range Range type
/// @param range The sequence range
/// @return The AT ratio as a double type (value between 0 and 1)
template<std::ranges::input_range Range>
inline double at_ratio(const Range& range)
{   auto map = gnx::count(range);
    double at = double(map['A']) + double(map['T']);
    double gc = double(map['G']) + double(map['C']);
    return at / (at + gc);
}
/// @brief Calculate the AT ratio of a DNA sequence range with parallel execution.
/// @tparam ExecPolicy Execution policy type (e.g., gnx::execution::par)
/// @tparam Range Range type
/// @param policy Execution policy controlling algorithm execution
/// @param range The sequence range
/// @return The AT ratio as a double type (value between 0 and 1)
template<typename ExecPolicy, std::ranges::input_range Range>
inline double at_ratio(ExecPolicy&& policy, const Range& range)
{   auto map = gnx::count(std::forward<ExecPolicy>(policy), range);
    double at = double(map['A']) + double(map['T']);
    double gc = double(map['G']) + double(map['C']);
    return at / (at + gc);
}
/// @brief Calculate the GC ratio of a DNA sequence range.
/// @tparam Range Range type
/// @param range The sequence range
/// @return The GC ratio as a double type (value between 0 and 1)
template<std::ranges::input_range Range>
inline double gc_ratio(const Range& range)
{   auto map = gnx::count(range);
    double at = double(map['A']) + double(map['T']);
    double gc = double(map['G']) + double(map['C']);
    return gc / (at + gc);
}
/// @brief Calculate the GC ratio of a DNA sequence range with parallel execution.
/// @tparam ExecPolicy Execution policy type (e.g., gnx::execution::par)
/// @tparam Range Range type
/// @param policy Execution policy controlling algorithm execution
/// @param range The sequence range
/// @return The GC ratio as a double type (value between 0 and 1)
template<typename ExecPolicy, std::ranges::input_range Range>
inline double gc_ratio(ExecPolicy&& policy, const Range& range)
{   auto map = gnx::count(std::forward<ExecPolicy>(policy), range);
    double at = double(map['A']) + double(map['T']);
    double gc = double(map['G']) + double(map['C']);
    return gc / (at + gc);
}
/// @brief Calculate the AT content percentage of a DNA sequence range.
/// @tparam Range Range type
/// @param range The sequence range
/// @return The AT content as a percentage (value between 0 and 100)
template<std::ranges::input_range Range>
inline double at_content(const Range& range)
{   return at_ratio(range) * 100.0;
}
/// @brief Calculate the AT content percentage of a DNA sequence range with parallel execution.
/// @tparam ExecPolicy Execution policy type (e.g., gnx::execution::par)
/// @tparam Range Range type
/// @param policy Execution policy controlling algorithm execution
/// @param range The sequence range
/// @return The AT content as a percentage (value between 0 and 100)
template<typename ExecPolicy, std::ranges::input_range Range>
inline double at_content(ExecPolicy&& policy, const Range& range)
{   return at_ratio(std::forward<ExecPolicy>(policy), range) * 100.0;
}
/// @brief Calculate the GC content percentage of a DNA sequence range.
/// @tparam Range Range type
/// @param range The sequence range
/// @return The GC content as a percentage (value between 0 and 100)
template<std::ranges::input_range Range>
inline double gc_content(const Range& range)
{   return gc_ratio(range) * 100.0;
}
/// @brief Calculate the GC content percentage of a DNA sequence range with parallel execution.
/// @tparam ExecPolicy Execution policy type (e.g., gnx::execution::par)
/// @tparam Range Range type
/// @param policy Execution policy controlling algorithm execution
/// @param range The sequence range
/// @return The GC content as a percentage (value between 0 and 100)
template<typename ExecPolicy, std::ranges::input_range Range>
inline double gc_content(ExecPolicy&& policy, const Range& range)
{   return gc_ratio(std::forward<ExecPolicy>(policy), range) * 100.0;
}

} // namespace gnx