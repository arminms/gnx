// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <gnx/utility/detail/print.hpp>

namespace gnx {

template<std::ranges::input_range Range>
inline std::string print
(   Range const& range
,   std::array<std::string, 256> const& color_scheme
,   std::size_t line_width = 80
,   std::size_t start_index = 1
,   std::size_t separator = 10
)
{   return detail::print_to_string
    (   range
    ,   color_scheme
    ,   line_width
    ,   start_index
    ,   separator
    );
}

template<std::ranges::input_range Range>
inline std::string print
(   Range const& range
,   std::size_t line_width = 80
,   std::size_t start_index = 1
,   std::size_t separator = 10
)
{   return detail::print_to_string
    (   range
    ,   gnx::is_peptide(range)
        ?   gnx::color_scheme::na_warn
        :   gnx::color_scheme::na
    ,   line_width
    ,   start_index
    ,   separator
    );
}

inline std::string print
(   gnx::alignment_result const& result
,   std::size_t line_width = 80
)
{   return detail::print_alignment_to_string(result, line_width);
}

}   // namespace gnx