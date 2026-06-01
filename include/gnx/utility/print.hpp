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
        ?   color_scheme::aa_clustal
        :   color_scheme::na
    ,   line_width
    ,   start_index
    ,   separator
    );
}

inline std::string print
(   gnx::alignment_result const& result
,   std::size_t line_width = 60
)
{
#ifdef __CLING__
    auto bundle = nlohmann::json::object();
    bundle["text/plain"]
    =   detail::print_alignment_to_string(result, line_width);;
    xeus::get_interpreter().clear_output(true);
    xeus::get_interpreter().display_data
    (   bundle
    ,   nlohmann::json::object()
    ,   nlohmann::json::object()
    );
    return std::string();
#else
    return detail::print_alignment_to_string(result, line_width);
#endif //__CLING__
}

#ifdef __CLING__
    nlohmann::json mime_bundle_repr(alignment_result const& result)
    {   return print(result);
    }
#endif //__CLING__


}   // namespace gnx