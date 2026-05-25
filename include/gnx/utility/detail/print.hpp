// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <string>
#include <array>

#include <fmt/base.h>
#include <fmt/format.h>

#ifdef __CLING__
#   include <nlohmann/json.hpp>
#endif //__CLING__

#if defined(__CUDACC__) || defined(__HIPCC__)
    #include <thrust/device_vector.h>
    #include <thrust/host_vector.h>
#endif // __CUDACC__

#include <gnx/concepts.hpp>
#include <gnx/lut/color_schemes.hpp>
#include <gnx/algorithms/is_peptide.hpp>

namespace gnx::detail {

#if defined(__CUDACC__) || defined(__HIPCC__)
template<device_resident_iterator Iterator>
inline std::string print
(   Iterator first
,   Iterator last
,   std::array<std::string, 256> const& color_scheme
,   std::size_t line_width = 80
,   std::size_t start_index = 1
,   std::size_t separator = 10
)
{   using value_type = typename std::iterator_traits<Iterator>::value_type;
    thrust::host_vector<value_type> h_seq(first, last);
    return print
    (   h_seq.begin()
    ,   h_seq.end()
    ,   color_scheme
    ,   line_width
    ,   start_index
    ,   separator
    );
}
template<host_resident_iterator Iterator>
#else
template<typename Iterator>
#endif // __CUDACC__ || __HIPCC__
inline std::string print
(   Iterator first
,   Iterator last
,   std::array<std::string, 256> const& color_scheme
,   std::size_t line_width = 80
,   std::size_t start_index = 1
,   std::size_t separator = 10
)
{   assert(separator <= line_width);
    if (separator == 0 || separator > line_width)
        separator = line_width;
    fmt::memory_buffer buf;
    auto size = std::distance(first, last);

    if (separator != line_width)
    {   fmt::format_to
        (   std::back_inserter(buf)
        ,   "{}{}          # │"
        ,   gnx::ansi::ESC[style::bold]
        ,   gnx::ansi::vga::fg::ESC[250]
        );
        for (std::size_t i = separator; i <= line_width; i += separator)
            fmt::format_to
            (   std::back_inserter(buf)
            ,   "{:{}}"
            ,   i
            ,   separator + 1
            );
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "\n{}{}════════════╪"
        ,   gnx::ansi::ESC[style::bold]
        ,   gnx::ansi::vga::fg::ESC[250]
        );
        for (std::size_t i = separator; i <= line_width; i += separator)
            fmt::format_to
            (   std::back_inserter(buf)
            ,   "{:═>{}}"
            ,   i == line_width ? "╛" : "╧"
            ,   separator + 1
            );
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "{}\n"
        ,   gnx::ansi::ESC[style::reset]
        );
    }

    for
    (   std::size_t i = 0
    ;   i < size
    ;   i += line_width
    ,   start_index += line_width
    )
    {   fmt::format_to
        (   std::back_inserter(buf)
        ,   "{}{}{:11} │{} "
        ,   gnx::ansi::ESC[style::bold]
        ,   gnx::ansi::vga::fg::ESC[250]
        ,   start_index
        ,   gnx::ansi::ESC[style::reset]
        );
        for 
        (   std::size_t j = 0
        ;   j < line_width && i + j < size
        ;   ++j
        )
        {   fmt::format_to
            (   std::back_inserter(buf)
            ,   "{}"
            ,   color_scheme[static_cast<uint8_t>(*(first + i + j))]
            );
            if ((j + 1) % separator == 0)
                fmt::format_to
                (   std::back_inserter(buf)
                ,   "{} "
                ,   gnx::ansi::ESC[style::reset]
                );
        }
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "{}\n"
        ,   gnx::ansi::ESC[style::reset]
        );
    }

    return fmt::to_string(buf);
}

template<std::ranges::input_range Range>
inline std::string print_to_string
(   Range const& range
,   std::array<std::string, 256> const& color_scheme
,   std::size_t line_width = 80
,   std::size_t start_index = 1
,   std::size_t separator = 10
)
{   fmt::memory_buffer buf;
    if constexpr (requires { typename Range::map_type; })
    {   if (range.has("_id"))
            fmt::format_to
            (   std::back_inserter(buf)
            ,   "{}ID          {}│{} {}{}{}\n"
            ,   gnx::ansi::ESC[style::bold]
            ,   gnx::ansi::vga::fg::ESC[250]
            ,   gnx::ansi::ESC[style::reset]
            ,   gnx::ansi::ESC[style::italic]
            ,   std::any_cast<std::string>(range["_id"])
            ,   gnx::ansi::ESC[style::reset]
            );
        if (range.has("_desc"))
            fmt::format_to
            (   std::back_inserter(buf)
            ,   "{}DESCRIPTION {}│{} {}{}{}\n"
            ,   gnx::ansi::ESC[style::bold]
            ,   gnx::ansi::vga::fg::ESC[250]
            ,   gnx::ansi::ESC[style::reset]
            ,   gnx::ansi::ESC[style::italic]
            ,   std::any_cast<std::string>(range["_desc"])
            ,   gnx::ansi::ESC[style::reset]
            );
    }
#ifdef __CLING__
    auto bundle = nlohmann::json::object();
    bundle["text/plain"] = fmt::to_string(buf) + detail::print
    (   std::begin(range)
    ,   std::end(range)
    ,   color_scheme
    ,   line_width
    ,   start_index
    ,   separator
    );
    xeus::get_interpreter().clear_output(true);
    xeus::get_interpreter().display_data
    (   bundle
    ,   nlohmann::json::object()
    ,   nlohmann::json::object()
    );
    return std::string();
#else
    return fmt::to_string(buf) + detail::print
    (   std::begin(range)
    ,   std::end(range)
    ,   color_scheme
    ,   line_width
    ,   start_index
    ,   separator
    );
#endif //__CLING__
}

#ifdef __CLING__
template<std::ranges::input_range Range>
inline nlohmann::json print_to_bundle
(   Range const& range
,   std::size_t line_width = 60
,   std::size_t start_index = 1
,   std::size_t separator = 10
)
{   fmt::memory_buffer buf;
    if constexpr (requires { typename Range::map_type; })
    {   if (range.has("_id"))
            fmt::format_to
            (   std::back_inserter(buf)
            ,   "{}ID          {}│{} {}{}{}\n"
            ,   gnx::ansi::ESC[style::bold]
            ,   gnx::ansi::vga::fg::ESC[250]
            ,   gnx::ansi::ESC[style::reset]
            ,   gnx::ansi::ESC[style::italic]
            ,   std::any_cast<std::string>(range["_id"])
            ,   gnx::ansi::ESC[style::reset]
            );
        if (range.has("_desc"))
            fmt::format_to
            (   std::back_inserter(buf)
            ,   "{}DESCRIPTION {}│{} {}{}{}\n"
            ,   gnx::ansi::ESC[style::bold]
            ,   gnx::ansi::vga::fg::ESC[250]
            ,   gnx::ansi::ESC[style::reset]
            ,   gnx::ansi::ESC[style::italic]
            ,   std::any_cast<std::string>(range["_desc"])
            ,   gnx::ansi::ESC[style::reset]
            );
    }
    auto bundle = nlohmann::json::object();
    bundle["text/plain"] = fmt::to_string(buf) + detail::print
    (   std::begin(range)
    ,   std::end(range)
    ,   gnx::is_peptide(range)
        ?   gnx::color_scheme::aa_clustal
        :   gnx::color_scheme::na
    ,   line_width
    ,   start_index
    ,   separator
    );
    return bundle;
}
#endif //__CLING__

} // namespace gnx::detail