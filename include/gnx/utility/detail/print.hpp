// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <string>
#include <array>
#include <ranges>
#include <algorithm>

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
#include <gnx/algorithms/local_align.hpp>

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
            ,   ansi::ESC[fg::bright_magenta]
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
            ,   ansi::ESC[fg::bright_cyan]
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
            ,   ansi::ESC[fg::bright_magenta]
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
            ,   gnx::ansi::ESC[fg::bright_cyan]
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

inline std::string print_alignment_to_string
(   gnx::alignment_result const& result
,   std::size_t line_width = 80
)
{   fmt::memory_buffer buf;
    auto size = result.aligned_seq1.size();
    auto seq1_gaps = std::count(result.aligned_seq1.begin(), result.aligned_seq1.end(), '-');
    auto seq2_gaps = std::count(result.aligned_seq2.begin(), result.aligned_seq2.end(), '-');
    std::size_t start_index_q = result.max_i - result.aligned_seq1.size() + seq1_gaps + 1;
    std::size_t start_index_s = result.max_j - result.aligned_seq2.size() + seq2_gaps + 1;
    auto gap_count = seq1_gaps + seq2_gaps;
    auto identity_count = std::ranges::count_if
    (   std::views::iota(0u, size)
    ,   [&result](std::size_t i)
        {   char a = result.aligned_seq1[i];
            char b = result.aligned_seq2[i];
            return a == b;
        }
    );
    auto identity_percentage
    =   size > 0
    ?   static_cast<double>(identity_count) / size * 100.0
    :   0.0
    ;
    auto score_per_residue
    =   gap_count < size
    ?   static_cast<double>(result.score) / (size - gap_count)
    :   0.0
    ;
    bool peptide = gnx::is_peptide(result.aligned_seq1);
    auto color = peptide
    ?   gnx::color_scheme::aa_clustal
    :   gnx::color_scheme::na
    ;
    auto match_color = peptide
    ?   gnx::color_scheme::aa_clustal_inverted
    :   gnx::color_scheme::na_inverted
    ;
    auto q_start = result.aligned_seq1.begin();
    auto s_start = result.aligned_seq2.begin();

    fmt::format_to
    (   std::back_inserter(buf)
    ,   "{}{:15}{}{}│ {}{}\n"
    ,   gnx::ansi::ESC[style::bold]
    ,   "SCORE"
    ,   gnx::ansi::ESC[style::reset]
    ,   gnx::ansi::vga::fg::ESC[250]
    ,   gnx::ansi::ESC[style::reset]
    ,   result.score
    );
    fmt::format_to
    (   std::back_inserter(buf)
    ,   "{}{:15}{}{}│ {}{}/{} ({:.2f}%)\n"
    ,   gnx::ansi::ESC[style::bold]
    ,   "IDENTITIES"
    ,   gnx::ansi::ESC[style::reset]
    ,   gnx::ansi::vga::fg::ESC[250]
    ,   gnx::ansi::ESC[style::reset]
    ,   identity_count
    ,   size
    ,   identity_percentage
    );
    fmt::format_to
    (   std::back_inserter(buf)
    ,   "{}{:15}{}{}│ {}{}/{} ({:.2f}%)\n"
    ,   gnx::ansi::ESC[style::bold]
    ,   "GAPS"
    ,   gnx::ansi::ESC[style::reset]
    ,   gnx::ansi::vga::fg::ESC[250]
    ,   gnx::ansi::ESC[style::reset]
    ,   gap_count
    ,   size
    ,   gap_count > 0 ? static_cast<double>(gap_count) / size * 100.0 : 0.0
    );
    fmt::format_to
    (   std::back_inserter(buf)
    ,   "{}{:15}{}{}│ {}{:.2f}\n"
    ,   gnx::ansi::ESC[style::bold]
    ,   "SCORE/RESIDUE"
    ,   gnx::ansi::ESC[style::reset]
    ,   gnx::ansi::vga::fg::ESC[250]
    ,   gnx::ansi::ESC[style::reset]
    ,   score_per_residue
    );
    fmt::format_to
    (   std::back_inserter(buf)
    ,   "{}{:═>{}}{:═>{}}\n"
    ,   gnx::ansi::vga::fg::ESC[250]
    ,   "╪"
    ,   16
    ,   "═"
    ,   line_width + 3
    );
    for
    (   std::size_t i = 0
    ;   i < size
    ;   i += line_width
    )
    {   std::size_t index_x10{start_index_q}, separator{};
        for (separator = 1; index_x10 % 10 != 0; ++separator, ++index_x10);
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "{:15}{}│ {:>{}}"
        ,   " "
        ,   gnx::ansi::vga::fg::ESC[250]
        ,   separator < std::to_string(index_x10).length()
            ?   ""
            :   std::to_string(index_x10)
        ,   separator
        );
        for 
        (   std::size_t j = 0, width = 1
        ;   j < line_width && i + j < size
        ;   ++j, ++width
        )
        {   auto qc = *(q_start + i + j + separator);
            if (qc != '-')
                ++index_x10;
            if (index_x10 % 10 == 0 && j < line_width - 1)
            {   fmt::format_to
                (   std::back_inserter(buf)
                ,   "{:{}}"
                ,   index_x10
                ,   width
                );
                width = 0;
            }
        }
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "{}\n"
        ,   gnx::ansi::ESC[style::reset]
        );
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "{}Query{}{:9} │ {}"
        ,   gnx::ansi::ESC[style::bold]
        ,   gnx::ansi::vga::fg::ESC[250]
        ,   start_index_q
        ,   gnx::ansi::ESC[style::reset]
        );
        for 
        (   std::size_t j = 0
        ;   j < line_width && i + j < size
        ;   ++j
        )
        {   auto qc = *(q_start + i + j);
            auto sc = *(s_start + i + j);
            if (qc != '-')
                start_index_q++;
            fmt::format_to
            (   std::back_inserter(buf)
            ,   "{}{}"
            ,   qc == sc
                ?   match_color[static_cast<uint8_t>(qc)]
                :   color[static_cast<uint8_t>(qc)]
            ,   gnx::ansi::ESC[style::reset]
            );
        }
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "{}\n"
        ,   gnx::ansi::ESC[style::reset]
        );
        index_x10 = start_index_s;
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "{}Sbjct{}{:9} │ {}"
        ,   gnx::ansi::ESC[style::bold]
        ,   gnx::ansi::vga::fg::ESC[250]
        ,   start_index_s
        ,   gnx::ansi::ESC[style::reset]
        );
        for 
        (   std::size_t j = 0
        ;   j < line_width && i + j < size
        ;   ++j
        )
        {   auto qc = *(q_start + i + j);
            auto sc = *(s_start + i + j);
            if (sc != '-')
                start_index_s++;
            fmt::format_to
            (   std::back_inserter(buf)
            ,   "{}{}"
            ,   qc == sc
                ?   match_color[static_cast<uint8_t>(sc)]
                :   color[static_cast<uint8_t>(sc)]
            ,   gnx::ansi::ESC[style::reset]
            );
        }
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "{}\n"
        ,   gnx::ansi::ESC[style::reset]
        );
        for (separator = 1; index_x10 % 10 != 0; ++separator, ++index_x10);
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "{:15}{}│ {:>{}}"
        ,   " "
        ,   gnx::ansi::vga::fg::ESC[250]
        ,   separator < std::to_string(index_x10).length()
            ?   ""
            :   std::to_string(index_x10)
        ,   separator
        );
        for 
        (   std::size_t j = 0, width = 1
        ;   j < line_width && i + j < size
        ;   ++j, ++width
        )
        {   auto sc = *(s_start + i + j + separator);
            if (sc != '-')
                ++index_x10;
            if (index_x10 % 10 == 0 && j < line_width - 1)
            {   fmt::format_to
                (   std::back_inserter(buf)
                ,   "{:{}}"
                ,   index_x10
                ,   width
                );
                width = 0;
            }
        }
        fmt::format_to
        (   std::back_inserter(buf)
        ,   "\n{:15}│ {}\n"
        ,   " "
        ,   gnx::ansi::ESC[style::reset]
        );
}
#ifdef __CLING__
    auto bundle = nlohmann::json::object();
    bundle["text/plain"] = fmt::to_string(buf);
    xeus::get_interpreter().clear_output(true);
    xeus::get_interpreter().display_data
    (   bundle
    ,   nlohmann::json::object()
    ,   nlohmann::json::object()
    );
    return std::string();
#else
    return fmt::to_string(buf);
#endif //__CLING__
}

} // namespace gnx::detail