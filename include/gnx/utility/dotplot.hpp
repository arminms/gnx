// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <ranges>
#include <string>
#include <filesystem>

#include <g3p/gnuplot>

#include <gnx/utility/detail/dotplot.hpp>

namespace gnx {

template<std::ranges::random_access_range Range>
inline void dotplot
(   Range const& range
,   std::size_t width = 0
,   std::string_view filename = ""
)
{   std::string term = "pngcairo";
    if (!filename.empty())
    {   auto ext = std::filesystem::path(filename).extension().string();
        if (ext == ".svg")
            term = "svg";
        else if (ext == ".pdf")
            term = "pdfcairo";
        else if (ext == ".png")
            term = "pngcairo";
        else if (ext == ".jpg" || ext == ".jpeg")
            term = "jpeg";
        else
            log_error("unsupported image file format for dotplot output");
    }

    width = std::max(width, std::size(range));
    // cap the width to HD resolution to avoid excessive memory usage
    width = std::min(width, static_cast<std::size_t>(1920));

    g3p::gnuplot gp;
    if (!filename.empty())
        gp("set output '%s'", filename.data());
    gp("set term %s size %d,%d", term.c_str(), width, width);
    std::size(range) <= 500
    ?   detail::dotplot_image(gp, range)
    :   detail::dotplot_pm3d(gp, range)
    ;

#if defined(__CLING__)
    if (filename.empty())
        display(gp, false);
#endif // __CLING__
}

template
<   std::ranges::random_access_range Range1
,   std::ranges::random_access_range Range2
>
inline void dotplot
(   Range1 const& range1
,   Range2 const& range2
,   std::size_t width = 0
,   std::string_view filename = ""
)
{   std::string term = "pngcairo";
    if (!filename.empty())
    {   auto ext = std::filesystem::path(filename).extension().string();
        if (ext == ".svg")
            term = "svg";
        else if (ext == ".pdf")
            term = "pdfcairo";
        else if (ext == ".png")
            term = "pngcairo";
        else if (ext == ".jpg" || ext == ".jpeg")
            term = "jpeg";
        else
            log_error("unsupported image file format for dotplot output");
    }

    width = std::max(width, std::size(range1));
    // cap the width to HD resolution to avoid excessive memory usage
    width = std::min(width, static_cast<std::size_t>(1920));
    size_t height = std::size(range2) * width / std::size(range1);

    g3p::gnuplot gp;
    if (!filename.empty())
        gp("set output '%s'", filename.data());
    gp("set term %s size %d,%d", term.c_str(), width, height);
    std::size(range1) <= 500
    ?   detail::dotplot_image(gp, range1, range2)
    :   detail::dotplot_pm3d(gp, range1, range2)
    ;

#if defined(__CLING__)
    if (filename.empty())
        display(gp, false);
#endif // __CLING__
}

} // namespace gnx