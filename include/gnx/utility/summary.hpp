// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <cstdlib>
#include <filesystem>
#include <ranges>
#include <string>
#include <string_view>
#include <type_traits>

#include <g3p/gnuplot>

#include <gnx/utility/detail/summary.hpp>

namespace gnx {

/// @brief Generates a sector (donut) summary plot for a sequence or sequence file.
///
/// Dispatches on the type of `range`:
///
/// **Sequence range** (`generic_sequence` or any character range):
///   Produces a polar donut chart with one concentric sector per residue type.
///   Each sector is sized proportionally to its count and coloured using the
///   standard nucleotide palette (A/C/G/T/U/N) or the ClustalX amino-acid
///   palette.  The centre of the plot shows the total length; for DNA/RNA
///   sequences the GC content is also displayed there.
///
/// **Filename** (a type convertible to `std::filesystem::path`):
///   Reads the FASTA/FASTQ file (plain or bgzip-compressed with an `.fai`
///   index) and produces a polar donut chart with one sector per sequence-
///   length bucket, sized by the number of sequences in that bucket.  The
///   centre shows the total sequence count and total bases / residues.
///
/// @tparam Range   Any `std::ranges::input_range` over characters, a
///                 `gnx::generic_sequence`, or a string-like filename.
/// @param  range     The input: a sequence object, a view, or a filename.
/// @param  size      Width and height of the output image in pixels (default 600).
/// @param  filename  If non-empty, write to this file; the terminal is chosen
///                   from the extension (.png, .svg, .pdf, .jpg/.jpeg).
///                   If empty and compiled under Cling/Jupyter, the plot is
///                   displayed inline; otherwise gnuplot opens a persistent
///                   window.
template<std::ranges::input_range Range>
inline void summary
(   Range const& range
,   std::size_t  size     = 500
,   std::string_view filename = ""
)
{   // Choose gnuplot terminal from file extension
    std::string term = "pngcairo";
    if (!filename.empty())
    {   auto ext = std::filesystem::path(filename).extension().string();
        if      (ext == ".svg")                   term = "svg";
        else if (ext == ".pdf")                   term = "pdfcairo";
        else if (ext == ".png")                   term = "pngcairo";
        else if (ext == ".jpg" || ext == ".jpeg") term = "jpeg";
        else
        {   std::cerr
            <<  fmt::format
                (   "gnx::summary(): unsupported output format '{}'\n"
                ,   ext
                );
            return;
        }
    }

    // Auto-select gnuplot 6 if G3P_GNUPLOT_PATH is not already set:
    // prefer 'gnuplot6' wrapper on PATH, which the gnx project provides.
    if (!std::getenv("G3P_GNUPLOT_PATH"))
    {   // look for a gnuplot6 binary in standard user locations
        for (auto candidate : {"gnuplot6", "gnuplot"})
        {   std::string probe = std::string("which ") + candidate + " > /dev/null 2>&1";
            if (0 == std::system(probe.c_str()))
            {   setenv("G3P_GNUPLOT_PATH", candidate, 0);
                break;
            }
        }
    }

    g3p::gnuplot gp;

    // Sector plots require gnuplot 6+
    if (gp.version() < 6.0)
        throw std::runtime_error
        (   fmt::format
            (   "gnx::summary() requires gnuplot 6+, but found version {}. "
                "Set G3P_GNUPLOT_PATH to a gnuplot 6 binary (e.g. 'gnuplot6')."
            ,   gp.version_string()
            )
        );

    gp("set term %s size %zu,%zu enhanced font 'sans,10'", term.c_str(), size, size);
    if (!filename.empty())
        gp("set output '%s'", filename.data());

    if constexpr (requires { typename Range::map_type; })
    {   // generic_sequence: summarise residue composition
        std::string seq_id;
        if (range.has("_id"))
            seq_id = std::any_cast<std::string>(const_cast<Range&>(range)["_id"]);
        detail::summary_seq(gp, std::begin(range), std::end(range), seq_id);
    }
    else if constexpr (std::is_convertible_v<Range, std::filesystem::path>)
    {   // Filename: summarise sequence-length distribution
        std::string path(range);
        if (detail::is_valid_url(range))
        {   auto downloaded = gnx::wget(range);
            if (downloaded().empty())
                throw std::runtime_error
                (   fmt::format("Failed to download the sequence from {}", range)
                );
            else
                detail::summary_fs(gp, downloaded());
        } 
        else if (std::filesystem::exists(path + ".fai"))
            detail::summary_vv(gp, path);
        else
            detail::summary_fs(gp, path);
    }
    else
    {   // Plain character range / view: summarise residue composition
        detail::summary_seq(gp, std::begin(range), std::end(range));
    }

#if defined(__CLING__)
    if (filename.empty())
        display(gp, false);
#endif // __CLING__
}

} // namespace gnx
