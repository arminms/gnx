// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/core.h>
#include <fmt/format.h>

#include <g3p/gnuplot>

#include <gnx/sq.hpp>
#include <gnx/algorithms/count.hpp>
#include <gnx/algorithms/is_peptide.hpp>
#include <gnx/backend/virtual_vector.hpp>
#include <gnx/backend/forward_stream.hpp>

namespace gnx::detail {

// -- plot color helpers -------------------------------------------------------

/// Returns a gnuplot-compatible hex RGB color for a nucleotide character.
inline uint32_t na_plot_color(char c) noexcept
{   switch (static_cast<char>(std::toupper(static_cast<unsigned char>(c))))
    {   case 'A': return 0x00CC44; // green
        case 'C': return 0x3399FF; // blue
        case 'G': return 0x808080; // gray
        case 'T': return 0xFF3300; // red
        case 'U': return 0xFF6600; // orange-red
        case 'N': return 0xFFBB00; // amber
        default:  return 0xCC33CC; // magenta (ambiguous)
    }
}

/// Returns a gnuplot-compatible hex RGB color for an amino acid character
/// following the ClustalX colour scheme.
inline uint32_t aa_plot_color(char c) noexcept
{   switch (static_cast<char>(std::toupper(static_cast<unsigned char>(c))))
    {   case 'A': case 'I': case 'L': case 'M': case 'V':
            return 0x5588FF; // hydrophobic (blue)
        case 'K': case 'R':
            return 0xEE3333; // basic (red)
        case 'D': case 'E':
            return 0xCC22CC; // acidic (magenta)
        case 'N': case 'Q': case 'S': case 'T':
            return 0x22CC22; // polar (green)
        case 'C':
            return 0xFF99CC; // cysteine (pink)
        case 'G':
            return 0xFF8833; // glycine (orange)
        case 'P':
            return 0xDDCC00; // proline (yellow)
        case 'F': case 'H': case 'W': case 'Y':
            return 0x22BBCC; // aromatic (cyan)
        case '*':
            return 0xFF4444; // stop codon (red)
        default:
            return 0xAAAAAA; // unknown (gray)
    }
}

/// Returns a perceptually-spread rainbow color for bucket index `idx` out of
/// `total` buckets. Hue sweeps from red (idx=0) to blue-violet (idx=total-1).
inline uint32_t bucket_color(std::size_t idx, std::size_t total) noexcept
{   if (total <= 1) return 0x4499EE;
    double hue = 280.0 * static_cast<double>(idx) / static_cast<double>(total - 1);
    double h   = hue / 60.0;
    double c = 0.75, m = 0.20;
    double x = c * (1.0 - std::abs(std::fmod(h, 2.0) - 1.0));
    double r = m, g = m, b = m;
    if      (h < 1.0) { r += c; g += x; }
    else if (h < 2.0) { r += x; g += c; }
    else if (h < 3.0) { g += c; b += x; }
    else if (h < 4.0) { g += x; b += c; }
    else if (h < 5.0) { r += x; b += c; }
    else               { r += c; b += x; }
    return   (static_cast<uint32_t>(std::min(r, 1.0) * 255) << 16)
           | (static_cast<uint32_t>(std::min(g, 1.0) * 255) <<  8)
           |  static_cast<uint32_t>(std::min(b, 1.0) * 255);
}

// -- human-readable size (no ANSI codes) --------------------------------------

inline std::string hrs_plain(std::size_t size, bool peptide = false)
{   const char* unit = peptide ? "aa" : "bp";
    if (size < 1'000ULL)
        return fmt::format("{} {}", size, unit);
    if (size < 1'000'000ULL)
        return fmt::format("{:.1f} K{}", size / 1.0e3, unit);
    if (size < 1'000'000'000ULL)
        return fmt::format("{:.1f} M{}", size / 1.0e6, unit);
    return fmt::format("{:.2f} G{}", size / 1.0e9, unit);
}

// -- sector entry and bucket builder ------------------------------------------

struct sector_entry
{   std::string  label; ///< text label for the sector
    std::size_t  count; ///< raw count (proportional to sector angle)
    uint32_t     color; ///< fill colour as 0xRRGGBB
};

/// Converts a length-frequency map into a vector of at most `max_bk` named
/// sector_entry items.  When the number of unique lengths exceeds `max_bk`,
/// entries are merged into logarithmically-spaced buckets.
inline std::vector<sector_entry>
make_length_buckets
(   std::map<std::size_t, std::size_t> const& dist
,   std::size_t max_bk = 12
)
{   if (dist.empty()) return {};

    std::vector<sector_entry> result;

    if (dist.size() <= max_bk)
    {   std::size_t idx = 0, total = dist.size();
        for (auto const& [len, cnt] : dist)
            result.push_back({ std::to_string(len), cnt, bucket_color(idx++, total) });
        return result;
    }

    // Logarithmic bucketing into at most max_bk bins
    auto min_len = dist.begin()->first;
    auto max_len = dist.rbegin()->first;
    double log_min = std::log2(static_cast<double>(std::max(min_len, std::size_t{1})));
    double log_max = std::log2(static_cast<double>(max_len));
    double step    = (log_max - log_min) / static_cast<double>(max_bk);

    struct bk_t { std::size_t lo, hi, count; };
    std::vector<bk_t> bks(max_bk);
    for (std::size_t i = 0; i < max_bk; ++i)
    {   bks[i].lo    = static_cast<std::size_t>(std::pow(2.0, log_min + static_cast<double>(i    ) * step));
        bks[i].hi    = static_cast<std::size_t>(std::pow(2.0, log_min + static_cast<double>(i + 1) * step));
        bks[i].count = 0;
    }
    bks.back().hi = max_len + 1;   // ensure last bin is inclusive

    for (auto const& [len, cnt] : dist)
        for (auto& bk : bks)
            if (len >= bk.lo && len < bk.hi)
            {   bk.count += cnt;
                break;
            }

    // Collect non-empty buckets
    std::size_t non_empty{0};
    for (auto const& bk : bks)
        if (bk.count > 0) ++non_empty;

    std::size_t idx{0};
    for (auto const& bk : bks)
    {   if (bk.count == 0) continue;
        std::string lbl = (bk.lo == bk.hi - 1)
            ? std::to_string(bk.lo)
            : fmt::format("{}-{}", bk.lo, bk.hi - 1);
        result.push_back({ lbl, bk.count, bucket_color(idx++, non_empty) });
    }
    return result;
}

// -- gnuplot $data block builder ----------------------------------------------

/// Generates a gnuplot named inline data block "$data" from a collection of
/// sector_entry items.  The block has the following columns:
///   1  start_angle (degrees, clockwise from top)
///   2  inner_radius
///   3  span_angle  (degrees)
///   4  ring_width
///   5  fill_color  (0xRRGGBB)
///   6  label       (string token – no spaces)
///   7  percentage  (float)
///   8  count       (integer)
template<typename Entries>
std::string make_sector_data_block
(   Entries const& entries
,   double         total_count
,   double         r_inner
,   double         r_width
)
{   std::string out;
    out.reserve(entries.size() * 72);
    out += "$data <<EOD\n";
    double angle{0.0};
    for (auto const& e : entries)
    {   double span = 360.0 * static_cast<double>(e.count) / total_count;
        double pct  = 100.0 * static_cast<double>(e.count) / total_count;
        out += fmt::format
        (   "{:.6f} {:.3f} {:.6f} {:.3f} 0x{:06X} {} {:.2f} {}\n"
        ,   angle, r_inner, span, r_width, e.color, e.label, pct, e.count
        );
        angle += span;
    }
    out += "EOD";
    return out;
}

// -- common polar plot setup --------------------------------------------------

inline void setup_polar_sector_plot
(   g3p::gnuplot const& gp
,   double r_max
)
{   gp("set polar");
    gp("set angles degrees");
    gp("set theta top clockwise");
    gp("set size ratio -1");
    gp("set xrange [%f:%f]", -r_max, r_max);
    gp("set yrange [%f:%f]", -r_max, r_max);
    gp("unset border");
    gp("unset tics");
    gp("unset raxis");
    gp("unset key");
}

// -- plot command builder -----------------------------------------------------

/// Returns the gnuplot plot command string for the sector chart.
/// Sectors smaller than `min_span_deg` degrees receive no label.
inline std::string make_sector_plot_cmd(double r_label, double min_span_deg = 2.5)
{   // col1=start_angle  col2=inner_r  col3=span  col4=width  col5=color(rgb)
    // col6=label(string) col7=pct  col8=count
    // Labels: character/range on top, percentage on bottom
    return fmt::format
    (   "plot $data using 1:2:3:4:5 "
            "with sectors fc rgb variable fill solid 0.85 border lc 'white' lw 1.5 notitle, "
        "$data using ($3>{:.3f}?$1+$3/2.0:1/0):({:.3f})"
            ":(sprintf(\"%s\\n%.1f%%\",stringcolumn(6),$7)) "
            "with labels center font ',9' tc rgb '#222222' notitle"
    ,   min_span_deg
    ,   r_label
    );
}

// -- sequence composition summary ---------------------------------------------

template<typename Iterator>
void summary_seq
(   g3p::gnuplot const& gp
,   Iterator first
,   Iterator last
)
{   bool peptide = gnx::is_peptide(first, last);
    auto counts  = gnx::count(first, last);
    auto total   = static_cast<std::size_t>(std::distance(first, last));

    if (total == 0) return;

    // Sort by count descending so the dominant type gets the first sector
    std::vector<std::pair<char, std::size_t>> sorted(counts.begin(), counts.end());
    std::sort
    (   sorted.begin(), sorted.end()
    ,   [](auto const& a, auto const& b) { return a.second > b.second; }
    );

    if (peptide)
    {   // --- Protein: single coloured ring (ClustalX colours) ---------------
        std::vector<sector_entry> entries;
        entries.reserve(sorted.size());
        for (auto const& [ch, cnt] : sorted)
            entries.push_back
            ({  std::string(1, static_cast<char>(std::toupper(static_cast<unsigned char>(ch))))
            ,   cnt
            ,   aa_plot_color(ch)
            });

        constexpr double r_inner = 3.0;
        constexpr double r_width = 3.5;
        constexpr double r_label = 8.2;
        constexpr double r_max   = 11.5;

        auto data_block = make_sector_data_block
            (entries, static_cast<double>(total), r_inner, r_width);
        gp("%s", data_block.c_str());

        auto center = fmt::format("Total\\n{}", hrs_plain(total, true));
        gp("set label 1 \"%s\" at 0,0 center font ',11' front", center.c_str());

        setup_polar_sector_plot(gp, r_max);
        auto plot_cmd = make_sector_plot_cmd(r_label);
        gp("%s", plot_cmd.c_str());
    }
    else
    {   // --- Nucleic acid: two-ring layout inspired by sectors.3.gnu --------
        // Outer ring  (lavender, proportional): nucleotide letter inside
        // Inner ring  (standard NA colours)   : percentage label inside
        // Center hole (white disk)             : total bp + GC%

        // GC content
        std::size_t gc{0};
        for (auto const& [ch, cnt] : sorted)
        {   char uc = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
            if (uc == 'G' || uc == 'C') gc += cnt;
        }
        double gc_pct = 100.0 * static_cast<double>(gc) / static_cast<double>(total);

        // Build $data block
        // cols: start_angle  span_angle  color  label  pct  count
        std::string block;
        block.reserve(sorted.size() * 60);
        block += "$data <<EOD\n";
        double angle = 0.0;
        for (auto const& [ch, cnt] : sorted)
        {   char uc = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
            double span = 360.0 * static_cast<double>(cnt) / static_cast<double>(total);
            double pct  = 100.0 * static_cast<double>(cnt) / static_cast<double>(total);
            block += fmt::format
            (   "{:.6f} {:.6f} 0x{:06X} {} {:.2f} {}\n"
            ,   angle, span, na_plot_color(ch), uc, pct, cnt
            );
            angle += span;
        }
        block += "EOD";
        gp("%s", block.c_str());

        // Ring geometry
        //   outer lavender ring : r 6.0 → 9.5  (width 3.5)
        //   inner coloured ring : r 3.5 → 5.9  (width 2.4)
        //   white center disk   : r 0   → 3.4
        constexpr double r_out_in  = 6.0,  r_out_w  = 3.5;
        constexpr double r_in_in   = 3.5,  r_in_w   = 2.4;
        constexpr double r_out_lbl = 7.75, r_in_lbl  = 4.7;
        constexpr double r_center  = 3.4,  r_max     = 11.0;

        // White center disk sits in front of everything
        gp
        (   "set object 1 circle at 0,0 size %.2f "
            "fillcolor rgb 'white' fillstyle solid 1.0 noborder front"
        ,   r_center
        );

        // Center label: total + GC%
        auto center_str = fmt::format
            ("Total\\n{}\\nGC: {:.1f}%", hrs_plain(total), gc_pct);
        gp
        (   "set label 1 \"%s\" at 0,0 center font 'sans,11' front"
        ,   center_str.c_str()
        );

        setup_polar_sector_plot(gp, r_max);

        // Plot:
        //   1. Outer lavender sectors (uniform colour, sector dividers)
        //   2. Inner nucleotide-coloured sectors (col 3 = rgb variable)
        //   3. Nucleotide letter centred inside outer ring (white, bold)
        //   4. Percentage centred inside inner ring (dark, omit < 4 deg)
        auto plot_cmd = fmt::format
        (   "plot "
            "$data using 1:({:.3f}):2:({:.3f}) "
                "with sectors fc rgb '#AA99CC' fill solid 0.50 "
                "border lc '#888899' lw 1.5 notitle, "
            "$data using 1:({:.3f}):2:({:.3f}):3 "
                "with sectors fc rgb variable fill solid 0.90 "
                "border lc 'white' lw 1.5 notitle, "
            "$data using ($2>3.0?$1+$2/2.0:1/0):({:.3f}):(stringcolumn(4)) "
                "with labels center font 'sans Bold,13' tc rgb 'white' notitle, "
            "$data using ($2>4.0?$1+$2/2.0:1/0):({:.3f}):(sprintf(\"%.1f%%\",$5)) "
                "with labels center font 'sans,8' tc rgb '#222222' notitle"
        ,   r_out_in, r_out_w
        ,   r_in_in,  r_in_w
        ,   r_out_lbl
        ,   r_in_lbl
        );
        gp("%s", plot_cmd.c_str());
    }
}

// -- filename summary: FASTA/FASTQ with FAI index -----------------------------

inline void summary_vv(g3p::gnuplot const& gp, std::string_view filename)
{   gnx::virtual_vector<gnx::sq> vv{filename};
    if (vv.empty())
    {   std::cerr
        <<  fmt::format("gnx::summary(): {} is empty or unsupported\n", filename);
        return;
    }

    // Detect nucleotide vs peptide from first sequence
    auto first_seq = vv[0];
    bool peptide = gnx::is_peptide(std::begin(first_seq), std::end(first_seq));

    // Collect length distribution via the FAI entries (no decompression needed)
    std::map<std::size_t, std::size_t> length_dist;
    std::size_t total_bp{0};
    for (std::size_t i = 0; i < vv.size(); ++i)
    {   auto len = static_cast<std::size_t>(vv.entry(i).length);
        length_dist[len]++;
        total_bp += len;
    }
    std::size_t total_seqs = vv.size();

    auto buckets = make_length_buckets(length_dist);
    if (buckets.empty()) return;

    // Ring geometry
    constexpr double r_inner = 3.0;
    constexpr double r_width = 3.5;
    constexpr double r_label = 8.2;
    constexpr double r_max   = 11.5;

    auto data_block = make_sector_data_block(buckets, static_cast<double>(total_seqs), r_inner, r_width);
    gp("%s", data_block.c_str());

    // Center label: total sequences and total bases
    auto center = fmt::format
    (   "Total\\n{} seq{}\\n{}"
    ,   total_seqs
    ,   total_seqs != 1 ? "s" : ""
    ,   hrs_plain(total_bp, peptide)
    );
    gp("set label 1 \"%s\" at 0,0 center font ',11' front", center.c_str());

    // File name as title
    std::string title_cmd = fmt::format
    (   "set title \"{}\" font ',12'"
    ,   std::filesystem::path(filename).filename().string()
    );
    gp("%s", title_cmd.c_str());

    setup_polar_sector_plot(gp, r_max);
    auto plot_cmd = make_sector_plot_cmd(r_label);
    gp("%s", plot_cmd.c_str());
}

// -- filename summary: FASTA/FASTQ without FAI index (streaming) --------------

inline void summary_fs(g3p::gnuplot const& gp, std::string_view filename)
{   gnx::forward_stream<gnx::sq> stream{filename};
    auto it     = stream.begin();
    auto end_it = stream.end();

    if (it == end_it || it->sequence().empty())
    {   std::cerr
        <<  fmt::format("gnx::summary(): {} is not a supported file format\n", filename);
        return;
    }

    // Infer type from first record; FASTQ files always contain nucleotides
    bool is_fq  = !stream.quality().empty();
    bool peptide = is_fq
        ? false
        : gnx::is_peptide(stream.sequence().begin(), stream.sequence().end());

    // Stream through collecting length frequencies
    std::map<std::size_t, std::size_t> length_dist;
    std::size_t total_bp{0}, total_seqs{0};
    for (; it != end_it; ++it, ++total_seqs)
    {   auto len = it->sequence().size();
        length_dist[len]++;
        total_bp += len;
    }

    auto buckets = make_length_buckets(length_dist);
    if (buckets.empty()) return;

    // Ring geometry
    constexpr double r_inner = 3.0;
    constexpr double r_width = 3.5;
    constexpr double r_label = 8.2;
    constexpr double r_max   = 11.5;

    auto data_block = make_sector_data_block(buckets, static_cast<double>(total_seqs), r_inner, r_width);
    gp("%s", data_block.c_str());

    auto center = fmt::format
    (   "Total\\n{} seq{}\\n{}"
    ,   total_seqs
    ,   total_seqs != 1 ? "s" : ""
    ,   hrs_plain(total_bp, peptide)
    );
    gp("set label 1 \"%s\" at 0,0 center font ',11' front", center.c_str());

    std::string title_cmd = fmt::format
    (   "set title \"{}\" font ',12'"
    ,   std::filesystem::path(filename).filename().string()
    );
    gp("%s", title_cmd.c_str());

    setup_polar_sector_plot(gp, r_max);
    auto plot_cmd = make_sector_plot_cmd(r_label);
    gp("%s", plot_cmd.c_str());
}

} // namespace gnx::detail
