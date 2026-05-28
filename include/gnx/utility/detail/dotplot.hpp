// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

namespace gnx::detail {

template<std::ranges::random_access_range Range>
inline void dotplot_image
(   g3p::gnuplot const& gp
,   Range const& range
)
{   gp
    (   "unset key; unset colorbox" )
    (   "set origin 0,0" )
    (   "set size square" )
    (   "set tics out nomirror" )
    (   "set border 31 linecolor '#555555'" )
    (   "set xrange [0:%d]", std::size(range) )
    (   "set yrange [0:%d]", std::size(range) )
    (   "set palette gray" )
    (   "plot '-' u 1:2:3 w image" )
    ;
    for (size_t i = 0; i < std::size(range); ++i)
    {   for (size_t j = 0; j < std::size(range); ++j)
        {   int c = ((range[i] ^ range[j]) & 0xFF) ? 0xFFFFFF : 0x000000;
            gp << i << j << c << "\n";
        }
    }
    gp.end();
}

template<std::ranges::random_access_range Range>
inline void dotplot_pm3d
(   g3p::gnuplot const& gp
,   Range const& range
)
{   // adjust the tile size based on the sequence length
    size_t tile_size = static_cast<size_t>(std::log(std::size(range)));
    gp
    (   "set pm3d at b" )
    (   "unset key; unset colorbox" )
    (   "set tics out nomirror" )
    (   "set border 31 linecolor '#555555'" )
    (   "set view map" )
    (   "set palette gray" )
    (   "set size square" )
    (   "set ytics add ('' 0)" )
    (   "set xrange [0:%d]", std::size(range) )
    (   "set yrange [0:%d]", std::size(range) )
    ;
    gp
    (   "splot '-' using 1:2:3 with pm3d" );
    for (size_t i = 0; i < std::size(range); i += tile_size)
    {   for (size_t j = 0; j < std::size(range); j += tile_size)
            gp << i << j << (((range[i] ^ range[j]) & 0xFF) ? 1 : 0) << "\n";
        gp << "\n";
    }
    gp.end();
}

} // namespace gnx::detail