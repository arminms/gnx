// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#include <CLI/CLI.hpp>

#include <fmt/core.h>
#include <fmt/format.h>

int main
(   int argc
,   char** argv
)
{   CLI::App gnx
    {   fmt::format
        (   "gnx -- "
            "a command-line tool for biological sequence manipulation and analysis"
            "\nVersion: {}"
        ,   GNX_VERSION
        )
    };

    gnx.set_version_flag
    (   "-v,--version"
    ,   fmt::format("gnx {}\nCopyright (C) 2026 Armin Sobhani", GNX_VERSION)
    );

    CLI11_PARSE(gnx, argc, argv);
    return 0;
}