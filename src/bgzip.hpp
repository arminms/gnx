// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include "gnx.hpp"

struct bgzip_options
{   std::vector<std::string> input_files;
    std::string output_file;
    bool use_stdout{false};
    bool decompress{false};
    bool force{false};
    bool with_index{true};
    bool keep_input{false};
    int compression_level{-1};
    int threads = 1;
    // std::size_t line_width;
};

void setup_bgzip(CLI::App& app, gnx_options const& g_opt);
void run_bgzip(gnx_options const& g_opt, const bgzip_options& opt);
