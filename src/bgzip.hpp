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
    bool with_index{false};
    bool keep_input{false};
    int compression_level{-1};
    int threads = 1;
};

void setup_bgzip(CLI::App& app, gnx_options& g_opt);
void run_bgzip(gnx_options& g_opt, const bgzip_options& opt);
