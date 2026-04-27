// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include "gnx.hpp"

struct bgzip : public command
{   bgzip(CLI::App& app, gnx_options& opt);
    void run() override;

private:
    gnx_options& _opt;
    std::vector<std::string> _input_files;
    std::string _output_file;
    bool _use_stdout;
    bool _decompress;
    bool _force;
    bool _with_index;
    bool _keep_input;
    int _compression_level;
    int _threads;

    void run_bgzip();
    void run_bgzip(std::string const& file);
};
