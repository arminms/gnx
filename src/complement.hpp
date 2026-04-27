// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include "gnx.hpp"

struct complement_cmd : public command
{   complement_cmd(CLI::App& app, gnx_options& opt);
    void run() override;

private:
    gnx_options& _opt;
    std::vector<std::string> _input_files;
    std::string _output_file;
    bool _use_stdout;
    bool _force;

    void run_complement();
    void run_complement(std::string const& file);
};
