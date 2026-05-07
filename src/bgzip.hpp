// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include "gnx.hpp"

struct bgzip : public command
{   bgzip(CLI::App& app, gnx_options& opt);
    void run();

    // overrides
    virtual command_type type() const override;

private:
    gnx_options& _opt;
    std::vector<std::string> _input_files;
    bool _decompress;
    bool _with_index;
    bool _keep_input;
    int _compression_level;

    void run_bgzip();
    void run_bgzip(std::string const& file);
};
