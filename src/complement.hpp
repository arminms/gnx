// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include "gnx.hpp"

struct complement_cmd : public command
{   complement_cmd(CLI::App& app, gnx_options& opt);
    void run();

    // overrides
    virtual command_type type() const override;
    virtual void process(gnx::sq& s) const override;

private:
    gnx_options& _opt;
    std::vector<std::string> _input_files;
    std::size_t _line_width;
    bool _faidx;
    bool _reverse;

    void run_complement();
    template <typename T>
    void run_complement(std::string const& file);
};
