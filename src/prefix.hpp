// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include "gnx.hpp"

struct prefix_cmd : public command
{   prefix_cmd(CLI::App& app, gnx_options& opt);
    void run();

    // overrides
    virtual command_type type() const override;
    virtual std::string modify(std::string_view filename) const override;

private:
    gnx_options& _opt;
    std::vector<std::string> _prefix;

    void run_prefix();
};
