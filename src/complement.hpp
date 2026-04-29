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
#if defined(__CUDACC__) || defined(__HIPCC__)
    bool _use_gpu{0};
#endif // __CUDACC__ || __HIPCC__

    void run_complement();
    template <typename T>
    void run_complement(std::string const& file);
};
