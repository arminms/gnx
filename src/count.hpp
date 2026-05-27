// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include "gnx.hpp"

template <typename T>
struct count_analyzer : public analyzer<T>
{   count_analyzer
    (   std::string_view in_filename
    ,   std::string_view out_filename
    )
    :   analyzer<T>()
    ,   _in_filename(in_filename)
    ,   _out_filename(out_filename)
    {}

    virtual ~count_analyzer() = default;
    virtual void analyze(T s) override
    {   auto counts = gnx::count(s.begin(), s.end());
        for (const auto& [c, count] : counts)
            _total_counts[c] += count;
    }

private:
    std::map<char, std::size_t> _total_counts;
    std::string_view _in_filename, _out_filename;
};

struct count_cmd : public command
{   count_cmd(CLI::App& app, gnx_options& opt);
    void run();

    // overrides
    virtual command_type type() const override;
    virtual std::unique_ptr<analyzer<std::string_view>> get_analyzer
    (   std::string_view input
    ,   std::string_view output
    )   const override;

private:
    gnx_options& _opt;
    std::vector<std::string> _prefix;

    void run_count();
};
