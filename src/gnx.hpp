// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <cstdio>
#include <filesystem>
#include <cmath>

#include <CLI/CLI.hpp>
#include <CLI/Timer.hpp>
#include <fmt/core.h>

#include <omp.h>

#include <gnx/sq.hpp>

/// @brief Prints an error message to stderr
/// @tparam ...Args format string arguments
/// @param fmt format string
/// @param ...args format string arguments 
template <typename... Args>
void printerr(fmt::format_string<Args...> fmt, Args&&... args)
{   fmt::print(stderr, fmt, std::forward<Args>(args)...);
}

enum command_type
{   sequence_processor = 1,
    sequence_processor_2bit,
    sequence_processor_4bit,
    in_place_sequence_processor = 50, // can modify the sequence in-place
    sequence_analyzer = 100,
    output_filename_modifier = 200,
    format_convertor = 300,
    // add more command types here as needed
};

template <typename T>
struct analyzer
{   analyzer() = default;
    virtual ~analyzer() = default;
    virtual void analyze(T s) = 0;
};

/// @brief A base class for subcommands of the gnx command-line tool
struct command
{   virtual ~command() = default;
    virtual command_type type() const = 0;
    virtual void process(gnx::sq& s) const; // in-place sequence processor
    // virtual void process(gnx::sq& in, gnx::sq& out) const; // sequence processor that writes to a separate output sequence

    virtual std::unique_ptr<analyzer<std::string_view>> get_analyzer
    (   std::string_view input
    ,   std::string_view output
    )   const
    ;

    virtual std::string modify(std::string_view filename) const;
};

/// @brief A struct to hold global options and state for the gnx command-line tool
struct gnx_options
{   gnx_options();
    void run();

    std::vector<command*> commands;
    std::vector<std::string> input_files;
    std::string output_file;
    bool use_stdout, force, time_it;
    int return_code, num_procs, threads;
#if defined(__CUDACC__) || defined(__HIPCC__)
    bool gpu_available, use_gpu;
    std::string gpu_name, runtime_version;
#endif // __CUDACC__ || __HIPCC__

private:
    template <typename T>
    void run(std::string const& filename);
};
