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

#include "ansi.hpp"

/// @brief Prints an error message to stderr
/// @tparam ...Args format string arguments
/// @param fmt format string
/// @param ...args format string arguments 
template <typename... Args>
void printerr(fmt::format_string<Args...> fmt, Args&&... args)
{   fmt::print(stderr, fmt, std::forward<Args>(args)...);
}

/// @brief A struct to hold global options and state for the gnx command-line tool
struct gnx_options
{   bool time_it{false};
    int return_code{0};
    int num_procs{0};
};
