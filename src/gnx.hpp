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

/// @brief A base class for subcommands of the gnx command-line tool
struct command
{   virtual ~command() = default;
    virtual void run() = 0;
};

/// @brief A struct to hold global options and state for the gnx command-line tool
struct gnx_options
{   gnx_options()
    :   time_it(false)
    ,   return_code(0)
    ,   num_procs(0)
    {   // Detect GPU availability and name (if applicable)
#if defined(__CUDACC__)
        int device_count{0}, version{0};
        if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0)
        {   gpu_available = true;
            cudaDeviceProp prop;
            gpu_name
            =   cudaGetDeviceProperties(&prop, 0) == cudaSuccess
            ?   prop.name
            :   "Not detected";
        }
        else gpu_available = false;
        if (cudaSuccess == cudaRuntimeGetVersion(&version))
        {   int major = version / 1000;
            int minor = (version % 1000) / 10;
            int patch = version % 10;
            runtime_version = fmt::format("{}.{}.{}", major, minor, patch);
        }
        else runtime_version = "failed to get CUDA version";
#elif defined(__HIPCC__)
        int device_count{0}, version{0} ;
        if (hipGetDeviceCount(&device_count) == hipSuccess && device_count > 0)
        {   gpu_available = true;
            hipDeviceProp_t prop;
            gpu_name
            =   hipGetDeviceProperties(&prop, 0) == hipSuccess
            ?   prop.name
            :   "Not detected";
        }
        else gpu_available = false;
        if (hipSuccess == hipRuntimeGetVersion(&version))
        {   int major = version / 10000000;
            int minor = (version % 10000000) / 100000;
            int patch = version % 100000;
            runtime_version = fmt::format("{}.{}.{}", major, minor, patch);
        }
        else runtime_version = "failed to get HIP version";
#endif //__CUDACC__ || __HIPCC__
    }

    std::vector<command*> commands;
    bool time_it;
    int return_code;
    int num_procs;
#if defined(__CUDACC__) || defined(__HIPCC__)
    bool gpu_available;
    std::string gpu_name;
    std::string runtime_version;
#endif // __CUDACC__ || __HIPCC__
};
