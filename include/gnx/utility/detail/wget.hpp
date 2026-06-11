// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <gnx/utility/detail/knetfile.h>

#include <fmt/base.h>
#include <fmt/format.h>

#include <string_view>
#include <filesystem>
#include <stdexcept>

namespace gnx {

struct wget_result
{   wget_result() = default;
    wget_result(wget_result const&) = delete;
    wget_result& operator=(wget_result const&) = delete;
    wget_result(wget_result&&) = default;
    wget_result& operator=(wget_result&&) = default;
    wget_result(std::filesystem::path const& temp_file_path)
    :   temp_file_path(temp_file_path)
    {}
    ~wget_result()
    {   if (std::filesystem::exists(temp_file_path))
            std::filesystem::remove(temp_file_path);
    }

// private:
    std::filesystem::path temp_file_path;
};

} // namespace gnx