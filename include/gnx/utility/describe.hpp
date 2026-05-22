// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <gnx/utility/detail/describe.hpp>

namespace gnx {

inline void describe(std::string_view filename)
{   if (std::filesystem::exists(std::string(filename) + ".fai"))
        detail::describe_vv(filename);
    else
        detail::describe_fs(filename);
}

}   // namespace gnx

