// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
#pragma once

#include <ranges>
#include <type_traits>
#include <utility>

namespace gnx {

template <std::ranges::range BackendType>
class sequence_bank
{   BackendType _backend;
public:
    using value_type = typename BackendType::value_type;
    using iterator = typename BackendType::iterator;
    using backend_type = BackendType;

    // disable default constructor
    sequence_bank() = delete;
    sequence_bank(const BackendType& backend)
    requires std::is_copy_constructible_v<BackendType>
    :   _backend(backend)
    {}
    sequence_bank(BackendType&& backend) noexcept(std::is_nothrow_move_constructible_v<BackendType>)
    :   _backend(std::move(backend))
    {}
    iterator begin()
    {   return _backend.begin();
    }
    iterator end()
    {   return _backend.end();
    }
};

} // namespace gnx