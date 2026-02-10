// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
#pragma once

#include <ranges>
#include <type_traits>
#include <utility>

namespace gnx {

template <std::ranges::range InterfaceType>
class sequence_bank
{   InterfaceType _interface;
public:
    using value_type = typename InterfaceType::value_type;
    using iterator = typename InterfaceType::iterator;
    using interface_type = InterfaceType;

    // disable default constructor
    sequence_bank() = delete;
    sequence_bank(const InterfaceType& interface)
    requires std::is_copy_constructible_v<InterfaceType>
    :   _interface(interface)
    {}
    sequence_bank(InterfaceType&& interface) noexcept(std::is_nothrow_move_constructible_v<InterfaceType>)
    :   _interface(std::move(interface))
    {}
    iterator begin()
    {   return _interface.begin();
    }
    iterator end()
    {   return _interface.end();
    }
};

} // namespace gnx