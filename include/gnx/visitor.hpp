// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
#pragma once

#include <fmt/core.h>
#include <fmt/format.h>

#include <vector>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <functional>
#include <variant>
#include <sstream>
#include <typeindex>
#include <typeinfo>
#include <cstdint>

#include <gnx/detail/libpopcnt.h>
#include <gnx/detail/dynamic_bitset.hpp>

namespace gnx {

// -- td_value_t ---------------------------------------------------------------
// The default tagged-data value type: a closed variant covering all C++
// fundamental types plus std::string and std::vector<int>.
// std::monostate represents an unset (null) value and maps to the "void" tag.

using td_value_t = std::variant
<   std::monostate
,   bool
,   int8_t
,   int16_t
,   int32_t
,   int64_t
,   uint8_t
,   uint16_t
,   uint32_t
,   uint64_t
,   float
,   double
,   long double
,   std::string
,   std::vector<int>
,   sul::dynamic_bitset<>
>;

// -- td_type_name_map ---------------------------------------------------------

static std::unordered_map<std::type_index, std::string> td_type_name_map
{   {   std::type_index(typeid(std::monostate)), "void"
    }
,
    {   std::type_index(typeid(bool)), "bool"
    }
,
    {   std::type_index(typeid(int8_t)), "int8_t"
    }
,
    {   std::type_index(typeid(int16_t)), "int16_t"
    }
,
    {   std::type_index(typeid(int32_t)), "int32_t"
    }
,
    {   std::type_index(typeid(int64_t)), "int64_t"
    }
,
    {   std::type_index(typeid(uint8_t)), "uint8_t"
    }
,
    {   std::type_index(typeid(uint16_t)), "uint16_t"
    }
,
    {   std::type_index(typeid(uint32_t)), "uint32_t"
    }
,
    {   std::type_index(typeid(uint64_t)), "uint64_t"
    }
,
    {   std::type_index(typeid(float)),"float"
    }
,
    {   std::type_index(typeid(double)),"double"
    }
,
    {   std::type_index(typeid(long double)),"long double"
    }
,
    {   std::type_index(typeid(std::string)),"string"
    }
,
    {   std::type_index(typeid(sul::dynamic_bitset<>)),"dynamic_bitset"
    }
,
    {   std::type_index(typeid(std::vector<int>)),"std::vector<int>"
    }
};

// -- td_value_print ----------------------------------------------------------
/// Appends the type-tagged serialization of a td_value_t to @a buf.

inline void td_value_print(fmt::memory_buffer& buf, const td_value_t& v)
{   std::visit
    (   [&buf](const auto& x)
        {   using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, std::monostate>)
            {   fmt::format_to(std::back_inserter(buf), "|void|{{}}");
            }
            else if constexpr (std::is_same_v<T, std::string>)
            {   fmt::format_to(std::back_inserter(buf), "|string|\"{}\"", x);
            }
            else if constexpr (std::is_same_v<T, sul::dynamic_bitset<>>)
            {   fmt::format_to(std::back_inserter(buf), "|dynamic_bitset|{}", x.to_string());
            }
            else if constexpr (std::ranges::common_range<T>)
            {   fmt::format_to
                (   std::back_inserter(buf)
                ,   "|{}|"
                ,   td_type_name_map[std::type_index(typeid(x))]
                );
                fmt::format_to(std::back_inserter(buf), "{{");
                for (auto i : x)
                    fmt::format_to(std::back_inserter(buf), "{},", i);
                fmt::format_to(std::back_inserter(buf), "}}");
            }
            else
            {   fmt::format_to
                (   std::back_inserter(buf)
                ,   "|{}|{}"
                ,   td_type_name_map[std::type_index(typeid(x))]
                ,   static_cast<T>(x)
                );
            }
        }
    ,   v
    );
}

// -- td_scan_visitor ----------------------------------------------------------

static std::unordered_map<std::string, std::function<void(std::istream&, td_value_t&)>>
    td_scan_visitor
{   {   "void"
    ,   [] (std::istream& is, td_value_t& a)
        { is.ignore(2); a = std::monostate{}; }
    }
    ,
    {   "bool"
    ,   [] (std::istream& is, td_value_t& a)
        { bool x; is >> std::boolalpha >> x; a = x; }
    }
    ,
    {   "int8_t"
    ,   [] (std::istream& is, td_value_t& a)
        { int8_t x; is >> x; a = x; }
    }
    ,
    {   "int16_t"
    ,   [] (std::istream& is, td_value_t& a)
        { int16_t x; is >> x; a = x; }
    }
    ,
    {   "int32_t"
    ,   [] (std::istream& is, td_value_t& a)
        { int32_t x; is >> x; a = x; }
    }
    ,
    {   "int64_t"
    ,   [] (std::istream& is, td_value_t& a)
        { int64_t x; is >> x; a = x; }
    }
    ,
    {   "uint8_t"
    ,   [] (std::istream& is, td_value_t& a)
        { uint8_t x; is >> x; a = x; }
    }
    ,
    {   "uint16_t"
    ,   [] (std::istream& is, td_value_t& a)
        { uint16_t x; is >> x; a = x; }
    }
    ,
    {   "uint32_t"
    ,   [] (std::istream& is, td_value_t& a)
        { uint32_t x; is >> x; a = x; }
    }
    ,
    {   "uint64_t"
    ,   [] (std::istream& is, td_value_t& a)
        { uint64_t x; is >> x; a = x; }
    }
    ,
    {   "float"
    ,   [] (std::istream& is, td_value_t& a)
        { float x; is >> x; a = x; }
    }
    ,
    {   "double"
    ,   [] (std::istream& is, td_value_t& a)
        { double x; is >> x; a = x; }
    }
    ,
    {   "long double"
    ,   [] (std::istream& is, td_value_t& a)
        { double x; is >> x; a = static_cast<long double>(x); }
    }
    ,
    {   "dynamic_bitset"
    ,   [] (std::istream& is, td_value_t& a)
        {   sul::dynamic_bitset<> db; is >> db; a = db; }
    }
    ,
    {   "string"
    ,   [] (std::istream& is, td_value_t& a)
        { std::string s; 
          char delim;
          is >> delim; // Read opening quote
          std::getline(is, s, '"'); // Read until closing quote
          a = s; 
        }
    }
    ,
    {   "std::vector<int>"
    ,   [] (std::istream& is, td_value_t& a)
        {   std::vector<int> v;
            int i;
            is.ignore();
            while (is.peek() != '}')
            {   is >> i;
                is.ignore();
                v.push_back(i);
            }
            is.ignore();
            a = v;
        }
    }
    ,
    {   "UNREGISTERED TYPE"
    ,   [] (std::istream& is, td_value_t& a)
        { is.ignore(2); a = std::monostate{}; }
    }
    // ... add more handlers here ...
};

// -- register_td_scan_visitor ------------------------------------------------

template<class F>
inline void register_td_scan_visitor(std::string type, const F& f)
{   td_scan_visitor.insert(std::make_pair(type, f));   }

// -- td_value_size ------------------------------------------------------------
/// Returns the actual heap size (in bytes) occupied by a td_value_t value,
/// accounting for dynamic allocations inside std::string and std::vector<int>.

inline std::size_t td_value_size(const td_value_t& v)
{   return std::visit
    (   [](const auto& x) -> std::size_t
        {   using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, std::monostate>)
                return 0;
            else if constexpr (std::is_same_v<T, std::string>)
                return sizeof(std::string) + x.capacity() * sizeof(char);
            else if constexpr (std::is_same_v<T, sul::dynamic_bitset<>>)
                return sizeof(sul::dynamic_bitset<>) + (x.capacity() / 8);
            else if constexpr (std::is_same_v<T, std::vector<int>>)
                return sizeof(std::vector<int>) + x.capacity() * sizeof(int);
            else
                return sizeof(T);
        }
    ,   v
    );
}


}   // end gnx namespace
