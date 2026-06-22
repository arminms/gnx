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
#include <typeindex>
#include <functional>
#include <variant>
#include <sstream>

namespace gnx {

// -- td_value_t ---------------------------------------------------------------
// The default tagged-data value type: a closed variant covering all C++
// fundamental types plus std::string and std::vector<int>.
// std::monostate represents an unset (null) value and maps to the "void" tag.

using td_value_t = std::variant
<   std::monostate
,   bool
,   char
,   signed char
,   unsigned char
,   short
,   unsigned short
,   int
,   unsigned int
,   long
,   unsigned long
,   long long
,   unsigned long long
,   float
,   double
,   long double
,   std::string
,   std::vector<int>
>;

// -- quote_with_delimiter -----------------------------------------------------

inline void quote_with_delimiter(fmt::memory_buffer& buf, std::string_view str, char delimiter = '|')
{
    fmt::format_to(std::back_inserter(buf), "{}{}{}", delimiter, str, delimiter);
}

// -- make_td_print_visitor() --------------------------------------------------

template<class T, class F>
inline
std::pair<const std::type_index, std::function<void(fmt::memory_buffer&, const td_value_t&)>>
    make_td_print_visitor(const F& f)
{   return
    {   std::type_index(typeid(T)),
        [g = f](fmt::memory_buffer& buf, const td_value_t& a)
        {   if constexpr (std::is_same_v<T, std::monostate>) g(buf);
            else g(buf, std::get<T>(a));
        }
    };
}

// -- td_print_visitor ---------------------------------------------------------

static std::unordered_map<std::type_index, std::function<void(fmt::memory_buffer&, const td_value_t&)>>
    td_print_visitor
{   make_td_print_visitor<std::monostate>
    (   [] (fmt::memory_buffer& buf)
        { quote_with_delimiter(buf, "void"); fmt::format_to(std::back_inserter(buf), "{{}}"); }
    )
,   make_td_print_visitor<bool>
    (   [](fmt::memory_buffer& buf, bool x)
        { quote_with_delimiter(buf, "bool"); fmt::format_to(std::back_inserter(buf), "{}", x); }
    )
,   make_td_print_visitor<char>
    (   [](fmt::memory_buffer& buf, char x)
        { quote_with_delimiter(buf, "char"); fmt::format_to(std::back_inserter(buf), "{}", static_cast<int>(x)); }
    )
,   make_td_print_visitor<signed char>
    (   [](fmt::memory_buffer& buf, signed char x)
        { quote_with_delimiter(buf, "signed char"); fmt::format_to(std::back_inserter(buf), "{}", static_cast<int>(x)); }
    )
,   make_td_print_visitor<unsigned char>
    (   [](fmt::memory_buffer& buf, unsigned char x)
        { quote_with_delimiter(buf, "unsigned char"); fmt::format_to(std::back_inserter(buf), "{}", static_cast<unsigned>(x)); }
    )
,   make_td_print_visitor<short>
    (   [](fmt::memory_buffer& buf, short x)
        { quote_with_delimiter(buf, "short"); fmt::format_to(std::back_inserter(buf), "{}", x); }
    )
,   make_td_print_visitor<unsigned short>
    (   [](fmt::memory_buffer& buf, unsigned short x)
        { quote_with_delimiter(buf, "unsigned short"); fmt::format_to(std::back_inserter(buf), "{}", x); }
    )
,   make_td_print_visitor<int>
    (   [](fmt::memory_buffer& buf, int x)
        { quote_with_delimiter(buf, "int"); fmt::format_to(std::back_inserter(buf), "{}", x); }
    )
,   make_td_print_visitor<unsigned int>
    (   [](fmt::memory_buffer& buf, unsigned int x)
        { quote_with_delimiter(buf, "unsigned"); fmt::format_to(std::back_inserter(buf), "{}", x); }
    )
,   make_td_print_visitor<long>
    (   [](fmt::memory_buffer& buf, long x)
        { quote_with_delimiter(buf, "long"); fmt::format_to(std::back_inserter(buf), "{}", x); }
    )
,   make_td_print_visitor<unsigned long>
    (   [](fmt::memory_buffer& buf, unsigned long x)
        { quote_with_delimiter(buf, "unsigned long"); fmt::format_to(std::back_inserter(buf), "{}", x); }
    )
,   make_td_print_visitor<long long>
    (   [](fmt::memory_buffer& buf, long long x)
        { quote_with_delimiter(buf, "long long"); fmt::format_to(std::back_inserter(buf), "{}", x); }
    )
,   make_td_print_visitor<unsigned long long>
    (   [](fmt::memory_buffer& buf, unsigned long long x)
        { quote_with_delimiter(buf, "unsigned long long"); fmt::format_to(std::back_inserter(buf), "{}", x); }
    )
,   make_td_print_visitor<float>
    (   [](fmt::memory_buffer& buf, float x)
        { quote_with_delimiter(buf, "float"); fmt::format_to(std::back_inserter(buf), "{}", x); }
    )
,   make_td_print_visitor<double>
    (   [](fmt::memory_buffer& buf, double x)
        { quote_with_delimiter(buf, "double"); fmt::format_to(std::back_inserter(buf), "{}", x); }
    )
,   make_td_print_visitor<long double>
    (   [](fmt::memory_buffer& buf, long double x)
        { quote_with_delimiter(buf, "long double"); fmt::format_to(std::back_inserter(buf), "{}", static_cast<double>(x)); }
    )
,   make_td_print_visitor<std::string>
    (   [] (fmt::memory_buffer& buf, const std::string& s)
        { quote_with_delimiter(buf, "string"); fmt::format_to(std::back_inserter(buf), "\"{}\"", s); }
    )
,   make_td_print_visitor<std::vector<int>>
    (   [] (fmt::memory_buffer& buf, const std::vector<int>& v)
        {   quote_with_delimiter(buf, "std::vector<int>"); 
            fmt::format_to(std::back_inserter(buf), "{{");
            for (auto i : v)
                fmt::format_to(std::back_inserter(buf), "{},", i);
            fmt::format_to(std::back_inserter(buf), "}}");
        }
    )
    // ... add more handlers here ...
};

// -- register_td_print_visitor ------------------------------------------------

template<class T, class F>
inline void register_td_print_visitor(const F& f)
{   td_print_visitor.insert(make_td_print_visitor<T>(f));   }

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
    {   "char"
    ,   [] (std::istream& is, td_value_t& a)
        { int x; is >> x; a = static_cast<char>(x); }
    }
    ,
    {   "signed char"
    ,   [] (std::istream& is, td_value_t& a)
        { int x; is >> x; a = static_cast<signed char>(x); }
    }
    ,
    {   "unsigned char"
    ,   [] (std::istream& is, td_value_t& a)
        { unsigned x; is >> x; a = static_cast<unsigned char>(x); }
    }
    ,
    {   "short"
    ,   [] (std::istream& is, td_value_t& a)
        { short x; is >> x; a = x; }
    }
    ,
    {   "unsigned short"
    ,   [] (std::istream& is, td_value_t& a)
        { unsigned short x; is >> x; a = x; }
    }
    ,
    {   "int"
    ,   [] (std::istream& is, td_value_t& a)
        { int x; is >> x; a = x; }
    }
    ,
    {   "unsigned"
    ,   [] (std::istream& is, td_value_t& a)
        { unsigned int x; is >> x; a = x; }
    }
    ,
    {   "long"
    ,   [] (std::istream& is, td_value_t& a)
        { long x; is >> x; a = x; }
    }
    ,
    {   "unsigned long"
    ,   [] (std::istream& is, td_value_t& a)
        { unsigned long x; is >> x; a = x; }
    }
    ,
    {   "long long"
    ,   [] (std::istream& is, td_value_t& a)
        { long long x; is >> x; a = x; }
    }
    ,
    {   "unsigned long long"
    ,   [] (std::istream& is, td_value_t& a)
        { unsigned long long x; is >> x; a = x; }
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
            else if constexpr (std::is_same_v<T, std::vector<int>>)
                return sizeof(std::vector<int>) + x.capacity() * sizeof(int);
            else
                return sizeof(T);
        }
    ,   v
    );
}


}   // end gnx namespace
