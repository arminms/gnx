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
#include <any>
#include <sstream>

namespace gnx {

// -- quote_with_delimiter -----------------------------------------------------

inline void quote_with_delimiter(fmt::memory_buffer& buf, std::string_view str, char delimiter = '|')
{
    fmt::format_to(std::back_inserter(buf), "{}{}{}", delimiter, str, delimiter);
}

// -- make_td_print_visitor() --------------------------------------------------

template<class T, class F>
inline
std::pair<const std::type_index, std::function<void(fmt::memory_buffer&, const std::any&)>>
    make_td_print_visitor(const F& f)
{   return
    {   std::type_index(typeid(T)),
        [g = f](fmt::memory_buffer& buf, const std::any& a)
        {   if constexpr (std::is_void_v<T>) g(buf);
            else g(buf, std::any_cast<const T&>(a));
        }
    };
}

// -- td_print_visitor ---------------------------------------------------------

static std::unordered_map<std::type_index, std::function<void(fmt::memory_buffer&, const std::any&)>>
    td_print_visitor
{   make_td_print_visitor<void>
    (   [] (fmt::memory_buffer& buf)
        { quote_with_delimiter(buf, "void"); fmt::format_to(std::back_inserter(buf), "{{}}"); }
    )
,   make_td_print_visitor<bool>
    (   [](fmt::memory_buffer& buf, bool x)
        { quote_with_delimiter(buf, "bool"); fmt::format_to(std::back_inserter(buf), "{}", x); }
    )
,   make_td_print_visitor<int>
    (   [](fmt::memory_buffer& buf, int x)
        { quote_with_delimiter(buf, "int"); fmt::format_to(std::back_inserter(buf), "{}", x); }
    )
,   make_td_print_visitor<unsigned>
    (   [](fmt::memory_buffer& buf, unsigned x)
        { quote_with_delimiter(buf, "unsigned"); fmt::format_to(std::back_inserter(buf), "{}", x); }
    )
,   make_td_print_visitor<float>
    (   [](fmt::memory_buffer& buf, float x)
        { quote_with_delimiter(buf, "float"); fmt::format_to(std::back_inserter(buf), "{}", x); }
    )
,   make_td_print_visitor<double>
    (   [](fmt::memory_buffer& buf, double x)
        { quote_with_delimiter(buf, "double"); fmt::format_to(std::back_inserter(buf), "{}", x); }
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

static std::unordered_map<std::string, std::function<void(std::istream&, std::any&)>>
    td_scan_visitor
{   {   "void"
    ,   [] (std::istream& is, std::any& a)
        { is.ignore(2) ; a = {}; }
    }
    ,
    {   "bool"
    ,   [] (std::istream& is, std::any& a)
        { bool x; is >> std::boolalpha >> x; a = x; }
    }
    ,
    {   "int"
    ,   [] (std::istream& is, std::any& a)
        { int x; is >> x; a = x; }
    }
    ,
    {   "unsigned"
    ,   [] (std::istream& is, std::any& a)
        { unsigned x; is >> x; a = x; }
    }
    ,
    {   "float"
    ,   [] (std::istream& is, std::any& a)
        { float x; is >> x; a = x; }
    }
    ,
    {   "double"
    ,   [] (std::istream& is, std::any& a)
        { double x; is >> x; a = x; }
    }
    ,
    {   "string"
    ,   [] (std::istream& is, std::any& a)
        { std::string s; 
          char delim;
          is >> delim; // Read opening quote
          std::getline(is, s, '"'); // Read until closing quote
          a = s; 
        }
    }
    ,
    {   "std::vector<int>"
    ,   [] (std::istream& is, std::any& a)
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
    ,   [] (std::istream& is, std::any& a)
        { is.ignore(2) ; a = {}; }
    }
    // ... add more handlers here ...
};

// -- register_td_scan_visitor ------------------------------------------------

template<class F>
inline void register_td_scan_visitor(std::string type, const F& f)
{   td_scan_visitor.insert(std::make_pair(type, f));   }


}   // end gnx namespace
