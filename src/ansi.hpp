// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
// Implementation of the ANSI color codes for terminal output
//
#pragma once

#ifdef _WIN32
#include <io.h>
#define ISATTY _isatty
#define FILENO _fileno
#else
#include <unistd.h>
#define ISATTY isatty
#define FILENO fileno
#endif // _WIN32

#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <string>
#include <string_view>

namespace ansi {

using namespace std::string_view_literals;

inline bool is_terminal() noexcept
{   static const bool terminal = ISATTY(FILENO(stdout));
    return terminal;
}

    inline bool supports_color() noexcept
    {
#if defined(_WIN32)
        // all windows versions support colors through native console methods
        static constexpr bool result = true;
#else
        static const bool result = []
        {   const char *terms[] =
            {   "ansi"
            ,   "color"
            ,   "console"
            ,   "cygwin"
            ,   "gnome"
            ,   "konsole"
            ,   "kterm"
            ,   "linux"
            ,   "msys"
            ,   "putty"
            ,   "rxvt"
            ,   "screen"
            ,   "vt100"
            ,   "xterm"
            };

            const char *env_p = std::getenv("TERM");
            if (nullptr == env_p)
                return false;
            return std::any_of
            (   std::begin(terms)
            ,   std::end(terms)
            ,   [&](const char *term)
                {   return std::strstr(env_p, term) != nullptr;
                }
            );
        }   ();
#endif // _WIN32
        return result;
    }

namespace fg {

    inline std::string_view black()
    {   return is_terminal() && supports_color() ? "\033[30m"sv : ""sv;
    };
    inline std::string_view red()
    {   return is_terminal() && supports_color() ? "\033[31m"sv : ""sv;
    };
    inline std::string_view green()
    {   return is_terminal() && supports_color() ? "\033[32m"sv : ""sv;
    };
    inline std::string_view yellow()
    {   return is_terminal() && supports_color() ? "\033[33m"sv : ""sv;
    };
    inline std::string_view blue()
    {   return is_terminal() && supports_color() ? "\033[34m"sv : ""sv;
    };
    inline std::string_view magenta()
    {   return is_terminal() && supports_color() ? "\033[35m"sv : ""sv;
    };
    inline std::string_view cyan()
    {   return is_terminal() && supports_color() ? "\033[36m"sv : ""sv;
    };
    inline std::string_view white()
    {   return is_terminal() && supports_color() ? "\033[37m"sv : ""sv;
    };
    inline std::string_view bright_black()
    {   return is_terminal() && supports_color() ? "\033[90m"sv : ""sv;
    };
    inline std::string_view bright_red()
    {   return is_terminal() && supports_color() ? "\033[91m"sv : ""sv;
    };
    inline std::string_view bright_green()
    {   return is_terminal() && supports_color() ? "\033[92m"sv : ""sv;
    };
    inline std::string_view bright_yellow()
    {   return is_terminal() && supports_color() ? "\033[93m"sv : ""sv;
    };
    inline std::string_view bright_blue()
    {   return is_terminal() && supports_color() ? "\033[94m"sv : ""sv;
    };
    inline std::string_view bright_magenta()
    {   return is_terminal() && supports_color() ? "\033[95m"sv : ""sv;
    };
    inline std::string_view bright_cyan()
    {   return is_terminal() && supports_color() ? "\033[96m"sv : ""sv;
    };
    inline std::string_view bright_white()
    {   return is_terminal() && supports_color() ? "\033[97m"sv : ""sv;
    };
    inline std::string_view reset()
    {   return is_terminal() && supports_color() ? "\033[39m"sv : ""sv;
    };

}   // namespace fg

namespace bg {

    inline std::string_view black()
    {   return is_terminal() && supports_color() ? "\033[40m"sv : ""sv;
    };
    inline std::string_view red()
    {   return is_terminal() && supports_color() ? "\033[41m"sv : ""sv;
    };
    inline std::string_view green()
    {   return is_terminal() && supports_color() ? "\033[42m"sv : ""sv;
    };
    inline std::string_view yellow()
    {   return is_terminal() && supports_color() ? "\033[43m"sv : ""sv;
    };
    inline std::string_view blue()
    {   return is_terminal() && supports_color() ? "\033[44m"sv : ""sv;
    };
    inline std::string_view magenta()
    {   return is_terminal() && supports_color() ? "\033[45m"sv : ""sv;
    };
    inline std::string_view cyan()
    {   return is_terminal() && supports_color() ? "\033[46m"sv : ""sv;
    };
    inline std::string_view white()
    {   return is_terminal() && supports_color() ? "\033[47m"sv : ""sv;
    };
    inline std::string_view bright_black()
    {   return is_terminal() && supports_color() ? "\033[100m"sv : ""sv;
    };
    inline std::string_view bright_red()
    {   return is_terminal() && supports_color() ? "\033[101m"sv : ""sv;
    };
    inline std::string_view bright_green()
    {   return is_terminal() && supports_color() ? "\033[102m"sv : ""sv;
    };
    inline std::string_view bright_yellow()
    {   return is_terminal() && supports_color() ? "\033[103m"sv : ""sv;
    };
    inline std::string_view bright_blue()
    {   return is_terminal() && supports_color() ? "\033[104m"sv : ""sv;
    };
    inline std::string_view bright_magenta()
    {   return is_terminal() && supports_color() ? "\033[105m"sv : ""sv;
    };
    inline std::string_view bright_cyan()
    {   return is_terminal() && supports_color() ? "\033[106m"sv : ""sv;
    };
    inline std::string_view bright_white()
    {   return is_terminal() && supports_color() ? "\033[107m"sv : ""sv;
    };
    inline std::string_view reset()
    {   return is_terminal() && supports_color() ? "\033[49m"sv : ""sv;
    };

} // namespace bg

namespace vga {

    inline bool supports_vga() noexcept
    {
#if defined(_WIN32)
        // all windows versions support colors through native console methods
        static constexpr bool result = true;
#else
        static const bool result = []
        {   const char *env_p = std::getenv("TERM");
            if (nullptr == env_p)
                return false;
            return std::strstr(env_p, "-256color") != nullptr;
        }   ();
#endif // _WIN32
        return result;
    }
    inline std::string fg(uint8_t color_code)
    {   return is_terminal() && supports_vga()
        ?   fmt::format("\033[38;5;{}m", color_code)
        :   std::string{};
    };
    inline std::string bg(uint8_t color_code)
    {   return is_terminal() && supports_vga()
        ?   fmt::format("\033[48;5;{}m", color_code)
        :   std::string{};
    };

} // namespace vga

namespace rgb {

    inline bool supports_true_color() noexcept
    {
#if defined(_WIN32)
        // all windows versions support true color through native console methods
        static constexpr bool result = true;
#else
        static const bool result = []
        {   const char *env_p = std::getenv("COLORTERM");
            if (nullptr == env_p)
                return false;
            return std::strstr(env_p, "truecolor") != nullptr
            ||     std::strstr(env_p, "24bit") != nullptr;
        }   ();
#endif // _WIN32
        return result;
    }
    inline std::string fg(uint8_t r, uint8_t g, uint8_t b)
    {   return is_terminal() && supports_true_color()
        ?   fmt::format("\033[38;2;{};{};{}m", r, g, b)
        :   std::string{};
    };
    inline std::string bg(uint8_t r, uint8_t g, uint8_t b)
    {   return is_terminal() && supports_true_color()
        ?   fmt::format("\033[48;2;{};{};{}m", r, g, b)
        :   std::string{};
    };

} // namespace rgb

namespace style {

    inline std::string_view reset()
    {   return is_terminal() && supports_color() ? "\033[0m"sv : ""sv;
    };
    inline std::string_view bold()
    {   return is_terminal() && supports_color() ? "\033[1m"sv : ""sv;
    };
    inline std::string_view dim()
    {   return is_terminal() && supports_color() ? "\033[2m"sv : ""sv;
    };
    inline std::string_view italic()
    {   return is_terminal() && supports_color() ? "\033[3m"sv : ""sv;
    };
    inline std::string_view underline()
    {   return is_terminal() && supports_color() ? "\033[4m"sv : ""sv;
    };
    inline std::string_view inverse()
    {   return is_terminal() && supports_color() ? "\033[7m"sv : ""sv;
    };
    inline std::string_view strikethrough()
    {   return is_terminal() && supports_color() ? "\033[9m"sv : ""sv;
    };

}   // namespace style

} // namespace ansi
