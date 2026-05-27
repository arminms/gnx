// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
// Implementation of the ANSI color codes for terminal output
//
#pragma once

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif // _WIN32

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstddef>
#include <charconv>
#include <string>
#include <algorithm>

#include <fmt/base.h>
#include <fmt/format.h>

namespace gnx::ansi {

inline std::array<std::string, 40> create_ansi_table() noexcept
{   std::array<std::string, 40> table{};
#ifdef __CLING__
    const bool jupyter = true;
#else
    const bool jupyter = std::getenv("JPY_PARENT_PID") != nullptr;
#endif
#if defined(_WIN32)
    const bool terminal = _isatty(_fileno(stdout));
    // all windows versions support colors through native console methods
    const bool supports_color = true;
#else
    const bool terminal = isatty(fileno(stdout));
    const bool supports_color = []
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

    if ((terminal || jupyter) && supports_color)
    {   table[0] = "\033[30m";  // black
        table[1] = "\033[31m";  // red
        table[2] = "\033[32m";  // green
        table[3] = "\033[33m";  // yellow
        table[4] = "\033[34m";  // blue
        table[5] = "\033[35m";  // magenta
        table[6] = "\033[36m";  // cyan
        table[7] = "\033[37m";  // white
        table[8] = "\033[90m";  // bright black (gray)
        table[9] = "\033[91m";  // bright red
        table[10] = "\033[92m"; // bright green
        table[11] = "\033[93m"; // bright yellow
        table[12] = "\033[94m"; // bright blue
        table[13] = "\033[95m"; // bright magenta
        table[14] = "\033[96m"; // bright cyan
        table[15] = "\033[97m"; // bright white

        table[16] = "\033[40m";  // background black
        table[17] = "\033[41m";  // background red
        table[18] = "\033[42m";  // background green
        table[19] = "\033[43m";  // background yellow
        table[20] = "\033[44m";  // background blue
        table[21] = "\033[45m";  // background magenta
        table[22] = "\033[46m";  // background cyan
        table[23] = "\033[47m";  // background white
        table[24] = "\033[100m"; // background bright black (gray)
        table[25] = "\033[101m"; // background bright red
        table[26] = "\033[102m"; // background bright green
        table[27] = "\033[103m"; // background bright yellow
        table[28] = "\033[104m"; // background bright blue
        table[29] = "\033[105m"; // background bright magenta
        table[30] = "\033[106m"; // background bright cyan
        table[31] = "\033[107m"; // background bright white

        table[32] = "\033[39m"; // reset foreground color
        table[33] = "\033[49m"; // reset background color

        table[34] = "\033[0m"; // reset all attributes
        table[35] = "\033[1m"; // bold
        table[36] = "\033[3m"; // italic
        table[37] = "\033[4m"; // underline
        table[38] = "\033[5m"; // blink
        table[39] = "\033[7m"; // reverse
    }

    return table;
}

thread_local static const auto ESC = create_ansi_table();

} // namespace gnx::ansi

namespace fg
{   enum forground
    {   black = 0,
        red = 1,
        green = 2,
        yellow = 3,
        blue = 4,
        magenta = 5,
        cyan = 6,
        white = 7,
        bright_black = 8,
        bright_red = 9,
        bright_green = 10,
        bright_yellow = 11,
        bright_blue = 12,
        bright_magenta = 13,
        bright_cyan = 14,
        bright_white = 15,
        reset = 32
    };
} // namespace fg

namespace bg
{   enum background
    {   black = 16,
        red = 17,
        green = 18,
        yellow = 19,
        blue = 20,
        magenta = 21,
        cyan = 22,
        white = 23,
        bright_black = 24,
        bright_red = 25,
        bright_green = 26,
        bright_yellow = 27,
        bright_blue = 28,
        bright_magenta = 29,
        bright_cyan = 30,
        bright_white = 31,
        reset = 33
    };
} // namespace bg

namespace style
{   enum text_style
    {   reset = 34,
        bold = 35,
        italic = 36,
        underline = 37,
        blink = 38,
        reverse = 39
    };
} // namespace style

namespace gnx::ansi::vga {

// std::array<std::string, 256> create_vga_fg_table() noexcept
// {   std::array<std::string, 256> table{};
// #ifdef __CLING__
//     const bool jupyter = true;
// #else
//     const bool jupyter = std::getenv("JPY_PARENT_PID") != nullptr;
// #endif
// #if defined(_WIN32)
//     const bool terminal = _isatty(_fileno(stdout));
//     // all windows versions support colors through native console methods
//     const bool supports_vga = true;
// #else
//     const bool terminal = isatty(fileno(stdout));
//     const bool supports_vga = []
//     {   const char *env_p = std::getenv("TERM");
//         if (nullptr == env_p)
//             return false;
//         return std::strstr(env_p, "-256color") != nullptr;
//     }   ();
// #endif // _WIN32

//     if ((terminal || jupyter) && supports_vga)
//     {   for (int i = 0; i < 256; ++i)
//             table[i] = fmt::format("\033[38;5;{}m", i);
//     }

//     return table;
// }

// std::array<std::string, 256> create_vga_bg_table() noexcept
// {   std::array<std::string, 256> table{};
// #ifdef __CLING__
//     const bool jupyter = true;
// #else
//     const bool jupyter = std::getenv("JPY_PARENT_PID") != nullptr;
// #endif
// #if defined(_WIN32)
//     const bool terminal = _isatty(_fileno(stdout));
//     // all windows versions support colors through native console methods
//     const bool supports_vga = true;
// #else
//     const bool terminal = isatty(fileno(stdout));
//     const bool supports_vga = []
//     {   const char *env_p = std::getenv("TERM");
//         if (nullptr == env_p)
//             return false;
//         return std::strstr(env_p, "-256color") != nullptr;
//     }   ();
// #endif // _WIN32

//     if ((terminal || jupyter) && supports_vga)
//     {   for (int i = 0; i < 256; ++i)
//             table[i] = fmt::format("\033[48;5;{}m", i);
//     }

//     return table;
// }

inline std::array<std::string, 256> vga_empty_table()
{   std::array<std::string, 256> table{}; // default to empty
    return table;
}

inline std::array<std::string, 256> vga_fg_table()
{   std::array<std::string, 256> table;
    for (int i = 0; i < 256; ++i)
        table[i] = fmt::format("\033[38;5;{}m", i);
    return table;
}

inline std::array<std::string, 256> vga_bg_table()
{   std::array<std::string, 256> table;
    for (int i = 0; i < 256; ++i)
        table[i] = fmt::format("\033[48;5;{}m", i);
    return table;
}

// // C++23 consteval versions
//
// consteval std::array<std::string_view, 256> vga_fg_map()
// {   std::array<std::string_view, 256> table;
//     const size_t buf_size = 13;
//     char buf[buf_size]{"\033[38;5;"};
//     for (size_t i = 0; i < 256; ++i)
//     {   std::to_chars_result result = std::to_chars(buf + 9, buf + buf_size, i);
//         buf[result.ptr - buf] = 'm';
//         std::string_view code(buf, result.ptr - buf + 1);
//         table[i] = code;
//     }
//     return table;
// }

// consteval std::array<std::string_view, 256> vga_bg_map()
// {   std::array<std::string_view, 256> table;
//     const size_t buf_size = 13;
//     char buf[buf_size]{"\033[38;5;"};
//     for (size_t i = 0; i < 256; ++i)
//     {   std::to_chars_result result = std::to_chars(buf + 9, buf + buf_size, i);
//         buf[result.ptr - buf] = 'm';
//         std::string_view code(buf, result.ptr - buf + 1);
//         table[i] = code;
//     }
//     return table;
// }

inline bool supports_vga_color() noexcept
{
#ifdef __CLING__
    const bool jupyter = true;
#else
    const bool jupyter = std::getenv("JPY_PARENT_PID") != nullptr;
#endif
#if defined(_WIN32)
    const bool terminal = _isatty(_fileno(stdout));
    // all windows versions support colors through native console methods
    const bool supports_vga = true;
#else
    const bool terminal = isatty(fileno(stdout));
    const bool supports_vga = []
    {   const char *env_p = std::getenv("TERM");
        if (nullptr == env_p)
            return false;
        return std::strstr(env_p, "-256color") != nullptr;
    }   ();
#endif // _WIN32
    return (terminal || jupyter) && supports_vga;
}

inline std::array<std::string, 256> create_vga_fg_table() noexcept
{   return supports_vga_color()
    ?   vga_fg_table()
    :   vga_empty_table();
}

inline std::array<std::string, 256> create_vga_bg_table() noexcept
{   return supports_vga_color()
    ?   vga_bg_table()
    :   vga_empty_table();
}

namespace fg {
    thread_local static const auto ESC = create_vga_fg_table();
} // namespace fg
namespace bg {
    thread_local static const auto ESC = create_vga_bg_table();
} // namespace bg

} // namespace gnx::ansi::vga
