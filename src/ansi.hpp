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

namespace ansi {

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

    inline std::string black()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[30m"}
        :   std::string{};
    };
    inline std::string red()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[31m"}
        :   std::string{};
    };
    inline std::string green()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[32m"}
        :   std::string{};
    };
    inline std::string yellow()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[33m"}
        :   std::string{};
    };
    inline std::string blue()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[34m"}
        :   std::string{};
    };
    inline std::string magenta()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[35m"}
        :   std::string{};
    };
    inline std::string cyan()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[36m"}
        :   std::string{};
    };
    inline std::string white()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[37m"}
        :   std::string{};
    };
    inline std::string bright_black()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[90m"}
        :   std::string{};
    };
    inline std::string bright_red()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[91m"}
        :   std::string{};
    };
    inline std::string bright_green()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[92m"}
        :   std::string{};
    };
    inline std::string bright_yellow()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[93m"}
        :   std::string{};
    };
    inline std::string bright_blue()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[94m"}
        :   std::string{};
    };
    inline std::string bright_magenta()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[95m"}
        :   std::string{};
    };
    inline std::string bright_cyan()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[96m"}
        :   std::string{};
    };
    inline std::string bright_white()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[97m"}
        :   std::string{};
    };
    inline std::string reset()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[39m"}
        :   std::string{};
    };

}   // namespace fg

namespace bg {

    inline std::string black()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[40m"}
        :   std::string{};
    };
    inline std::string red()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[41m"}
        :   std::string{};
    };
    inline std::string green()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[42m"}
        :   std::string{};
    };
    inline std::string yellow()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[43m"}
        :   std::string{};
    };
    inline std::string blue()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[44m"}
        :   std::string{};
    };
    inline std::string magenta()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[45m"}
        :   std::string{};
    };
    inline std::string cyan()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[46m"}
        :   std::string{};
    };
    inline std::string white()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[47m"}
        :   std::string{};
    };
    inline std::string bright_black()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[100m"}
        :   std::string{};
    };
    inline std::string bright_red()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[101m"}
        :   std::string{};
    };
    inline std::string bright_green()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[102m"}
        :   std::string{};
    };
    inline std::string bright_yellow()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[103m"}
        :   std::string{};
    };
    inline std::string bright_blue()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[104m"}
        :   std::string{};
    };
    inline std::string bright_magenta()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[105m"}
        :   std::string{};
    };
    inline std::string bright_cyan()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[106m"}
        :   std::string{};
    };
    inline std::string bright_white()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[107m"}
        :   std::string{};
    };
    inline std::string reset()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[49m"}
        :   std::string{};
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

    inline std::string reset()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[0m"}
        :   std::string{};
    };
    inline std::string bold()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[1m"}
        :   std::string{};
    };
    inline std::string dim()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[2m"}
        :   std::string{};
    };
    inline std::string italic()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[3m"}
        :   std::string{};
    };
    inline std::string underline()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[4m"}
        :   std::string{};
    };
    inline std::string inverse()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[7m"}
        :   std::string{};
    };
    inline std::string strikethrough()
    {   return is_terminal() && supports_color()
        ?   std::string{"\033[9m"}
        :   std::string{};
    };

}   // namespace style

} // namespace ansi
