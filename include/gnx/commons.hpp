// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/ostream.h>

#include <iostream>
#include <source_location>

#include <gnx/lut/ansi.hpp>

namespace gnx {

void log_error
(   std::string_view message
,   const std::source_location location = std::source_location::current()
)
{   fmt::print
    (   std::cerr
    ,   "{}{}:{}{}:{}{}: {}error:{} {}\n"
    ,   ansi::ESC[fg::bright_green]
    ,   location.file_name()
    ,   ansi::ESC[fg::bright_yellow]
    ,   location.line()
    ,   ansi::ESC[fg::bright_cyan]
    ,   location.column()
    ,   ansi::ESC[fg::bright_red]
    ,   ansi::ESC[fg::reset]
    ,   message
    );
}

} // namespace gnx