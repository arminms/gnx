// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
// Implementation of the complement subcommand for the gnx CLI tool
//
#include "prefix.hpp"

prefix_cmd::prefix_cmd
(   CLI::App& app
,   gnx_options& opt
)
:  _opt(opt)
,  _prefix()
{   auto* sub = app.add_subcommand
    (   "prefix"
    ,   "Add a prefix to the output filename for each input FILE. Can only be "
        "used with internal pipeline mode, allowing you to keep the original "
        "files unchanged."
    );
    sub
    ->  footer
    (   fmt::format
        (   "{}With no FILE, or when FILE is -, read standard input and write "
            "to standard output if no output file specified.\n\n"
            "Report bugs to <https://github.com/arminms/gnx/issues>.{}"
        ,   ansi::style::bold()
        ,   ansi::style::reset()
        )
    )
    ->  group("OUTPUT FILENAME MODIFIERS")
    ;
    sub->add_option
    (   "p,--prefix"
    ,   _prefix
    ,   "Prefix to add to the output filename (ignored if writing to stdout)"
    )
    ->  allow_extra_args(true)
    ;
    sub->callback([this]() { run(); });
}

void prefix_cmd::run()
{   //fmt::print("[prefix]: prefixes {}\n", fmt::join(_prefix, ", "));

    if (!_prefix.empty() && _prefix.back() == ":")
        _prefix.pop_back(); // remove trailing ':' if present

    if (_prefix.empty())
    {   printerr
        (   "[prefix]: no prefix specified, nothing to do\n"
        );
        _opt.return_code = 1;
        return;
    }
    else if (_opt.input_files.back() == ":")
    {   if (_prefix.size() > 1)
            printerr
            (   "[prefix]: multiple prefixes specified, only the first one "
                "will be used\n"
            );
        _opt.commands.push_back(this);
        return;
    }
    else
    {   printerr
        (   "[prefix]: only work in internal pipeline mode (with ':' as the "
            "pipeline operator, separating commands)\n"
        );
        _opt.return_code = 1;
        return;
    }
}

command_type prefix_cmd::type() const
{   return command_type::output_filename_modifier;
}

std::string prefix_cmd::modify(std::string_view filename) const
{   if (filename == "-" || filename.empty())
    {   printerr
        (   "[prefix]: cannot add prefix to standard input or empty filename "
            "-- ignored\n"
        );
        _opt.return_code = 1;
        return std::string(filename);
    }
    return fmt::format
    (   "{}{}"
    ,   _prefix.empty() ? "" : _prefix.front()
    ,   std::filesystem::path(filename).filename().string()
    );
}
