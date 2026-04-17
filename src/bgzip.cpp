// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
// Implementation of the bgzip subcommand for compressing FASTA/FASTQ files
//
#include "bgzip.hpp"

namespace detail {

void bgzip
(   gnx_options const& g_opt
,   bgzip_options const& opt
)
{   std::cout << "Working on file(s):\n";
    for(const auto& file : opt.input_files)
        std::cout << file << "\n";
}

}   // namespace detail

void setup_bgzip
(   CLI::App& app
,   gnx_options const& g_opt
)
{   auto opt = std::make_shared<bgzip_options>();
    auto* bgzip = app.add_subcommand
    (   "bgzip"
    ,   "Compress/decompress FASTA/FASTQ files with bgzip (blocked gzip)"
        // "(by default, compress/decompress FILEs in-place)"
    );
    bgzip
    ->  alias("compress")
    ->  alias("decompress")
    ->  footer
    (   fmt::format
        (   "{}With no FILE, or when FILE is -, read standard input.\n\n"
            "Report bugs to <https://github.com/arminms/gnx/issues>.{}"
        ,   ansi::style::bold()
        ,   ansi::style::reset()
        )
    );

    // options and flags for the bgzip subcommand
    bgzip->add_option
    (   "FILE"
    ,   opt->input_files
    ,   "input FILEs, null or '-' for stdin"
    )
    ->  type_name("PATH")
    ->  allow_extra_args(true)
    ->  default_val("-")
    ;
    bgzip->add_flag
    (   "-c,--stdout"
    ,   opt->use_stdout
    ,   "Write to standard output, keep original file unchanged"
    )
    ;
    bgzip->add_flag
    (   "-d,--decompress"
    ,   opt->decompress
    ,   "Decompress input file(s) instead of compressing"
    )
    ;
    bgzip->add_flag
    (   "-f,--force"
    ,   opt->force
    ,   "Overwrite files without asking"
    )
    ;
    bgzip->add_flag
    (   "-i,--index"
    ,   opt->with_index
    ,   "Compress and create .fai and .gzi indeces"
    )
    ;
    bgzip->add_flag
    (   "-k,--keep"
    ,   opt->keep_input
    ,   "Don't delete input file(s) after compression/decompression"
    )
    ;
    bgzip->add_option
    (   "-l,--level"
    ,   opt->compression_level
    ,   "Compression level (0-9, -1 for default)"
    )
    ->  default_val("-1")
    ->  check(CLI::Range(0, 9))
    ;
    bgzip->add_option
    (   "-n,--threads"
    ,   opt->threads
    ,   "Number of compression threads to use"
    )
    ->  default_val("1")
    ->  check(CLI::PositiveNumber)
    ;
    bgzip->add_option
    (   "-o,--out,--output"
    ,   opt->output_file
    ,   "Write to file, keep original file unchanged"
    )
    ->  type_name("PATH")
    ;

    // set the callback for the bgzip subcommand
    bgzip->callback([&g_opt, opt]() { run_bgzip(g_opt, *opt); });
}

void run_bgzip
(   gnx_options const& g_opt
,   bgzip_options const& opt
)
{   if (g_opt.time_it)
    {   CLI::AutoTimer timer{"\nBGZIP runtime", CLI::Timer::Simple};
        detail::bgzip(g_opt, opt);
    }
    else
        detail::bgzip(g_opt, opt);
}
