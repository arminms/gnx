// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#include <vector>
#include <string>

#include <CLI/CLI.hpp>

#include <gnx/sq.hpp>

#include "ansi.hpp"

//-- get_runtime_version -----------------------------------------------------//

#if defined(__CUDACC__)
std::string get_runtime_version()
{   int version{0};
    std::string result;
    if (cudaSuccess == cudaRuntimeGetVersion(&version))
    {   int major = version / 1000;
         int minor = (version % 1000) / 10;
         int patch = version % 10;
         result = fmt::format("{}.{}.{}", major, minor, patch);
    }
    else result = "failed to get CUDA version";
    return result;
}
#endif //__CUDACC__

#if defined(__HIPCC__)
std::string get_runtime_version()
{   int version{0};
    std::string result;
    if (hipSuccess == hipRuntimeGetVersion(&version))
    {    int major = version / 10000000;
         int minor = (version % 10000000) / 100000;
         int patch = version % 100000;
         result = fmt::format("{}.{}.{}", major, minor, patch);
    }
    else result = "failed to get HIP version";
    return result;
}
#endif //__HIPCC__

/// --- main() ---------------------------------------------------------------//

int main
(   int argc
,   char** argv
)
{   std::vector<std::string> input_files;

#if defined(__CUDACC__)
    cudaDeviceProp prop;
    std::string device_name
    =   (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
    ?   prop.name
    :   ""
    ;
#elif defined(__HIPCC__)
    hipDeviceProp_t prop;
    std::string device_name
    =   (hipGetDeviceProperties(&prop, 0) == hipSuccess)
    ?   prop.name
    :   ""
    ;
#endif

    CLI::App gnx_cli
    {   fmt::format
        (   "Program : gnx "
            "(a command-line tool for biological sequence manipulation and analysis)\n"
#if defined(__CUDACC__) || defined(__HIPCC__)
            "Version : {}\nDevice   : {}"
        ,   GNX_VERSION
        ,   device_name.empty() ? "Not detected" : device_name
#else
            "Version : {}"
        ,   GNX_VERSION
#endif
        )
    };
    argv = gnx_cli.ensure_utf8(argv);

    // global configuration
    gnx_cli.require_subcommand(1);              // Force user to pick a subcommand
    gnx_cli.get_formatter()->description_paragraph_width(88); // Make help output look clean
    gnx_cli.get_formatter()->column_width(45);  // Make help output look clean
    gnx_cli.fallthrough(true);                  // allow options to be specified after subcommand
    gnx_cli.allow_windows_style_options();      // allow /option style for Windows users
    // gnx_cli.description("A command-line tool for biological sequence manipulation and analysis");
    gnx_cli.footer
    (   ansi::fg::yellow()
    +   "With no FILE, or when FILE is -, read standard input.\n"
        "Report bugs to <https://github.com/arminms/gnx/issues>."
    +   ansi::fg::reset()
    );

    // global options
    auto* version_flag = gnx_cli.set_version_flag
    (   "-v,--version"
    ,   fmt::format
#if defined(__CUDACC__)
        (   "GNX\t: {}\nCUDA\t: {}\n\nCopyright (C) 2026 Armin Sobhani"
        ,   GNX_VERSION
        ,   get_runtime_version()
        )
#elif defined(__HIPCC__)
        (   "GNX\t: {}\nROCm\t: {}\n\nCopyright (C) 2026 Armin Sobhani"
        ,   GNX_VERSION
        ,   get_runtime_version()
        )
#else
        (   "GNX\t: {}\n\nCopyright (C) 2026 Armin Sobhani"
        ,   GNX_VERSION
        )
#endif
    );
    version_flag
    ->  group("COMMON OPTIONS")
    // ->  configurable(false)
    // ->  callback_priority(CLI::CallbackPriority::First)
    ;
    int threads = 1;
    gnx_cli.add_option("-t,--threads", threads, "number of threads to use")
    ->  group("COMMON OPTIONS")
    ->  default_val("1")
    ->  check(CLI::PositiveNumber)
    ;
    // gnx_cli.add_option("input", input_files, "Input file(s) to process")
    // ->  group("COMMON OPTIONS")
    // // ->  required()
    // // ->  check(CLI::ExistingFile)
    // ->  expected(-1) // allow multiple input files
    // ;

    // --- subcommand: gnx faidx ---
    auto* faidx = gnx_cli.add_subcommand
    (   "faidx", "Index FASTA/FASTQ (optionally BGZF-compressed) file(s)"
    );
    faidx->alias("index");

    // --- subcommand: gnx count ---
    auto* count = gnx_cli.add_subcommand
    (   "count", "Count sequences in FastA/FastQ file(s)"
    )
    ->  ignore_case();
    count->description
    (   //"Count sequences in FastA/FastQ file(s)\n\n"
        //  "Usage: gnx count [OPTIONS] FILE...\n\n"
        "Counts the number of sequences, residues, and optionally GC and AT "
        "contents for each FILE, and total values if more than one FILE is "
        "specified. Supports both uncompressed and gzipped FastA and FastQ "
        "files. With no FILE, or when FILE is -, read standard input. The "
        "options below may be used to select which counts are printed, always "
        "in the following order: #seq, #res, residue counts, AT-Content and "
        "GC-Content."
    );
    count->add_option("FILE,-i,--in,--input", input_files, "input file(s) or '-' for stdin")
    ->  type_name("PATH")
    ->  allow_extra_args(true)
    // ->  check(CLI::ExistingFile)
    ;
    count->add_flag
    (   "-s,--seqs"
    ,   "print the number of sequences in each FILE and total"
    );


    try
    {   gnx_cli.parse(argc, argv);
    }
    catch (const CLI::ParseError& e)
    {   return gnx_cli.exit(e);
    }

    return 0;
}