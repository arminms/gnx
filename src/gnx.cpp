// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#include <cassert>

#include "gnx.hpp"
#include "bgzip.hpp"
#include "complement.hpp"
#include "prefix.hpp"

#include <gnx/backend/forward_stream.hpp>

/// --- default_analyzer -----------------------------------------------------//

struct default_analyzer : public analyzer<std::string_view>
{   void analyze(std::string_view) override {}
};

/// --- gnx_options ----------------------------------------------------------//

gnx_options::gnx_options()
:   commands()
,   input_files()
,   output_file()
,   use_stdout(false)
,   force(false)
,   time_it(false)
,   return_code(0)
,   num_procs(0)
,   threads(-1)
{   // Detect GPU availability and name (if applicable)
#if defined(__CUDACC__)
    int device_count{0}, version{0};
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0)
    {   gpu_available = true;
        cudaDeviceProp prop;
        gpu_name
        =   cudaGetDeviceProperties(&prop, 0) == cudaSuccess
        ?   prop.name
        :   "Not detected";
    }
    else gpu_available = false;
    if (cudaSuccess == cudaRuntimeGetVersion(&version))
    {   int major = version / 1000;
        int minor = (version % 1000) / 10;
        int patch = version % 10;
        runtime_version = fmt::format("{}.{}.{}", major, minor, patch);
    }
    else runtime_version = "failed to get CUDA version";
#elif defined(__HIPCC__)
    int device_count{0}, version{0} ;
    if (hipGetDeviceCount(&device_count) == hipSuccess && device_count > 0)
    {   gpu_available = true;
        hipDeviceProp_t prop;
        gpu_name
        =   hipGetDeviceProperties(&prop, 0) == hipSuccess
        ?   prop.name
        :   "Not detected";
    }
    else gpu_available = false;
    if (hipSuccess == hipRuntimeGetVersion(&version))
    {   int major = version / 10000000;
        int minor = (version % 10000000) / 100000;
        int patch = version % 100000;
        runtime_version = fmt::format("{}.{}.{}", major, minor, patch);
    }
    else runtime_version = "failed to get HIP version";
#endif //__CUDACC__ || __HIPCC__
}

void gnx_options::run()
{   if (!commands.empty())
    {   assert(input_files.back() == ":");
        input_files.pop_back(); // remove the ":" marker
        if (input_files.empty())
#if defined(__CUDACC__) || defined(__HIPCC__)
            if (use_gpu)
                run<gnx::dsq>("-");
            else
                run<gnx::sq>("-");
#else
        run<gnx::sq>("-");
#endif // __CUDACC__ || __HIPCC__
        else
        {
#if defined(__CUDACC__) || defined(__HIPCC__)
            if (use_gpu)
                // #pragma omp parallel for schedule(dynamic) if(input_files.size() > 1)
                for (std::size_t i = 0; i < input_files.size(); ++i)
                    run<gnx::dsq>(input_files[i]);
            else
                #pragma omp parallel for schedule(dynamic) if(input_files.size() > 1)
                for (std::size_t i = 0; i < input_files.size(); ++i)
                    run<gnx::sq>(input_files[i]);
#else
            #pragma omp parallel for schedule(dynamic) if(input_files.size() > 1)
            for (std::size_t i = 0; i < input_files.size(); ++i)
                run<gnx::sq>(input_files[i]);
#endif // __CUDACC__ || __HIPCC__
        }
    }
}

template <typename T>
void gnx_options::run(std::string const& filename)
{   bool is_stdin = (filename == "-");
    if (!is_stdin && std::filesystem::is_directory(filename))
    {   printerr("[gnx]: {} is a directory -- ignored\n", filename);
        return;
    }

    // cluster commands by type
    std::vector<command*>
        processors
    ,   analyzers
    ,   output_filename_modifiers
    ,   format_convertors
    ;
    for (auto* cmd : commands)
    {   if (cmd->type() >= 1 && cmd->type() < 100)
            processors.push_back(cmd);
        else if (cmd->type() == command_type::sequence_analyzer)
            analyzers.push_back(cmd);
        else if (cmd->type() == command_type::output_filename_modifier)
            output_filename_modifiers.push_back(cmd);
        else if (cmd->type() == command_type::format_convertor)
            format_convertors.push_back(cmd);
        else
        {   printerr
            (   "[gnx]: unknown command type -> {}\n"
            ,   static_cast<int>(cmd->type())
            );
            exit(1);
        }
    }

    // Determine whether input is gzipped (by extension)
    auto ext = std::filesystem::path(filename).extension().string();
    bool in_gz = (ext == ".gz" || ext == ".bgz" || ext == ".gzip");

    // Determine output path and format
    bool out_gz = false;
    bool in_place = false;
    std::string out_path;
    std::string tmp_path;

    if (!output_filename_modifiers.empty())
    {   if (use_stdout || (is_stdin && output_file.empty()))
        {   printerr
            (   "[gnx]: output filename modifiers cannot be used when writing "
                "to standard output\n"
             );
            exit(1);
        }
        else
        {   out_path = !output_file.empty() ? output_file : filename;
            for (auto* modifier : output_filename_modifiers)
                out_path = modifier->modify(out_path);
            auto oext = std::filesystem::path(out_path).extension().string();
            out_gz = (oext == ".gz" || oext == ".bgz" || oext == ".gzip");
        }
    }
    else if (is_stdin && output_file.empty())
    {   out_path = "-";
        out_gz   = false;
    }
    else if (use_stdout)
    {   out_path = "-";
        out_gz   = false;
    }
    else if (!output_file.empty())
    {   out_path = output_file;
        auto oext = std::filesystem::path(output_file).extension().string();
        out_gz = (oext == ".gz" || oext == ".bgz" || oext == ".gzip");
    }
    else
    {   // in-place: write to a temp file, then atomically replace original
        in_place = true;
        out_gz   = in_gz;
        tmp_path = filename + ".gnxtmp";
        out_path = tmp_path;
    }

    if
    (   !force
    &&  out_path != "-"
    &&  !in_place
    &&  std::filesystem::exists(out_path)
    )
    {   printerr
        (   "[gnx]: {} already exists -- "
            "ignored (use -f|--force to override)\n"
        ,   out_path
        );
        return_code = 1;
        return;
    }

    if (!analyzers.empty()  && out_path == "-")
    {   if (commands.back()->type() != command_type::sequence_analyzer)
        {   printerr
            (   "[gnx]: sequence analyzer must be the last command in the "
                "pipeline to send output to standard output\n"
            );
            exit(1);
        }
        if (analyzers.size() > 1)
        {   printerr
            (   "[gnx]: multiple sequence analyzers cannot share standard output\n"
            );
            exit(1);
        }
    }

    if (processors.empty() && analyzers.empty() && format_convertors.empty())
    {   printerr
        (   "[gnx]: no commands to execute for input file {}\n"
        ,   filename
        );
        return_code = 1;
        return;
    }

    try
    {   gnx::forward_stream<T> stream{filename};
        auto it     = stream.begin();
        auto end_it = stream.end();

        if (it == end_it)
            return; // empty file — nothing to do

        // Detect FASTA vs FASTQ from the quality field of the first record
        bool is_fastq = !stream.quality().empty();

        auto analyze = [&](auto& analyzer)
        {   for (; it != end_it; ++it)
                analyzer->analyze(it->sequence());
        };

        auto process = [&](auto& writer)
        {   writer.open(out_path);
            for (; it != end_it; ++it)
            {   auto seq = stream();
                for (auto* proc : processors)
                    proc->process(seq);
                writer.write(seq);
            }
            writer.close();
        };

        int threads_4_compression{1};
        if (out_gz)
            threads_4_compression
            =   !std::filesystem::exists(filename)
            ?   1
            :   threads == -1
                ?   std::min
                    (   int(std::log(double(std::filesystem::file_size(filename)) / 1.0e4))
                    ,   num_procs
                    )
                :   threads;

        if (!analyzers.empty())
        {   auto analyzer = analyzers.front()->get_analyzer(filename, out_path);
            analyze(analyzer);
        }
        else if (is_fastq && out_gz)
        {   gnx::out::fastq_gz w(false, threads_4_compression);
            process(w);
        }
        else if (is_fastq)
        {   gnx::out::fastq w;
            process(w);
        }
        else if (out_gz)
        {   gnx::out::fasta_gz w(false, threads_4_compression);
            process(w);
        }
        else
        {   gnx::out::fasta w;
            process(w);
        }
    }

    catch (std::exception const& e)
    {   printerr("[gnx]: {}\n", e.what());
        if (in_place && std::filesystem::exists(tmp_path))
            std::filesystem::remove(tmp_path);
        return_code = 1;
    }
}

// default implementation does nothing
void command::process(gnx::sq& s) const
{   printerr
    (   "process() not implemented for command type {}\n"
    ,   static_cast<int>(type())
    );
}

std::unique_ptr<analyzer<std::string_view>> command::get_analyzer
(   std::string_view input
,   std::string_view output
)   const
{   printerr
    (   "get_analyzer() not implemented for command type {}\n"
    ,   static_cast<int>(type())
    );
    return std::make_unique<default_analyzer>();
}

// default implementation returns the original filename
std::string command::modify(std::string_view filename) const
{   printerr
    (   "modify() not implemented for command type {}\n"
    ,   static_cast<int>(type())
    );
    return std::string(filename);
}

/// --- main() ---------------------------------------------------------------//

int main
(   int argc
,   char** argv
)
{   auto g_opt = std::make_shared<gnx_options>();
    g_opt->num_procs = omp_get_num_procs();

    CLI::App gnx_cli
    {   fmt::format
        (   "Program : gnx "
            "(a command-line tool for biological sequence manipulation and analysis)\n"
#if defined(__CUDACC__) || defined(__HIPCC__)
            "Version : {}\nDevice   : {}"
        ,   GNX_VERSION
        ,   g_opt->gpu_name
#else
            "Version : {}"
        ,   GNX_VERSION
#endif
        )
    };
    argv = gnx_cli.ensure_utf8(argv);

    // global configuration
    gnx_cli.callback([&g_opt]() { g_opt->run(); });
    gnx_cli.require_subcommand(0);         // Force user to pick a subcommand
    gnx_cli.get_formatter()->description_paragraph_width(88);    // Make help output look clean
    gnx_cli.get_formatter()->column_width(35);                   // Make help output look clean
    gnx_cli.get_formatter()->enable_option_type_names(false);   // Don't show option type names in help
    gnx_cli.fallthrough(true);             // allow App options to be specified after subcommand
    // gnx_cli.allow_windows_style_options(); // allow /option style for Windows users
    // gnx_cli.description("A command-line tool for biological sequence manipulation and analysis");
    gnx_cli.footer
    (   fmt::format
        (   "{}Report bugs to <https://github.com/arminms/gnx/issues>.{}"
        ,   ansi::style::bold()
        ,   ansi::style::reset()
        )
    );
    // gnx_cli.usage("Usage:\n\tgnx <COMMAND> [OPTIONS]\n\tgnx [OPTIONS] <FILE>... : <COMMAND>...");

    // global options
    auto* version_flag = gnx_cli.set_version_flag
    (   "-v,--version"
    ,   fmt::format
#if defined(__CUDACC__)
        (   "GNX\t: {}\nCUDA\t: {}\n\nCopyright (C) 2026 Armin Sobhani"
        ,   GNX_VERSION
        ,   g_opt->runtime_version
        )
#elif defined(__HIPCC__)
        (   "GNX\t: {}\nROCm\t: {}\n\nCopyright (C) 2026 Armin Sobhani"
        ,   GNX_VERSION
        ,   g_opt->runtime_version
        )
#else
        (   "GNX\t: {}\n\nCopyright (C) 2026 Armin Sobhani"
        ,   GNX_VERSION
        )
#endif
    );
    // version_flag
    // ->  group("COMMON OPTIONS")
    // ->  configurable(false)
    // ->  callback_priority(CLI::CallbackPriority::First)
    // ;
    gnx_cli.add_option
    (   "FILE"
    ,   g_opt->input_files
    ,   "Input FILE(s), null or '-' for stdin"
    )
    ->  type_name("PATH")
    ->  allow_extra_args(true)
    ->  default_val("-")
    ;
    gnx_cli.add_flag
    (   "-c,--stdout"
    ,   g_opt->use_stdout
    ,   "Write to standard output, keep original file(s) unchanged"
    )
    ;
    gnx_cli.add_flag
    (   "-f,--force"
    ,   g_opt->force
    ,   "Overwrite output file(s) without asking"
    )
    ;
    gnx_cli.add_option
    (   "-n,--threads"
    ,   g_opt->threads
    ,   "Number of threads to use (-1 for auto-detect)"
    )
    ->  default_val(-1)
    ->  check(CLI::Range(-1, std::numeric_limits<int>::max()))

    ;
    gnx_cli.add_option
    (   "-o,--out,--output"
    ,   g_opt->output_file
    ,   "Write to file instead of modifying FILE in-place"
    )
    ->  type_name("PATH")
    ;
    gnx_cli.add_flag
    (   "-t,--time,--time-it"
    ,   g_opt->time_it
    ,   "Time the execution of the command"
    )
    ;
#if defined(__CUDACC__) || defined(__HIPCC__)
    if (g_opt->gpu_available)
    {   gnx_cli.add_flag
        (   "-G,--gpu"
        ,   g_opt->use_gpu
        ,   fmt::format("Run on GPU ({})", g_opt->gpu_name)
        )
        ->  default_val(false)
        ;
    }
#endif // __CUDACC__ || __HIPCC__

    // create subcommands
    auto bgzip_command      = std::make_shared<bgzip>(gnx_cli, *g_opt);
    auto complement_command = std::make_shared<complement_cmd>(gnx_cli, *g_opt);
    auto prefix_command     = std::make_shared<prefix_cmd>(gnx_cli, *g_opt);

    // // --- subcommand: gnx faidx ---
    // auto* faidx = gnx_cli.add_subcommand
    // (   "faidx", "Index FASTA/FASTQ (optionally BGZF-compressed) file(s)"
    // );
    // faidx->alias("index");

    // // --- subcommand: gnx count ---
    // auto* count = gnx_cli.add_subcommand
    // (   "count", "Count sequences in FastA/FastQ file(s)"
    // )
    // ->  ignore_case();
    // count->description
    // (   //"Count sequences in FastA/FastQ file(s)\n\n"
    //     //  "Usage: gnx count [OPTIONS] FILE...\n\n"
    //     "Counts the number of sequences, residues, and optionally GC and AT "
    //     "contents for each FILE, and total values if more than one FILE is "
    //     "specified. Supports both uncompressed and gzipped FastA and FastQ "
    //     "files. With no FILE, or when FILE is -, read standard input. The "
    //     "options below may be used to select which counts are printed, always "
    //     "in the following order: #seq, #res, residue counts, AT-Content and "
    //     "GC-Content."
    // );
    // count->add_option("FILE,-i,--in,--input", input_files, "input file(s) or '-' for stdin")
    // ->  type_name("PATH")
    // ->  allow_extra_args(true)
    // // ->  check(CLI::ExistingFile)
    // ;
    // count->add_flag
    // (   "-s,--seqs"
    // ,   "print the number of sequences in each FILE and total"
    // );


    try
    {   gnx_cli.parse(argc, argv);
    }
    catch (const CLI::ParseError& e)
    {   return gnx_cli.exit(e);
    }

    return g_opt->return_code;
}