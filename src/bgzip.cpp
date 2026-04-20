// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
// Implementation of the bgzip subcommand for compressing FASTA/FASTQ files
//
#define BGZF_MT
#include <gnx/io/bgzf.h>

#include "bgzip.hpp"

namespace detail {

void bgzip
(   gnx_options& g_opt
,   bgzip_options const& opt
)
{   for (const auto& file : opt.input_files)
    {   gzFile in = file == "-"
        ?   gzdopen(STDIN_FILENO, "rb")
        :   gzopen(file.c_str(), "rb");
        if (!in)
        {   printerr("[bgzip] {}: no such file or directory\n", file);
            g_opt.return_code = 1;
            continue;
        }
        std::string out_file{};
        if (opt.decompress && !opt.use_stdout)
        {   // check if file has the right extension
            if (file.substr(file.length() - 3) == ".gz")
                out_file = file.substr(0, file.length() - 3);
            else if (file.substr(file.length() - 5) == ".gzip")
                out_file = file.substr(0, file.length() - 5);
            else if (file.substr(file.length() - 4) == ".tgz")
                out_file = file.substr(0, file.length() - 4) + ".tar";
            else if (file.substr(file.length() - 2) == ".z")
                out_file = file.substr(0, file.length() - 2);
            else
            {   gzclose(in);
                printerr("[bgzip] {}: unknown suffix -- ignored\n", file);
                g_opt.return_code = 1;
                continue;
            }
        }
        else if (opt.use_stdout)
        {   out_file = "standard output";
        }
        else
        {   out_file = file + ".gz";
        }
        if (!opt.force && !opt.use_stdout && std::filesystem::exists(out_file))
        {   gzclose(in);
            printerr("[bgzip] {}: already exists -- "
                "ignored (use -f|--force to override)\n", out_file);
            g_opt.return_code = 1;
            continue;
        }
        void* out
        =   opt.decompress
        ?   static_cast<void*>
            (   opt.use_stdout
            ?   stdout
            :   fopen(out_file.c_str(), "wb")
            )
        :   static_cast<void*>
            (   opt.use_stdout
            ?   bgzf_dopen(fileno(stdout), "wb")
            :   bgzf_open(out_file.c_str(), "wb")
            )
        ;
        if (!out)
        {   gzclose(in);
            printerr
            (   "[bgzip]: cannot open {}\n"
            ,   out_file
            );
            g_opt.return_code = 1;
            continue;
        }
        char buffer[16384];
        int bytes_read;
        if (opt.decompress)
        {   FILE* out_fp = static_cast<FILE*>(out);
            while ((bytes_read = gzread(in, buffer, sizeof(buffer))) > 0)
            {   if
                (   std::fwrite(buffer, 1, bytes_read, out_fp)
                !=  static_cast<size_t>(bytes_read)
                )
                {   fclose(out_fp);
                    gzclose(in);
                    printerr("[bgzip]: error writing to {}\n", out_file);
                    g_opt.return_code = 1;
                }
            }
            if (opt.use_stdout)
                std::fflush(out_fp);
            else
                fclose(out_fp);
        }
        else
        {   BGZF* out_bgzf = static_cast<BGZF*>(out);
            if (opt.threads > 1)
                bgzf_mt(out_bgzf, opt.threads, 256);
            while ((bytes_read = gzread(in, buffer, sizeof(buffer))) > 0)
            {   
                if (bgzf_write(out_bgzf, buffer, bytes_read) != bytes_read)
                {   bgzf_close(out_bgzf);
                    gzclose(in);
                    printerr("[bgzip]: error writing to {}\n", out_file);
                    g_opt.return_code = 1;
                }
            }
            bgzf_close(out_bgzf);
        }
        gzclose(in);
        if (!opt.keep_input && !opt.use_stdout)
        {   if (std::remove(file.c_str()) != 0)
            {   printerr("[bgzip] {}: error deleting input file\n", file);
                g_opt.return_code = 1;
            }
        }
    }
}

}   // namespace detail

void setup_bgzip
(   CLI::App& app
,   gnx_options& g_opt
)
{   auto opt = std::make_shared<bgzip_options>();
    auto* bgzip = app.add_subcommand
    (   "bgzip"
    ,   "Compress/decompress FILEs with bgzip (blocked gzip). "
        "By default, compress/decompress FILEs in-place."
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
(   gnx_options& g_opt
,   bgzip_options const& opt
)
{   if (g_opt.time_it)
    {   CLI::AutoTimer timer{"\nBGZIP runtime", CLI::Timer::Simple};
        detail::bgzip(g_opt, opt);
    }
    else
    {   detail::bgzip(g_opt, opt);
    }
}
