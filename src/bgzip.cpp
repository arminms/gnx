// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
// Implementation of the bgzip subcommand for compressing FASTA/FASTQ files
//
#include <gnx/sq.hpp>
#include <gnx/sqb.hpp>
#include <gnx/backend/forward_stream.hpp>
#include <gnx/io/fastaqz.hpp>
#include <gnx/utility/create_index.hpp>

#include "bgzip.hpp"

bgzip::bgzip
(   CLI::App& app
,   gnx_options& opt
)
:  _opt(opt)
,  _input_files()
,  _output_file()
,  _use_stdout(false)
,  _decompress(false)
,  _force(false)
,  _with_index(false)
,  _keep_input(false)
,  _compression_level(-1)
,  _threads(1)
{   auto* bgzip = app.add_subcommand
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
    ,   _input_files
    ,   "input FILEs, null or '-' for stdin"
    )
    ->  type_name("PATH")
    ->  allow_extra_args(true)
    ->  default_val("-")
    ;
    bgzip->add_flag
    (   "-c,--stdout"
    ,   _use_stdout
    ,   "Write to standard output, keep original file unchanged"
    )
    ;
    bgzip->add_flag
    (   "-d,--decompress"
    ,   _decompress
    ,   "Decompress input file(s) instead of compressing"
    )
    ;
    bgzip->add_flag
    (   "-f,--force"
    ,   _force
    ,   "Overwrite files without asking"
    )
    ;
    bgzip->add_flag
    (   "-i,--index"
    ,   _with_index
    ,   "Compress and create .fai and .gzi indeces"
    )
    ;
    bgzip->add_flag
    (   "-k,--keep"
    ,   _keep_input
    ,   "Don't delete input file(s) after compression/decompression"
    )
    ;
    bgzip->add_option
    (   "-l,--level"
    ,   _compression_level
    ,   "Compression level (0-9, -1 for default)"
    )
    ->  default_val("-1")
    ->  check(CLI::Range(-1, 9))
    ;
    bgzip->add_option
    (   "-n,--threads"
    ,   _threads
    ,   "Number of compression threads to use (-1 for auto-detect)"
    )
    ->  default_val("-1")
    ->  check(CLI::Range(-1, std::numeric_limits<int>::max()))
    ;
    bgzip->add_option
    (   "-o,--out,--output"
    ,   _output_file
    ,   "Write to file, keep original file unchanged"
    )
    ->  type_name("PATH")
    ;

    bgzip->callback([this]() { run(); });
}

void bgzip::run()
{   if (_opt.time_it)
    {   CLI::AutoTimer timer{"\nBGZIP runtime", CLI::Timer::Simple};
        run_bgzip();
    }
    else
    {   run_bgzip();
    }
}

void bgzip::run_bgzip()
{   if (_input_files.empty())
        run_bgzip("-");
    else
    {   if (!_output_file.empty() && _input_files.size() > 1)
        {   printerr
            (   "[bgzip]: output file {}{}{} cannot be specified for multiple "
                "input files\n"
            ,   ansi::fg::yellow()
            ,   _output_file
            ,   ansi::fg::reset()
            );
            _opt.return_code = 1;
            return;
        }
        // for (auto const& file : _opt.input_files)
        //     bgzip(file, _opt, *this);
        #pragma omp parallel for schedule(dynamic) if(_input_files.size() > 1)
        for (std::size_t i = 0; i < _input_files.size(); ++i)
            run_bgzip(_input_files[i]);
    }
}

void bgzip::run_bgzip(std::string const& file)
{   // make a local copy since we will be modifying it
    bool keep_input = _keep_input || _use_stdout || file == "-";

    // input file checkings...
    if (std::filesystem::is_directory(file))
    {   printerr("[bgzip]: {} is a directory -- ignored\n", file);
        return;
    }
    auto ext = std::filesystem::path(file).extension().string();
    if
    (   !_decompress
    &&  (  ext == ".bgz"
        || ext == ".gz"
        || ext == ".gzip"
        || ext == ".tgz"
        || ext == ".z"
        || ext == ".zip"
        )
    )
    {   printerr("[bgzip]: {} already has a compressed suffix -- ignored\n", file);
        return;
    }
    // output file checkings...
    std::string out_file{};
    if (_decompress && !_use_stdout && file != "-")
    {   if
        (   ext == ".gz"
        ||  ext == ".bgz"
        ||  ext == ".gzip"
        ||  ext == ".zip"
        ||  ext == ".z"
        )
            out_file = file.substr(0, file.size() - ext.size());
        else if (ext == ".tgz")
            out_file = file.substr(0, file.size() - ext.size()) + ".tar";
        else
        {   printerr("[bgzip]: {} has unknown suffix -- ignored\n", file);
            _opt.return_code = 1;
            return;
        }
    }
    else if (_use_stdout)
    {   out_file = "standard output";
    }
    else if (!(_output_file).empty())
    {   out_file = _output_file;
        keep_input = true;
    }
    else
    {   out_file = file + ".gz";
    }
    if (!(_force) && !(_use_stdout) && std::filesystem::exists(out_file))
    {   printerr("[bgzip]: {} already exists -- "
            "ignored (use -f|--force to override)\n", out_file);
        _opt.return_code = 1;
        return;
    }

    if (_decompress && _with_index)
        printerr("[bgzip]: ignored -i|--index option for decompression\n");

    if
    (   _with_index
    &&  (   ext == ".fa"
        ||  ext == ".fas"
        ||  ext == ".fasta"
        ||  ext == ".fna"
        ||  ext == ".faa"
        ||  ext == ".ffn"
        )
    )
    {   try
        {   gnx::sequence_bank sb{gnx::forward_stream<gnx::sq>{file}};
            gnx::out::fasta_gz out(true);
            out.open(out_file);
            for (const auto& s : sb)
            {   if (s().empty())
                {   printerr("[bgzip]: {} -- not a valid FASTA file\n", file);
                    _opt.return_code = 1;
                    out.close();
                    std::remove(out_file.c_str());
                    return;
                }
                out.write(s());
            }
            out.close();
        }
        catch (const std::runtime_error& e)
        {   printerr("[bgzip]: {}\n", e.what());
            _opt.return_code = 1;
            std::remove(out_file.c_str());
            return;
        }
    }
    else if
    (   _with_index
    &&  (   ext == ".fastq"
        ||  ext == ".fq"
        )
    )
    {   try
        {   gnx::sequence_bank sb{gnx::forward_stream<gnx::sq>{file}};
            gnx::out::fastq_gz out(true);
            out.open(out_file);
            for (const auto& s : sb)
            {   if (s.quality().empty())
                {   printerr("[bgzip]: {} -- not a valid FASTQ file\n", file);
                    _opt.return_code = 1;
                    out.close();
                    std::remove(out_file.c_str());
                    return;
                }
                out.write(s());
            }
            out.close();
        }
        catch (const std::runtime_error& e)
        {   printerr("[bgzip]: {}\n", e.what());
            _opt.return_code = 1;
            std::remove(out_file.c_str());
            return;
        }
    }
    else
    {   gzFile in = file == "-"
        ?   gzdopen(STDIN_FILENO, "rb")
        :   gzopen(file.c_str(), "rb");
        if (!in)
        {   printerr("[bgzip]: {} -- no such file or directory\n", file);
        }
        void* out
        =   _decompress
        ?   static_cast<void*>
            (   _use_stdout
            ?   stdout
            :   fopen(out_file.c_str(), "wb")
            )
        :   static_cast<void*>
            (   _use_stdout
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
            _opt.return_code = 1;
            return;
        }
        char buffer[16384];
        int bytes_read;
        if (_decompress)
        {   FILE* out_fp = static_cast<FILE*>(out);
            while ((bytes_read = gzread(in, buffer, sizeof(buffer))) > 0)
            {   if
                (   std::fwrite(buffer, 1, bytes_read, out_fp)
                !=  static_cast<size_t>(bytes_read)
                )
                {   fclose(out_fp);
                    gzclose(in);
                    printerr("[bgzip]: error writing to {}\n", out_file);
                    _opt.return_code = 1;
                }
            }
            if (_use_stdout)
                std::fflush(out_fp);
            else
                fclose(out_fp);
        }
        else
        {   BGZF* out_bgzf = static_cast<BGZF*>(out);
            int threads_to_use
            =   !std::filesystem::exists(file)
            ?   1
            :   _threads == -1
                ?   std::min
                    (   int(std::log(double(std::filesystem::file_size(file)) / 1.0e4))
                    ,   _opt.num_procs
                    )
                :   _threads;
            if (threads_to_use > 1)
                bgzf_mt(out_bgzf, threads_to_use, 256);
            // fmt::print
            // (   stderr
            // ,   "[bgzip]: compressing {} with {} thread(s)\n"
            // ,   file
            // ,   threads_to_use
            // );
            while ((bytes_read = gzread(in, buffer, sizeof(buffer))) > 0)
            {   
                if (bgzf_write(out_bgzf, buffer, bytes_read) != bytes_read)
                {   bgzf_close(out_bgzf);
                    gzclose(in);
                    printerr("[bgzip]: error writing to {}\n", out_file);
                    _opt.return_code = 1;
                }
            }
            bgzf_close(out_bgzf);
        }
        gzclose(in);
        if (_with_index && !_decompress)
        {   std::string index_file = out_file + ".gzi";
            try
            {   gnx::create_gzi(out_file, index_file);
            }
            catch (const std::runtime_error& e)
            {   printerr("[bgzip]: {}\n", e.what());
                _opt.return_code = 1;
                std::remove(index_file.c_str());
            }
        }
    }

    if (!keep_input)
    {   if (std::remove(file.c_str()) != 0)
        {   printerr("[bgzip]: {} -- error deleting input file\n", file);
            _opt.return_code = 1;
        }
    }
}
