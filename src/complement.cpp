// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
// Implementation of the complement subcommand for the gnx CLI tool
//
#include "complement.hpp"

#include <gnx/sq.hpp>
#include <gnx/io/fastaqz.hpp>
#include <gnx/backend/forward_stream.hpp>
#include <gnx/algorithms/complement.hpp>

complement_cmd::complement_cmd
(   CLI::App& app
,   gnx_options& opt
)
:  _opt(opt)
,  _input_files()
{   auto* sub = app.add_subcommand
    (   "complement"
    ,   "Complement all sequence(s) in FASTA/FASTQ FILE(s) in-place, "
        "preserving the original format (plain or gzipped)."
    );
    sub
    ->  footer
    (   fmt::format
        (   "{}With no FILE, or when FILE is -, read standard input and write "
            "to standard output if no output file specified.\n\n"
            "Report bugs to <https://github.com/arminms/gnx/issues>.{}"
        ,   gnx::ansi::ESC[style::bold]
        ,   gnx::ansi::ESC[style::reset]
        )
    )
    ->  group("ALGORITHMS")
    ;
    sub->add_option
    (   "INPUT"
    ,   _input_files
    ,   "input FILE(s), null or '-' for stdin"
    )
    ->  type_name("PATH")
    ->  allow_extra_args(true)
    ->  default_val("-")
    ->  group("")
    ;
    sub->add_flag
    (   "-i,--faidx"
    ,   _faidx
    ,   "Create .fai and .gzi (if gzipped) index file(s) for the output (ignored if writing to stdout)"
    )
    ->  default_val(false)
    ;
    sub->add_flag
    (   "-r,--reverse"
    ,   _reverse
    ,   "Reverse the sequence(s) in addition to complementing (ignored for FASTQ)"
    )
    ->  default_val(false)
    ;
    sub->add_option
    (   "-w,--width"
    ,   _line_width
    ,   "Line width for FASTA output (ignored for FASTQ)"
    )
    ->  default_val(80)
    ;

    sub->callback([this]() { run(); });
}

void complement_cmd::run()
{   if (_opt.input_files.back() == ":")
    {   _opt.commands.push_back(this);
        return;
    }

    if (_opt.time_it)
    {   CLI::AutoTimer timer{"\nComplement runtime", CLI::Timer::Simple};
        run_complement();
    }
    else
    {   run_complement();
    }
}

void complement_cmd::run_complement()
{   if (_input_files.empty())
#if defined(__CUDACC__) || defined(__HIPCC__)
        if (_opt.use_gpu)
            run_complement<gnx::dsq>("-");
        else
            run_complement<gnx::sq>("-");
#else
        run_complement<gnx::sq>("-");
#endif // __CUDACC__ || __HIPCC__
    else
    {   if (!_opt.output_file.empty() && _input_files.size() > 1)
        {   printerr
            (   "[complement]: output file {}{}{} cannot be specified for "
                "multiple input files\n"
            ,   gnx::ansi::ESC[fg::yellow]
            ,   _opt.output_file
            ,   gnx::ansi::ESC[style::reset]
            );
            _opt.return_code = 1;
            return;
        }
#if defined(__CUDACC__) || defined(__HIPCC__)
        if (_opt.use_gpu)
            // #pragma omp parallel for schedule(dynamic) if(_input_files.size() > 1)
            for (std::size_t i = 0; i < _input_files.size(); ++i)
                run_complement<gnx::dsq>(_input_files[i]);
        else
            #pragma omp parallel for schedule(dynamic) if(_input_files.size() > 1)
            for (std::size_t i = 0; i < _input_files.size(); ++i)
                run_complement<gnx::sq>(_input_files[i]);
#else
        #pragma omp parallel for schedule(dynamic) if(_input_files.size() > 1)
        for (std::size_t i = 0; i < _input_files.size(); ++i)
            run_complement<gnx::sq>(_input_files[i]);
#endif // __CUDACC__ || __HIPCC__
    }
}

template <typename T>
void complement_cmd::run_complement(std::string const& file)
{   bool is_stdin = (file == "-");

    if (!is_stdin && std::filesystem::is_directory(file))
    {   printerr("[complement]: {} is a directory -- ignored\n", file);
        return;
    }

    // Determine whether input is gzipped (by extension)
    auto ext = std::filesystem::path(file).extension().string();
    bool in_gz = (ext == ".gz" || ext == ".bgz" || ext == ".gzip");

    // Determine output path and format
    bool out_gz = false;
    bool in_place = false;
    std::string out_path;
    std::string tmp_path;

    if (is_stdin && _opt.output_file.empty())
    {   out_path = "-";
        out_gz   = false;
        _faidx = false;
    }
    else if (_opt.use_stdout)
    {   out_path = "-";
        out_gz   = false;
        _faidx = false;
    }
    else if (!_opt.output_file.empty())
    {   out_path = _opt.output_file;
        auto oext = std::filesystem::path(_opt.output_file).extension().string();
        out_gz = (oext == ".gz" || oext == ".bgz" || oext == ".gzip");
    }
    else
    {   // in-place: write to a temp file, then atomically replace original
        in_place = true;
        out_gz   = in_gz;
        tmp_path = file + ".gnxtmp";
        out_path = tmp_path;
    }

    if
    (   !_opt.force
    &&  out_path != "-"
    &&  !in_place
    &&  std::filesystem::exists(out_path)
    )
    {   printerr
        (   "[complement]: {} already exists -- "
            "ignored (use -f|--force to override)\n"
        ,   out_path
        );
        _opt.return_code = 1;
        return;
    }

    try
    {   gnx::forward_stream<T> stream{file};
        auto it     = stream.begin();
        auto end_it = stream.end();

        if (it == end_it)
            return; // empty file — nothing to do

        // Detect FASTA vs FASTQ from the quality field of the first record
        bool is_fastq = !stream.quality().empty();

        // Write all complemented records through the appropriate writer
        auto process = [&](auto& writer)
        {   writer.open(out_path);
            for (; it != end_it; ++it)
            {   auto seq = stream();
                gnx::complement(seq.begin(), seq.end());
                if (_reverse)
                    std::reverse(seq.begin(), seq.end());
                writer.write(seq);
            }
            writer.close();
        };

        int threads_4_compression{1};
        if (out_gz)
            threads_4_compression
            =   !std::filesystem::exists(file)
            ?   1
            :   _opt.threads == -1
                ?   std::min
                    (   int(std::log(double(std::filesystem::file_size(file)) / 1.0e4))
                    ,   _opt.num_procs
                    )
                :   _opt.threads;

        if (is_fastq && out_gz)
        {   _reverse = false;
            gnx::out::fastq_gz w(_faidx, threads_4_compression);
            process(w);
        }
        else if (is_fastq)
        {   _reverse = false;
            gnx::out::fastq w(_faidx);
            process(w);
        }
        else if (out_gz)
        {   gnx::out::fasta_gz w(_faidx, _line_width, threads_4_compression);
            process(w);
        }
        else
        {   gnx::out::fasta w(_faidx, _line_width);
            process(w);
        }

        if (in_place)
            std::filesystem::rename(tmp_path, file);
    }
    catch (std::exception const& e)
    {   printerr("[complement]: {}\n", e.what());
        if (in_place && std::filesystem::exists(tmp_path))
            std::filesystem::remove(tmp_path);
        _opt.return_code = 1;
    }
}

command_type complement_cmd::type() const
{   return command_type::in_place_sequence_processor;
}

void complement_cmd::process(gnx::sq& s) const
{   // fmt::print("[complement]: called with -w={}\n", _line_width);
    gnx::complement(s);
}