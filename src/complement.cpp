// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
// Implementation of the complement subcommand for the gnx CLI tool
//
#include <gnx/sq.hpp>
#include <gnx/backend/forward_stream.hpp>
#include <gnx/io/fastaqz.hpp>
#include <gnx/algorithms/complement.hpp>

#include "complement.hpp"

complement_cmd::complement_cmd
(   CLI::App& app
,   gnx_options& opt
)
:  _opt(opt)
,  _input_files()
,  _output_file()
,  _use_stdout(false)
,  _force(false)
{   auto* sub = app.add_subcommand
    (   "complement"
    ,   "Complement all sequences in FASTA/FASTQ FILE(s) in-place, "
        "preserving the original format (plain or gzipped)."
    );
    sub->footer
    (   fmt::format
        (   "{}With no FILE, or when FILE is -, read standard input and write "
            "to standard output.\n\n"
            "Report bugs to <https://github.com/arminms/gnx/issues>.{}"
        ,   ansi::style::bold()
        ,   ansi::style::reset()
        )
    );

    sub->add_option
    (   "FILE"
    ,   _input_files
    ,   "input FILEs, null or '-' for stdin"
    )
    ->  type_name("PATH")
    ->  allow_extra_args(true)
    ->  default_val("-")
    ;
    sub->add_flag
    (   "-c,--stdout"
    ,   _use_stdout
    ,   "Write to standard output, keep original file unchanged"
    )
    ;
    sub->add_flag
    (   "-f,--force"
    ,   _force
    ,   "Overwrite output files without asking"
    )
    ;
    sub->add_option
    (   "-o,--out,--output"
    ,   _output_file
    ,   "Write to file instead of modifying FILE in-place"
    )
    ->  type_name("PATH")
    ;

    sub->callback([this]() { run(); });
}

void complement_cmd::run()
{   if (_opt.time_it)
    {   CLI::AutoTimer timer{"\nComplement runtime", CLI::Timer::Simple};
        run_complement();
    }
    else
    {   run_complement();
    }
}

void complement_cmd::run_complement()
{   if (_input_files.empty())
        run_complement("-");
    else
    {   if (!_output_file.empty() && _input_files.size() > 1)
        {   printerr
            (   "[complement]: output file {}{}{} cannot be specified for "
                "multiple input files\n"
            ,   ansi::fg::yellow()
            ,   _output_file
            ,   ansi::fg::reset()
            );
            _opt.return_code = 1;
            return;
        }
        #pragma omp parallel for schedule(dynamic) if(_input_files.size() > 1)
        for (std::size_t i = 0; i < _input_files.size(); ++i)
            run_complement(_input_files[i]);
    }
}

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

    if (_use_stdout || is_stdin)
    {   out_path = "-";
        out_gz   = false;
    }
    else if (!_output_file.empty())
    {   out_path = _output_file;
        auto oext = std::filesystem::path(_output_file).extension().string();
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
    (   !_force
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
    {   gnx::forward_stream<gnx::sq> stream{file};
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
                writer.write(seq);
            }
            writer.close();
        };

        if      (is_fastq && out_gz) { gnx::out::fastq_gz w; process(w); }
        else if (is_fastq)           { gnx::out::fastq   w; process(w); }
        else if (out_gz)             { gnx::out::fasta_gz w; process(w); }
        else                         { gnx::out::fasta   w; process(w); }

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
