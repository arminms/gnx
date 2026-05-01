// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
#pragma once

#include <fmt/core.h>
#include <fmt/format.h>
#include <string>
#include <string_view>
#include <stdexcept>
#include <utility>
#include <vector>
#include <filesystem>

#include <zlib.h>
#include <gnx/io/kseq.h>
#include <gnx/memory.hpp>

#define BGZF_MT // enable multi-threading support in bgzf (only effective on writing)
#include <gnx/io/bgzf.h>
#include <gnx/utility/create_index.hpp>

namespace gnx {

KSEQ_INIT(gzFile, gzread)

/// Input file formats
namespace in {

/// @brief A function object for reading FASTA/FASTQ files (possibly compressed
/// with gzip) and returning a @a Sequence type.
/// @tparam Sequence
template <class Sequence>
struct fast_aqz
{   [[nodiscard]]
    Sequence operator() (std::string_view filename, size_t ndx)
    {   gzFile fp = filename == "-"
        ?   gzdopen(fileno(stdin), "r")
        :   gzopen(std::string(filename).c_str(), "r");
        if (nullptr == fp)
            throw std::runtime_error
                (fmt::format("gnx::fast_aqz: could not open file -> {}", filename));
        kseq_t* seq = kseq_init(fp);
        size_t count = 0;
        int r{};
        while ((r = kseq_read(seq)) >= 0)
            if (ndx == count++)
                break;
        Sequence s = (r > 0) ? Sequence(seq->seq.s) : Sequence();
        if (r > 0)
        {   s["_id"] = std::string(seq->name.s);
            if (seq->qual.l)
                s["_qs"] = std::string(seq->qual.s);
            if (seq->comment.l)
                s["_desc"] = std::string(seq->comment.s);
        }
        kseq_destroy(seq);
        gzclose(fp);
        if (-2 == r)
            throw std::runtime_error
            (   "gnx::fast_aqz: truncated quality string in file -> "
            +   std::string(filename)
            );
        if (-3 == r)
            throw std::runtime_error
            (   "gnx::fast_aqz: error reading file -> "
            +   std::string(filename)
            ); 
        return s;
    }
    [[nodiscard]]
    Sequence operator() (std::string_view filename, std::string_view id)
    {   gzFile fp = filename == "-"
        ?   gzdopen(fileno(stdin), "r")
        :   gzopen(std::string(filename).c_str(), "r");
        if (nullptr == fp)
            throw std::runtime_error
            (   fmt::format("gnx::fast_aqz: could not open file -> {}", filename)
            );
        kseq_t* seq = kseq_init(fp);
        int r{};
        while ((r = kseq_read(seq)) >= 0)
        {   std::string_view name(seq->name.s);
            if (name == id)
                break;
        }
        Sequence s = (r > 0) ? Sequence(seq->seq.s) : Sequence();
        if (r > 0)
        {   s["_id"] = std::string(seq->name.s);
            if (seq->qual.l)
                s["_qs"] = std::string(seq->qual.s);
            if (seq->comment.l)
                s["_desc"] = std::string(seq->comment.s);
        }
        kseq_destroy(seq);
        gzclose(fp);
        if (-2 == r)
            throw std::runtime_error
            (   fmt::format("gnx::fast_aqz: truncated quality string in file -> {}", filename)
            );
        if (-3 == r)
            throw std::runtime_error
            (   fmt::format("gnx::fast_aqz: error reading file -> {}", filename)
            );
        return s;
    }
};

}   // end gnx::in namespace

/// Output file formats

namespace out {

/// @brief A function object for writing sequences in FASTA format and
/// optionally with .fai index file.
struct fasta
{   fasta
    (   bool faidx = false
    ,   std::size_t line_width = 80
    ,   size_t buffer_size = 65536
    )
    :   _fp(nullptr)
    ,   _faidx_fp(nullptr)
    ,   _faidx(faidx)
    ,   _line_width(line_width)
    ,   _serial(1)
    ,   _buffer_size(buffer_size)
    {   if (!_faidx) _buffer.reserve(_buffer_size);
    }
    void open(std::string_view filename)
    {   _filename = filename;
        _fp = _filename == "-"
        ?   stdout
        :   fopen(_filename.c_str(), "wb");
        if (nullptr == _fp)
            throw std::runtime_error
            (   fmt::format("gnx::fasta: could not open file -> {}", _filename)
            );
        if (_faidx && _filename != "-")
        {   _faidx_fp = fopen((_filename + ".fai").c_str(), "wb");
            if (nullptr == _faidx_fp)
            {   fclose(_fp);
                throw std::runtime_error
                (   fmt::format
                    (   "gnx::fasta: could not create FAI index file -> {}.fai"
                    ,   filename
                    )
                );
            }
        }
        else
            _faidx = false; // disable faidx if output is stdout

        _filename
        =   filename == "-"
        ?   "gnx_sq"
        :   std::filesystem::path(filename).stem().string();
    }
    void close()
    {   if (_buffer.size() > 0)
            fwrite(_buffer.data(), sizeof(char), _buffer.size(), _fp);
        if (_fp && _fp != stdout)
            fclose(_fp);
        if (_faidx_fp)
            fclose(_faidx_fp);
        _fp = nullptr;
        _faidx_fp = nullptr;
    }
    template <class Sequence>
    void write(const Sequence& seq)
    {   fmt::format_to
        (   std::back_inserter(_buffer)
        ,   ">{}{}\n"
        ,   seq.has("_id") 
            ?   std::any_cast<std::string>(seq["_id"])
            :   fmt::format("{}.{}", _filename, _serial)
        ,   seq.has("_desc")
            ?   " " + std::any_cast<std::string>(seq["_desc"])
            :   ""
        );
        if (_faidx)
        {   fwrite(_buffer.data(), sizeof(char), _buffer.size(), _fp);
            _buffer.clear();
            std::size_t line_bases = _line_width ? _line_width : std::size(seq);
            std::size_t line_bytes = line_bases * sizeof(char) + 1;
            std::int64_t offset = ftello(_fp);
            fmt::print
            (   _faidx_fp
            ,   "{}\t{}\t{}\t{}\t{}\n"
            ,   seq.has("_id") 
                ?   std::any_cast<std::string>(seq["_id"])
                :   fmt::format("{}.{}", _filename, _serial)
            ,   std::size(seq)
            ,   offset
            ,   line_bases
            ,   line_bytes
            );
        }

        ++_serial;

        auto write_sequence = [&](auto& s)
        {   if (_line_width)
            {   for
                (   typename Sequence::size_type i = 0
                ;   i < std::size(s)
                ;   i += _line_width
                )
                {   std::copy_n
                    (   s.begin() + i
                    ,   std::min(_line_width, std::size(s) - i)
                    ,   std::back_inserter(_buffer)
                    );
                    fmt::format_to(std::back_inserter(_buffer), "\n");
                }
            }
            else
            {   std::copy(s.begin(), s.end(), std::back_inserter(_buffer));
                fmt::format_to(std::back_inserter(_buffer), "\n");
            }
        };

#if defined(__CUDACC__) || defined(__HIPCC__) // handle device_vector
        if constexpr
        (   std::is_same_v<typename Sequence::container_type, thrust::device_vector<typename Sequence::value_type>>
        )
        {   universal_host_pinned_vector<typename Sequence::value_type> pinned_seq(std::size(seq));
            thrust::copy(seq.begin(), seq.end(), pinned_seq.begin());
            write_sequence(pinned_seq);
        }
        else
        {   write_sequence(seq);
        }
#else
        write_sequence(seq);
#endif //__CUDACC__

        if (!_faidx || _buffer.size() > _buffer_size)
        {   fwrite(_buffer.data(), sizeof(char), _buffer.size(), _fp);
            _buffer.clear();
        }
    }
    template <class Sequence>
    int operator()
    (   std::string_view filename
    ,   const Sequence& seq
    )
    {   open(filename);
        write(seq);
        close();
        return 0;
    }

private:
    std::string _filename;
    size_t _serial, _buffer_size;
    FILE *_fp, *_faidx_fp;
    bool _faidx;
    std::size_t _line_width;
    fmt::memory_buffer _buffer;
};

/// @brief A function object for writing sequences to FASTA files compressed
/// with blocked gzip (BGZF) format and optionally with both .fai and .gzi
/// index files.
struct fasta_gz
{   fasta_gz
    (   bool faidx = false
    ,   std::size_t line_width = 80
    ,   int n_threads = 1
    ,   int n_sub_blks = 256
    ,   size_t buffer_size = 65536
    )
    :   _fp(nullptr)
    ,   _faidx_fp(nullptr)
    ,   _gzi_fp(nullptr)
    ,   _faidx(faidx)
    ,   _line_width(line_width)
    ,   _serial(1)
    ,   _buffer_size(buffer_size)
    ,   _cumul_upos(0)
    ,   _threads(n_threads)
    ,   _sub_blks(n_sub_blks)
    {   if (!_faidx) _buffer.reserve(_buffer_size);
    }
    void open(std::string_view filename)
    {   _filename = filename;
        _fp = _filename == "-"
        ?   bgzf_dopen(fileno(stdout), "wb")
        :   bgzf_open(_filename.c_str(), "wb");
        if (nullptr == _fp)
            throw std::runtime_error
            (   fmt::format("gnx::fasta_gz: could not open file -> {}", _filename)
            );
        if (_faidx && _filename != "-")
        {   _faidx_fp = fopen((_filename + ".fai").c_str(), "wb");
            if (nullptr == _faidx_fp)
            {   bgzf_close(_fp);
                throw std::runtime_error
                (   fmt::format
                    (   "gnx::fasta_gz: could not create FAI index file -> {}.fai"
                    ,   _filename
                    )
                );
            }
            // MT mode makes bgzf_tell unreliable for block boundary detection,
            // so we'll generate .gzi after writing the .gz file.
            if (_threads <= 1)
            {   _gzi_fp = fopen((_filename + ".gzi").c_str(), "wb");
                if (nullptr == _gzi_fp)
                {   bgzf_close(_fp); fclose(_faidx_fp);
                    throw std::runtime_error
                    (   fmt::format
                        (   "gnx::fasta_gz: could not create GZI index file -> {}.gzi"
                        ,   _filename
                        )
                    );
                }
            }
        }
        else
            _faidx = false; // disable faidx if output is stdout

        _id
        =   filename == "-"
        ?   "gnx_sq"
        :   std::filesystem::path(filename).stem().string();

        if (_threads > 1)
            bgzf_mt(_fp, _threads, _sub_blks);
    }
    void close()
    {   if (_buffer.size() > 0)
            tracked_write(_buffer.data(), _buffer.size());
        if (_fp)
            bgzf_close(_fp);
        if (_faidx_fp && !_gzi_fp)
            create_gzi(_filename, _filename + ".gzi");
        if (_faidx_fp)
            fclose(_faidx_fp);
        if (_gzi_fp)
        {   // Write accumulated GZI block entries in SAMtools little-endian format.
            auto write_u64 = [&](std::uint64_t v)
            {   std::uint8_t buf[8];
                buf[0] = static_cast<std::uint8_t>(v);
                buf[1] = static_cast<std::uint8_t>(v >> 8);
                buf[2] = static_cast<std::uint8_t>(v >> 16);
                buf[3] = static_cast<std::uint8_t>(v >> 24);
                buf[4] = static_cast<std::uint8_t>(v >> 32);
                buf[5] = static_cast<std::uint8_t>(v >> 40);
                buf[6] = static_cast<std::uint8_t>(v >> 48);
                buf[7] = static_cast<std::uint8_t>(v >> 56);
                fwrite(buf, 1, 8, _gzi_fp);
            };
            write_u64(static_cast<std::uint64_t>(_gzi_entries.size()));
            for (const auto& [coff, uoff] : _gzi_entries)
            {   write_u64(coff);
                write_u64(uoff);
            }
            fclose(_gzi_fp);
        }
        _fp = nullptr;
        _faidx_fp = nullptr;
        _gzi_fp = nullptr;
        _gzi_entries.clear();
        _cumul_upos = 0;
    }
    template <class Sequence>
    void write(const Sequence& seq)
    {   fmt::format_to
        (   std::back_inserter(_buffer)
        ,   ">{}{}\n"
        ,   seq.has("_id") 
            ?   std::any_cast<std::string>(seq["_id"])
            :   fmt::format("{}.{}", _id, _serial)
        ,   seq.has("_desc")
            ?   " " + std::any_cast<std::string>(seq["_desc"])
            :   ""
        );

        if (_faidx)
        {   tracked_write(_buffer.data(), _buffer.size());
            std::size_t line_bases = _line_width ? _line_width : std::size(seq);
            std::size_t line_bytes = line_bases * sizeof(typename Sequence::value_type) + 1;
            std::int64_t offset = _cumul_upos;
            fmt::print
            (   _faidx_fp
            ,   "{}\t{}\t{}\t{}\t{}\n"
            ,   seq.has("_id") 
                ?   std::any_cast<std::string>(seq["_id"])
                :   fmt::format("{}.{}", _id, _serial)
            ,   std::size(seq)
            ,   offset
            ,   line_bases
            ,   line_bytes
            );
        }

        ++_serial;

        auto write_sequence = [&](auto& s)
        {   if (_line_width)
            {   for
                (   typename Sequence::size_type i = 0
                ;   i < std::size(s)
                ;   i += _line_width
                )
                {   std::copy_n
                    (   s.begin() + i
                    ,   std::min(_line_width, std::size(s) - i)
                    ,   std::back_inserter(_buffer)
                    );
                    fmt::format_to(std::back_inserter(_buffer), "\n");
                }
            }
            else
            {   std::copy(s.begin(), s.end(), std::back_inserter(_buffer));
                fmt::format_to(std::back_inserter(_buffer), "\n");
            }
        };

#if defined(__CUDACC__) || defined(__HIPCC__) // handle device_vector
        if constexpr
        (   std::is_same_v<typename Sequence::container_type, thrust::device_vector<typename Sequence::value_type>>
        )
        {   universal_host_pinned_vector<typename Sequence::value_type> pinned_seq(std::size(seq));
            thrust::copy(seq.begin(), seq.end(), pinned_seq.begin());
            write_sequence(pinned_seq);
        }
        else
        {   write_sequence(seq);
        }
#else
        write_sequence(seq);
#endif //__CUDACC__

        if (!_faidx || _buffer.size() > _buffer_size)
            tracked_write(_buffer.data(), _buffer.size());
    }
    template <class Sequence>
    int operator()
    (   std::string_view filename
    ,   const Sequence& seq
    )
    {   open(filename);
        write(seq);
        close();
        return 0;
    }

private:
    std::string _filename, _id;
    size_t _serial, _buffer_size;
    BGZF* _fp;
    FILE *_faidx_fp, *_gzi_fp;
    bool _faidx;
    std::size_t _line_width;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> _gzi_entries;
    std::uint64_t _cumul_upos;
    int _threads, _sub_blks;
    fmt::memory_buffer _buffer;

    // wraps bgzf_write and tracks BGZF block boundaries for
    // the .gzi index. When _gzi_fp is null it degrades to a plain write.
    void tracked_write(const void* buf, std::size_t n)
    {   auto voff_before = static_cast<std::uint64_t>(bgzf_tell(_fp));
        bgzf_write(_fp, buf, n);
        _buffer.clear();
        _cumul_upos += n;
        if (_gzi_fp)
        {   auto voff_after = static_cast<std::uint64_t>(bgzf_tell(_fp));
            if ((voff_after >> 16) != (voff_before >> 16))
                _gzi_entries.emplace_back
                (   voff_after >> 16
                ,   _cumul_upos - (voff_after & 0xFFFFu)
                );
        }
    }
};

/// @brief A function object for writing sequences in FASTQ format and
/// optionally with a .fai index file.
struct fastq
{   fastq
    (   bool faidx = false
    ,   size_t buffer_size = 65536
    )
    :   _fp(nullptr)
    ,   _faidx_fp(nullptr)
    ,   _faidx(faidx)
    ,   _serial(1)
    ,   _buffer_size(buffer_size)
    {   if (!_faidx) _buffer.reserve(_buffer_size);
    }
    void open(std::string_view filename)
    {   _filename = filename;
        _fp = _filename == "-"
        ?   stdout
        :   fopen(_filename.c_str(), "wb");
        if (nullptr == _fp)
            throw std::runtime_error
            (   fmt::format("gnx::fastq: could not open file -> {}", filename)
            );
        if (_faidx && _filename != "-")
        {   _faidx_fp = fopen((_filename + ".fai").c_str(), "wb");
            if (nullptr == _faidx_fp)
            {   fclose(_fp);
                throw std::runtime_error
                (   fmt::format
                    (   "gnx::fastq: could not create FAI index file -> {}.fai"
                    ,   _filename
                    )
                );
            }
        }
        else
            _faidx = false; // disable faidx if output is stdout

        _filename
        =   filename == "-"
        ?   "gnx_sq"
        :   std::filesystem::path(filename).stem().string();
    }
    void close()
    {   if (_buffer.size() > 0)
            fwrite(_buffer.data(), sizeof(char), _buffer.size(), _fp);
        if (_fp && _fp != stdout)
            fclose(_fp);
        if (_faidx_fp)
            fclose(_faidx_fp);
        _fp = nullptr;
        _faidx_fp = nullptr;
    }
    template <class Sequence>
    void write(const Sequence& seq)
    {   fmt::format_to
        (   std::back_inserter(_buffer)
        ,   "@{}{}\n"
        ,   seq.has("_id") 
            ?   std::any_cast<std::string>(seq["_id"])
            :   fmt::format("{}.{}", _filename, _serial)
        ,   seq.has("_desc")
            ?   " " + std::any_cast<std::string>(seq["_desc"])
            :   ""
        );

        if (_faidx)
        {   fwrite(_buffer.data(), sizeof(char), _buffer.size(), _fp);
            _buffer.clear();
            std::size_t line_bases = std::size(seq);
            std::size_t line_bytes = line_bases * sizeof(char) + 1;
            std::int64_t offset = ftello(_fp);
            const std::size_t seq_lines = 1;
            std::int64_t qualoffset = offset
                + static_cast<std::int64_t>(std::size(seq))
                + static_cast<std::int64_t>(seq_lines)  // newlines after sequence
                + 2;  // "+\n"
            fmt::print
            (   _faidx_fp
            ,   "{}\t{}\t{}\t{}\t{}\t{}\n"
            ,   seq.has("_id") 
                ?   std::any_cast<std::string>(seq["_id"])
                :   fmt::format("{}.{}", _filename, _serial)
            ,   std::size(seq)
            ,   offset
            ,   line_bases
            ,   line_bytes
            ,   qualoffset
            );
        }

        ++_serial;

        auto write_sequence_and_quality = [&](auto& s)
        {   std::copy(s.begin(), s.end(), std::back_inserter(_buffer));
            fmt::format_to
            (   std::back_inserter(_buffer)
            ,   "\n+\n{}\n"
            ,   seq.has("_qs")
                ?   std::any_cast<std::string>(seq["_qs"])
                :   std::string(std::size(seq), 'I') // dummy quality string
            );
        };

#if defined(__CUDACC__) || defined(__HIPCC__)
        if constexpr
        (   std::is_same_v<typename Sequence::container_type, thrust::device_vector<typename Sequence::value_type>>
        )
        {   universal_host_pinned_vector<typename Sequence::value_type> pinned_seq(std::size(seq));
            thrust::copy(seq.begin(), seq.end(), pinned_seq.begin());
            write_sequence_and_quality(pinned_seq);
        }
        else
        {   write_sequence_and_quality(seq);
        }
#else
        write_sequence_and_quality(seq);
#endif //__CUDACC__

        if (!_faidx || _buffer.size() > _buffer_size)
        {   fwrite(_buffer.data(), sizeof(char), _buffer.size(), _fp);
            _buffer.clear();
        }
    }
    template <class Sequence>
    int operator()
    (   std::string_view filename
    ,   const Sequence& seq
    )
    {   open(filename);
        write(seq);
        close();
        return 0;
    }

private:
    std::string _filename;
    size_t _serial, _buffer_size;
    FILE *_fp, *_faidx_fp;
    bool _faidx;    /// whether to create a .fai index file
    fmt::memory_buffer _buffer;
    // gnx::pinned_buffer _buffer;
};

/// @brief A function object for writing sequences to FASTQ files compressed
/// with blocked gzip (BGZF) format and optionally with both .fai and .gzi
/// index files.
struct fastq_gz
{   fastq_gz
    (   bool faidx = false
    ,   int n_threads = 1
    ,   int n_sub_blks = 256
    ,   size_t buffer_size = 65536
    )
    :   _fp(nullptr)
    ,   _faidx_fp(nullptr)
    ,   _gzi_fp(nullptr)
    ,   _faidx(faidx)
    ,   _cumul_upos(0)
    ,   _threads(n_threads)
    ,   _sub_blks(n_sub_blks)
    ,   _buffer_size(buffer_size)
    {   if (!_faidx) _buffer.reserve(_buffer_size);
    }
    void open(std::string_view filename)
    {   _filename = filename;
        _fp = _filename == "-"
        ?   bgzf_dopen(fileno(stdout), "wb")
        :   bgzf_open(_filename.c_str(), "wb");
        if (nullptr == _fp)
            throw std::runtime_error
            (   fmt::format("gnx::fastq_gz: could not open file -> {}", _filename)
            );
        if (_faidx && _filename != "-")
        {   _faidx_fp = fopen((_filename + ".fai").c_str(), "wb");
            if (nullptr == _faidx_fp)
            {   bgzf_close(_fp);
                throw std::runtime_error
                (   fmt::format
                    (   "gnx::fastq_gz: could not create FAI index file -> {}.fai"
                    ,   _filename
                    )
                );
            }
            // MT mode makes bgzf_tell unreliable for block boundary detection,
            // so we'll generate .gzi after writing the .gz file.
            if (_threads <= 1)
            {   _gzi_fp = fopen((_filename + ".gzi").c_str(), "wb");
                if (nullptr == _gzi_fp)
                {   bgzf_close(_fp); fclose(_faidx_fp);
                    throw std::runtime_error
                    (   fmt::format
                        (   "gnx::fastq_gz: could not create GZI index file -> {}.gzi"
                        ,   _filename
                        )
                    );
                }
            }
        }
        else
            _faidx = false; // disable faidx if output is stdout

        _id
        =   filename == "-"
        ?   "gnx_sq"
        :   std::filesystem::path(filename).stem().string();

        if (_threads > 1)
            bgzf_mt(_fp, _threads, _sub_blks);
    }
    void close()
    {   if (_buffer.size() > 0)
            tracked_write(_buffer.data(), _buffer.size());
        if (_fp)
            bgzf_close(_fp);
        if (_faidx_fp && !_gzi_fp)
            create_gzi(_filename, _filename + ".gzi");
        if (_faidx_fp)
            fclose(_faidx_fp);
        if (_gzi_fp)
        {   // Write accumulated GZI block entries in SAMtools little-endian format.
            auto write_u64 = [&](std::uint64_t v)
            {   std::uint8_t buf[8];
                buf[0] = static_cast<std::uint8_t>(v);
                buf[1] = static_cast<std::uint8_t>(v >> 8);
                buf[2] = static_cast<std::uint8_t>(v >> 16);
                buf[3] = static_cast<std::uint8_t>(v >> 24);
                buf[4] = static_cast<std::uint8_t>(v >> 32);
                buf[5] = static_cast<std::uint8_t>(v >> 40);
                buf[6] = static_cast<std::uint8_t>(v >> 48);
                buf[7] = static_cast<std::uint8_t>(v >> 56);
                fwrite(buf, 1, 8, _gzi_fp);
            };
            write_u64(static_cast<std::uint64_t>(_gzi_entries.size()));
            for (const auto& [coff, uoff] : _gzi_entries)
            {   write_u64(coff);
                write_u64(uoff);
            }
            fclose(_gzi_fp);
        }
        _fp = nullptr;
        _faidx_fp = nullptr;
        _gzi_fp = nullptr;
        _gzi_entries.clear();
        _cumul_upos = 0;
    }
    template <class Sequence>
    void write(const Sequence& seq)
    {   fmt::format_to
        (   std::back_inserter(_buffer)
        ,   "@{}{}\n"
        ,   seq.has("_id") 
            ?   std::any_cast<std::string>(seq["_id"])
            :   fmt::format("{}.{}", _id, _serial)
        ,   seq.has("_desc")
            ?   " " + std::any_cast<std::string>(seq["_desc"])
            :   ""
        );
        if (_faidx)
        {   tracked_write(_buffer.data(), _buffer.size());
            _buffer.clear();
        }
        const std::uint64_t offset = _cumul_upos;

#if defined(__CUDACC__) || defined(__HIPCC__)
        if constexpr
        (   std::is_same_v<typename Sequence::container_type, thrust::device_vector<typename Sequence::value_type>>
        )
        {   universal_host_pinned_vector<typename Sequence::value_type> pinned_seq(std::size(seq));
            thrust::copy(seq.begin(), seq.end(), pinned_seq.begin());
            std::copy(pinned_seq.begin(), pinned_seq.end(), std::back_inserter(_buffer));
        }
        else
        {   std::copy(seq.begin(), seq.end(), std::back_inserter(_buffer));
        }
#else
        std::copy(seq.begin(), seq.end(), std::back_inserter(_buffer));
#endif //__CUDACC__

        fmt::format_to(std::back_inserter(_buffer), "\n+\n");

        if (_faidx)
        {   tracked_write(_buffer.data(), _buffer.size());
            _buffer.clear();
            const std::uint64_t qualoffset = _cumul_upos;
            std::size_t line_bases = std::size(seq);
            std::size_t line_bytes = line_bases * sizeof(char) + 1;
            fmt::print
            (   _faidx_fp
            ,   "{}\t{}\t{}\t{}\t{}\t{}\n"
            ,   seq.has("_id") 
                ?   std::any_cast<std::string>(seq["_id"])
                :   fmt::format("{}.{}", _id, _serial)
            ,   std::size(seq)
            ,   offset
            ,   line_bases
            ,   line_bytes
            ,   qualoffset
            );
        }

        ++_serial;

        fmt::format_to
        (   std::back_inserter(_buffer)
        ,   "{}\n"
        ,   seq.has("_qs")
            ?   std::any_cast<std::string>(seq["_qs"])
            :   std::string(std::size(seq), 'I') // dummy quality string
        );

        if (!_faidx || _buffer.size() > _buffer_size)
        {   tracked_write(_buffer.data(), _buffer.size());
            _buffer.clear();
        }
    }
    template <class Sequence>
    int operator()
    (   std::string_view filename
    ,   const Sequence& seq
    )
    {   open(filename);
        write(seq);
        close();
        return 0;
    }
private:
    std::string _filename, _id;
    size_t _serial, _buffer_size;
    BGZF* _fp;
    FILE *_faidx_fp, *_gzi_fp;
    bool _faidx;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> _gzi_entries;
    std::uint64_t _cumul_upos;
    int _threads, _sub_blks;
    fmt::memory_buffer _buffer;

    // wraps bgzf_write and tracks BGZF block boundaries for
    // the .gzi index. When _gzi_fp is null it degrades to a plain write.
    void tracked_write(const void* buf, std::size_t n)
    {   auto voff_before = static_cast<std::uint64_t>(bgzf_tell(_fp));
        bgzf_write(_fp, buf, n);
        _buffer.clear();
        _cumul_upos += n;
        if (_gzi_fp)
        {   auto voff_after = static_cast<std::uint64_t>(bgzf_tell(_fp));
            if ((voff_after >> 16) != (voff_before >> 16))
                _gzi_entries.emplace_back
                (   voff_after >> 16
                ,   _cumul_upos - (voff_after & 0xFFFFu)
                );
        }
    }
};

}   // end gnx::out namespace
}   // end gnx namespace

