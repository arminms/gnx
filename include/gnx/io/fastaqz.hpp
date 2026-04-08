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

#include <zlib.h>
#include <gnx/io/kseq.h>
#include <gnx/memory.hpp>

#define BGZF_MT // enable multi-threading support in bgzf (only effective on writing)
#include <gnx/io/bgzf.h>

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
{   fasta(bool faidx = false, std::size_t line_width = 80)
    :   _fp(nullptr)
    ,   _faidx_fp(nullptr)
    ,   _faidx(faidx)
    ,   _line_width(line_width)
    {}
    void open(std::string_view filename)
    {   _fp = filename == "-"
        ?   stdout
        :   fopen(std::string(filename).c_str(), "wb");
        if (nullptr == _fp)
            throw std::runtime_error
            (   fmt::format("gnx::fasta: could not open file -> {}", filename)
            );
        if (_faidx && filename != "-")
        {   _faidx_fp = fopen((std::string(filename) + ".fai").c_str(), "wb");
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
    }
    void close()
    {   if (_fp && _fp != stdout)
            fclose(_fp);
        if (_faidx_fp)
            fclose(_faidx_fp);
        _fp = nullptr;
        _faidx_fp = nullptr;
    }
    template <class Sequence>
    void write(const Sequence& seq)
    {   std::string id = seq.has("_id")
        ?   std::any_cast<std::string>(seq["_id"])
        :   "gnx_seq";
        std::string header = ">" + id;
        header += seq.has("_desc")
        ?   " " + std::any_cast<std::string>(seq["_desc"]) + "\n"
        :   "\n";
        fwrite
        (   header.c_str()
        ,   sizeof(typename Sequence::value_type)
        ,   header.size()
        ,   _fp
        );
        if (_faidx)
        {   std::size_t line_bases = _line_width ? _line_width : std::size(seq);
            std::size_t line_bytes = line_bases * sizeof(typename Sequence::value_type) + 1;
            std::int64_t offset = ftello(_fp);
            fmt::print
            (   _faidx_fp
            ,   "{}\t{}\t{}\t{}\t{}\n"
            ,   id
            ,   std::size(seq)
            ,   offset
            ,   line_bases
            ,   line_bytes
            );
        }
        const typename Sequence::value_type* data = nullptr;
        universal_host_pinned_vector<typename Sequence::value_type>
            pinned_seq(std::size(seq));
#if defined(__CUDACC__) || defined(__HIPCC__) // handle device_vector
        thrust::copy(seq.begin(), seq.end(), pinned_seq.begin());
        data = thrust::raw_pointer_cast(pinned_seq.data());
#else
        std::copy(seq.begin(), seq.end(), pinned_seq.begin());
        data = static_cast<const typename Sequence::value_type*>
            (pinned_seq.data());
#endif //__CUDACC__
        if (_line_width)
        {   for
            (   typename Sequence::size_type i = 0
            ;   i < std::size(seq)
            ;   i += _line_width
            )
            {   std::string_view line
                (   data + i
                ,   std::min(_line_width, std::size(seq) - i)
                );
                fwrite
                (   line.data()
                ,   sizeof(typename Sequence::value_type)
                ,   line.size()
                ,   _fp
                );
                fwrite("\n", 1, 1, _fp);
            }
        }
        else
        {   fwrite
            (   data
            ,   sizeof(typename Sequence::value_type)
            ,   std::size(seq)
            ,   _fp
            );
            fwrite("\n", 1, 1, _fp);
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
    FILE *_fp, *_faidx_fp;
    bool _faidx;    /// whether to create a .fai index file
    std::size_t _line_width;
};

/// @brief A function object for writing sequences to FASTA files compressed
/// with blocked gzip (BGZF) format and optionally with both .fai and .gzi
/// index files.
struct fasta_gz
{   fasta_gz(bool faidx = false, std::size_t line_width = 80)
    :   _fp(nullptr)
    ,   _faidx_fp(nullptr)
    ,   _gzi_fp(nullptr)
    ,   _faidx(faidx)
    ,   _line_width(line_width)
    ,   _cumul_upos(0)
    {}
    void open(std::string_view filename)
    {   _fp = filename == "-"
        ?   bgzf_dopen(fileno(stdout), "wb")
        :   bgzf_open(std::string(filename).c_str(), "wb");
        if (nullptr == _fp)
            throw std::runtime_error
            (   fmt::format("gnx::fasta_gz: could not open file -> {}", filename)
            );
        if (_faidx && filename != "-")
        {   _faidx_fp = fopen((std::string(filename) + ".fai").c_str(), "wb");
            if (nullptr == _faidx_fp)
            {   bgzf_close(_fp);
                throw std::runtime_error
                (   fmt::format
                    (   "gnx::fasta_gz: could not create FAI index file -> {}.fai"
                    ,   filename
                    )
                );
            }
            _gzi_fp = fopen((std::string(filename) + ".gzi").c_str(), "wb");
            if (nullptr == _gzi_fp)
            {   bgzf_close(_fp); fclose(_faidx_fp);
                throw std::runtime_error
                (   fmt::format
                    (   "gnx::fasta_gz: could not create GZI index file -> {}.gzi"
                    ,   filename
                    )
                );
            }
        }
        else
            _faidx = false; // disable faidx if output is stdout
    }
    void close()
    {   if (_fp)
            bgzf_close(_fp);
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
    void write
    (   const Sequence& seq
    ,   int n_threads = 1
    ,   int n_sub_blks = 256
    )
    {   // MT mode makes bgzf_tell unreliable for block boundary detection,
        // so only enable it when we are not tracking GZI index entries.
        if (n_threads > 1 && !_gzi_fp)
            bgzf_mt(_fp, n_threads, n_sub_blks);
        // Lambda that wraps bgzf_write and tracks BGZF block boundaries for
        // the .gzi index. When _gzi_fp is null it degrades to a plain write.
        auto tracked_write = [&](const void* buf, std::size_t n)
        {   auto voff_before = static_cast<std::uint64_t>(bgzf_tell(_fp));
            bgzf_write(_fp, buf, n);
            _cumul_upos += n;
            if (_gzi_fp)
            {   auto voff_after = static_cast<std::uint64_t>(bgzf_tell(_fp));
                if ((voff_after >> 16) != (voff_before >> 16))
                    _gzi_entries.emplace_back
                    (   voff_after >> 16
                    ,   _cumul_upos - (voff_after & 0xFFFFu)
                    );
            }
        };
        std::string id = seq.has("_id")
        ?   std::any_cast<std::string>(seq["_id"])
        :   "gnx_seq";
        std::string header = ">" + id;
        header += seq.has("_desc")
        ?   " " + std::any_cast<std::string>(seq["_desc"]) + "\n"
        :   "\n";
        tracked_write(header.c_str(), header.size());
        if (_faidx)
        {   std::size_t line_bases = _line_width ? _line_width : std::size(seq);
            std::size_t line_bytes = line_bases * sizeof(typename Sequence::value_type) + 1;
            std::int64_t offset = _cumul_upos;
            fmt::print
            (   _faidx_fp
            ,   "{}\t{}\t{}\t{}\t{}\n"
            ,   id
            ,   std::size(seq)
            ,   offset
            ,   line_bases
            ,   line_bytes
            );
        }
        const typename Sequence::value_type* data = nullptr;
        universal_host_pinned_vector<typename Sequence::value_type>
            pinned_seq(std::size(seq));
#if defined(__CUDACC__) || defined(__HIPCC__) // handle device_vector
        thrust::copy(seq.begin(), seq.end(), pinned_seq.begin());
        data = thrust::raw_pointer_cast(pinned_seq.data());
#else
        std::copy(seq.begin(), seq.end(), pinned_seq.begin());
        data = static_cast<const typename Sequence::value_type*>
            (pinned_seq.data());
#endif //__CUDACC__
        if (_line_width)
        {   for
            (   typename Sequence::size_type i = 0
            ;   i < std::size(seq)
            ;   i += _line_width
            )
            {   std::string_view line
                (   data + i
                ,   std::min(_line_width, std::size(seq) - i)
                );
                tracked_write(line.data(), line.size());
                tracked_write("\n", 1);
            }
        }
        else
        {   tracked_write(data, std::size(seq));
            tracked_write("\n", 1);
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
    BGZF* _fp;
    FILE *_faidx_fp, *_gzi_fp;
    bool _faidx;    /// whether to create .fai and .gzi index files
    std::size_t _line_width;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> _gzi_entries;
    std::uint64_t _cumul_upos;
};

/// @brief A function object for writing sequences in FASTQ format and
/// optionally with a .fai index file.
struct fastq
{   fastq(bool faidx = false, std::size_t line_width = 0)
    :   _fp(nullptr)
    ,   _faidx_fp(nullptr)
    ,   _faidx(faidx)
    ,   _line_width(line_width)
    {}
    void open(std::string_view filename)
    {   _fp = filename == "-"
        ?   stdout
        :   fopen(std::string(filename).c_str(), "wb");
        if (nullptr == _fp)
            throw std::runtime_error
            (   fmt::format("gnx::fastq: could not open file -> {}", filename)
            );
        if (_faidx && filename != "-")
        {   _faidx_fp = fopen((std::string(filename) + ".fai").c_str(), "wb");
            if (nullptr == _faidx_fp)
            {   fclose(_fp);
                throw std::runtime_error
                (   fmt::format
                    (   "gnx::fastq: could not create FAI index file -> {}.fai"
                    ,   filename
                    )
                );
            }
        }
        else
            _faidx = false; // disable faidx if output is stdout
    }
    void close()
    {   if (_fp && _fp != stdout)
            fclose(_fp);
        if (_faidx_fp)
            fclose(_faidx_fp);
        _fp = nullptr;
        _faidx_fp = nullptr;
    }
    template <class Sequence>
    void write(const Sequence& seq)
    {   std::string id = seq.has("_id")
        ?   std::any_cast<std::string>(seq["_id"])
        :   "gnx_seq";
        std::string header = "@" + id;
        header += seq.has("_desc")
        ?   " " + std::any_cast<std::string>(seq["_desc"]) + "\n"
        :   "\n";
        fwrite
        (   header.c_str()
        ,   sizeof(typename Sequence::value_type)
        ,   header.size()
        ,   _fp
        );
        if (_faidx)
        {   std::size_t line_bases = _line_width ? _line_width : std::size(seq);
            std::size_t line_bytes = line_bases * sizeof(typename Sequence::value_type) + 1;
            std::int64_t offset = ftello(_fp);
            const std::size_t seq_lines = _line_width
            ?   (std::size(seq) + _line_width - 1) / _line_width
            :   1;
            std::int64_t qualoffset = offset
                + static_cast<std::int64_t>(std::size(seq))
                + static_cast<std::int64_t>(seq_lines)  // newlines after sequence
                + 2;  // "+\n"
            fmt::print
            (   _faidx_fp
            ,   "{}\t{}\t{}\t{}\t{}\t{}\n"
            ,   id
            ,   std::size(seq)
            ,   offset
            ,   line_bases
            ,   line_bytes
            ,   qualoffset
            );
        }
        const typename Sequence::value_type* data = nullptr;
        universal_host_pinned_vector<typename Sequence::value_type>
            pinned_seq(std::size(seq));
#if defined(__CUDACC__) || defined(__HIPCC__) // handle device_vector
        thrust::copy(seq.begin(), seq.end(), pinned_seq.begin());
        data = thrust::raw_pointer_cast(pinned_seq.data());
#else
        std::copy(seq.begin(), seq.end(), pinned_seq.begin());
        data = static_cast<const typename Sequence::value_type*>
            (pinned_seq.data());
#endif //__CUDACC__
        if (_line_width)
        {   for
            (   typename Sequence::size_type i = 0
            ;   i < std::size(seq)
            ;   i += _line_width
            )
            {   std::string_view line
                (   data + i
                ,   std::min(_line_width, std::size(seq) - i)
                );
                fwrite
                (   line.data()
                ,   sizeof(typename Sequence::value_type)
                ,   line.size()
                ,   _fp
                );
                fwrite("\n", 1, 1, _fp);
            }
        }
        else
        {   fwrite
            (   data
            ,   sizeof(typename Sequence::value_type)
            ,   std::size(seq)
            ,   _fp
            );
            fwrite("\n", 1, 1, _fp);
        }
        fwrite("+\n", 1, 2, _fp);
        std::string qs = seq.has("_qs")
        ?   std::any_cast<std::string>(seq["_qs"])
        :   std::string(std::size(seq), 'I'); // dummy quality string
        if (_line_width)
        {   for
            (   typename Sequence::size_type i = 0
            ;   i < qs.size()
            ;   i += _line_width
            )
            {   std::string_view line
                (   qs.data() + i
                ,   std::min(_line_width, qs.size() - i)
                );
                fwrite
                (   line.data()
                ,   sizeof(std::string_view::value_type)
                ,   line.size()
                ,   _fp
                );
                fwrite("\n", 1, 1, _fp);
            }
        }
        else
        {   fwrite
            (   qs.data()
            ,   sizeof(std::string_view::value_type)
            ,   qs.size()
            ,   _fp
            );
            fwrite("\n", 1, 1, _fp);
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
    FILE *_fp, *_faidx_fp;
    bool _faidx;    /// whether to create a .fai index file
    std::size_t _line_width;
};

/// @brief A function object for writing sequences to FASTQ files compressed
/// with blocked gzip (BGZF) format and optionally with both .fai and .gzi
/// index files.
struct fastq_gz
{   fastq_gz(bool faidx = false, std::size_t line_width = 0)
    :   _fp(nullptr)
    ,   _faidx_fp(nullptr)
    ,   _gzi_fp(nullptr)
    ,   _faidx(faidx)
    ,   _line_width(line_width)
    ,   _cumul_upos(0)
    {}
    void open(std::string_view filename)
    {   _fp = filename == "-"
        ?   bgzf_dopen(fileno(stdout), "wb")
        :   bgzf_open(std::string(filename).c_str(), "wb");
        if (nullptr == _fp)
            throw std::runtime_error
            (   fmt::format("gnx::fastq_gz: could not open file -> {}", filename)
            );
        if (_faidx && filename != "-")
        {   _faidx_fp = fopen((std::string(filename) + ".fai").c_str(), "wb");
            if (nullptr == _faidx_fp)
            {   bgzf_close(_fp);
                throw std::runtime_error
                (   fmt::format
                    (   "gnx::fastq_gz: could not create FAI index file -> {}.fai"
                    ,   filename
                    )
                );
            }
            _gzi_fp = fopen((std::string(filename) + ".gzi").c_str(), "wb");
            if (nullptr == _gzi_fp)
            {   bgzf_close(_fp); fclose(_faidx_fp);
                throw std::runtime_error
                (   fmt::format
                    (   "gnx::fastq_gz: could not create GZI index file -> {}.gzi"
                    ,   filename
                    )
                );
            }
        }
        else
            _faidx = false; // disable faidx if output is stdout
    }
    void close()
    {   if (_fp)
            bgzf_close(_fp);
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
    void write
    (   const Sequence& seq
    ,   int n_threads = 1
    ,   int n_sub_blks = 256
    )
    {   // MT mode makes bgzf_tell unreliable for block boundary detection,
        // so only enable it when we are not tracking GZI index entries.
        if (n_threads > 1 && !_gzi_fp)
            bgzf_mt(_fp, n_threads, n_sub_blks);
        // Lambda that wraps bgzf_write and tracks BGZF block boundaries for
        // the .gzi index. When _gzi_fp is null it degrades to a plain write.
        auto tracked_write = [&](const void* buf, std::size_t n)
        {   auto voff_before = static_cast<std::uint64_t>(bgzf_tell(_fp));
            bgzf_write(_fp, buf, n);
            _cumul_upos += n;
            if (_gzi_fp)
            {   auto voff_after = static_cast<std::uint64_t>(bgzf_tell(_fp));
                if ((voff_after >> 16) != (voff_before >> 16))
                    _gzi_entries.emplace_back
                    (   voff_after >> 16
                    ,   _cumul_upos - (voff_after & 0xFFFFu)
                    );
            }
        };
        std::string id = seq.has("_id")
        ?   std::any_cast<std::string>(seq["_id"])
        :   "gnx_seq";
        std::string header = "@" + id;
        header += seq.has("_desc")
        ?   " " + std::any_cast<std::string>(seq["_desc"]) + "\n"
        :   "\n";
        tracked_write(header.c_str(), header.size());
        const std::uint64_t offset = _cumul_upos;

        const typename Sequence::value_type* data = nullptr;
        universal_host_pinned_vector<typename Sequence::value_type>
            pinned_seq(std::size(seq));
#if defined(__CUDACC__) || defined(__HIPCC__) // handle device_vector
        thrust::copy(seq.begin(), seq.end(), pinned_seq.begin());
        data = thrust::raw_pointer_cast(pinned_seq.data());
#else
        std::copy(seq.begin(), seq.end(), pinned_seq.begin());
        data = static_cast<const typename Sequence::value_type*>
            (pinned_seq.data());
#endif //__CUDACC__
        if (_line_width)
        {   for
            (   typename Sequence::size_type i = 0
            ;   i < std::size(seq)
            ;   i += _line_width
            )
            {   std::string_view line
                (   data + i
                ,   std::min(_line_width, std::size(seq) - i)
                );
                tracked_write(line.data(), line.size());
                tracked_write("\n", 1);
            }
        }

        else
        {   tracked_write(data, std::size(seq));
            tracked_write("\n", 1);
        }
        tracked_write("+\n", 2);
        const std::uint64_t qualoffset = _cumul_upos;
        if (_faidx)
        {   std::size_t line_bases = _line_width ? _line_width : std::size(seq);
            std::size_t line_bytes = line_bases * sizeof(typename Sequence::value_type) + 1;
            fmt::print
            (   _faidx_fp
            ,   "{}\t{}\t{}\t{}\t{}\t{}\n"
            ,   id
            ,   std::size(seq)
            ,   offset
            ,   line_bases
            ,   line_bytes
            ,   qualoffset
            );
        }
        std::string qs = seq.has("_qs")
        ?   std::any_cast<std::string>(seq["_qs"])
        :   std::string(std::size(seq), 'I'); // dummy quality string
        if (_line_width)
        {   for
            (   typename Sequence::size_type i = 0
            ;   i < qs.size()
            ;   i += _line_width
            )
            {   std::string_view line
                (   qs.data() + i
                ,   std::min(_line_width, qs.size() - i)
                );
                tracked_write(line.data(), line.size());
                tracked_write("\n", 1);
            }
        }
        else
        {   tracked_write(qs.data(), qs.size());
            tracked_write("\n", 1);
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
    BGZF* _fp;
    FILE *_faidx_fp, *_gzi_fp;
    bool _faidx;    /// whether to create .fai and .gzi index files
    std::size_t _line_width;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> _gzi_entries;
    std::uint64_t _cumul_upos;
};

}   // end gnx::out namespace
}   // end gnx namespace

