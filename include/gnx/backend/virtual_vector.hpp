// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
#pragma once

#include <compare>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/core.h>
#include <gnx/concepts.hpp>

namespace gnx {

/// @brief A vector-like backend for gnx::sequence_bank that keeps almost no
/// sequence data in RAM.
///
/// On construction, the FASTA index (.fai) is parsed and its entries are
/// stored in memory (~40 bytes each). When a sequence is requested via
/// operator[] or through an iterator, the corresponding bytes are read from
/// the uncompressed FASTA file by seeking to the precomputed byte offset
/// recorded in the FAI entry.
///
/// Compatible with gnx::sequence_bank as a BackendType.
///
/// @tparam SequenceType  A type satisfying gnx::sequence_container.
///
/// @note If the .fai index file does not exist it is created automatically
///       the first time the constructor runs. Alternatively it can be
///       generated externally with: @code samtools faidx genome.fa @endcode
///
/// @note Not thread-safe: all calls to operator[] share a single file handle.
///       Create separate instances per thread for concurrent access.
template <gnx::sequence_container SequenceType>
class virtual_vector
{
public:
    // -- FAI entry ------------------------------------------------------------

    /// @brief One record from a FASTA index (.fai) file.
    struct fai_entry
    {   std::string  name;         ///< Sequence name
        std::int64_t length{};     ///< Number of bases in the sequence
        std::int64_t offset{};     ///< Byte offset of the first base in FASTA
        std::int32_t linebases{};  ///< Bases per line
        std::int32_t linewidth{};  ///< Bytes per line (including newline bytes)
    };

    // -- proxy random-access iterator -----------------------------------------

    /// @brief Random-access iterator that materialises SequenceType on
    /// dereference by reading from disk. Iteration is efficient for sequential
    /// access patterns; each dereference performs a seek + read.
    struct iterator
    {   const virtual_vector* _vv  = nullptr;
        std::ptrdiff_t        _idx = 0;

        using value_type        = SequenceType;
        using difference_type   = std::ptrdiff_t;
        using reference         = SequenceType;   ///< Proxy: returned by value
        using pointer           = void;
        using iterator_category = std::random_access_iterator_tag;

        iterator() noexcept = default;
        iterator(const virtual_vector* vv, std::ptrdiff_t idx) noexcept
        :   _vv(vv), _idx(idx)
        {}

        reference operator*() const
        {   return (*_vv)[static_cast<std::size_t>(_idx)];
        }
        reference operator[](difference_type n) const
        {   return (*_vv)[static_cast<std::size_t>(_idx + n)];
        }

        iterator& operator++()    noexcept { ++_idx; return *this; }
        iterator  operator++(int) noexcept { auto t = *this; ++*this; return t; }
        iterator& operator--()    noexcept { --_idx; return *this; }
        iterator  operator--(int) noexcept { auto t = *this; --*this; return t; }

        iterator& operator+=(difference_type n) noexcept
        {   _idx += n; return *this;
        }
        iterator& operator-=(difference_type n) noexcept
        {   _idx -= n; return *this;
        }

        friend iterator operator+(iterator a, difference_type n) noexcept
        {   return a += n;
        }
        friend iterator operator+(difference_type n, iterator a) noexcept
        {   return a += n;
        }
        friend iterator operator-(iterator a, difference_type n) noexcept
        {   return a -= n;
        }
        friend difference_type operator-(const iterator& a, const iterator& b) noexcept
        {   return a._idx - b._idx;
        }

        friend bool operator==(const iterator&, const iterator&) noexcept = default;
        friend auto operator<=>(const iterator& a, const iterator& b) noexcept
        {   return a._idx <=> b._idx;
        }
    };

    using value_type      = SequenceType;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using const_iterator  = iterator;

    // -- constructors ---------------------------------------------------------

    virtual_vector() = delete;

    /// @brief Opens a FASTA file and loads its index for on-demand access.
    ///
    /// If the .fai index file does not exist it is built automatically by
    /// scanning @p fasta_path once and written next to it (or to @p fai_path).
    ///
    /// @param fasta_path  Path to the uncompressed FASTA file.
    /// @param fai_path    Path to the .fai index file. When empty the path is
    ///                    derived as @p fasta_path + ".fai".
    /// @throws std::runtime_error  If a file cannot be opened/created or if
    ///                             the index contains a malformed line.
    explicit virtual_vector
    (   std::string_view fasta_path
    ,   std::string_view fai_path = {}
    )
    :   _fasta_path(fasta_path)
    {   std::string idx_path = fai_path.empty()
        ?   _fasta_path + ".fai"
        :   std::string(fai_path);
        {   std::ifstream probe(idx_path);
            if (!probe.good())
                build_fai(idx_path);
        }
        load_fai(idx_path);
        _fp = std::fopen(_fasta_path.c_str(), "rb");
        if (!_fp)
            throw std::runtime_error
            (   fmt::format
                (   "gnx::virtual_vector: could not open FASTA file -> {}"
                ,   _fasta_path
                )
            );
    }

    virtual_vector(const virtual_vector&)            = delete;
    virtual_vector& operator=(const virtual_vector&) = delete;

    virtual_vector(virtual_vector&& o) noexcept
    :   _index(std::move(o._index))
    ,   _fasta_path(std::move(o._fasta_path))
    ,   _fp(o._fp)
    {   o._fp = nullptr;
    }

    virtual_vector& operator=(virtual_vector&& o) noexcept
    {   if (this != &o)
        {   if (_fp) std::fclose(_fp);
            _index      = std::move(o._index);
            _fasta_path = std::move(o._fasta_path);
            _fp         = o._fp;
            o._fp       = nullptr;
        }
        return *this;
    }

    ~virtual_vector()
    {   if (_fp) std::fclose(_fp);
    }

    // -- capacity -------------------------------------------------------------

    [[nodiscard]] size_type size()  const noexcept { return _index.size(); }
    [[nodiscard]] bool      empty() const noexcept { return _index.empty(); }

    // -- element access -------------------------------------------------------

    /// @brief Returns the sequence name of entry @p i (no disk I/O).
    [[nodiscard]] std::string_view name(size_type i) const
    {   return _index[i].name;
    }

    /// @brief Reads and returns the @p i-th sequence from disk.
    ///
    /// Complexity: O(sequence length). A seek plus one read per line.
    /// @throws std::runtime_error  If seeking the file fails.
    [[nodiscard]] SequenceType operator[](size_type i) const
    {   return read_at(_index[i]);
    }

    /// @brief Bounds-checked element access.
    /// @throws std::out_of_range  If @p i >= size().
    [[nodiscard]] SequenceType at(size_type i) const
    {   if (i >= _index.size())
            throw std::out_of_range
            (   fmt::format
                (   "gnx::virtual_vector::at: index {} out of range [0, {})"
                ,   i
                ,   _index.size()
                )
            );
        return read_at(_index[i]);
    }

    /// @brief Returns the raw FAI entry for the @p i-th sequence (no disk I/O).
    [[nodiscard]] const fai_entry& entry(size_type i) const
    {   return _index[i];
    }

    // -- iterators ------------------------------------------------------------

    [[nodiscard]] iterator begin() const noexcept { return {this, 0}; }
    [[nodiscard]] iterator end()   const noexcept
    {   return {this, static_cast<std::ptrdiff_t>(_index.size())};
    }

private:
    std::vector<fai_entry> _index;
    std::string            _fasta_path;
    mutable FILE*          _fp = nullptr;  ///< mutable: seek/read are logically const

    /// @brief Scans @c _fasta_path and writes a SAMtools-compatible .fai index
    ///        to @p fai_path.
    ///
    /// The FASTA file is read in binary mode so byte offsets are exact on all
    /// platforms. Both LF and CRLF line endings are handled correctly.
    ///
    /// @throws std::runtime_error  If @c _fasta_path cannot be read or
    ///                             @p fai_path cannot be created.
    void build_fai(const std::string& fai_path)
    {   FILE* fa = std::fopen(_fasta_path.c_str(), "rb");
        if (!fa)
            throw std::runtime_error
            (   fmt::format
                (   "gnx::virtual_vector: cannot open FASTA to build index -> {}"
                ,   _fasta_path
                )
            );
        std::ofstream out(fai_path);
        if (!out)
        {   std::fclose(fa);
            throw std::runtime_error
            (   fmt::format
                (   "gnx::virtual_vector: cannot create FAI index -> {}"
                ,   fai_path
                )
            );
        }
        // Byte position in the file; updated inside read_line.
        std::int64_t pos = 0;
        // Read one raw line; returns {content_without_CR_LF, raw_byte_count}
        // or nullopt at EOF.
        auto read_line = [&]() -> std::optional<std::pair<std::string, std::int32_t>>
        {   std::string content;
            std::int32_t raw = 0;
            bool have = false;
            int c;
            while ((c = std::fgetc(fa)) != EOF)
            {   ++pos; ++raw; have = true;
                if (c == '\n') break;
                if (c != '\r') content += static_cast<char>(c);
            }
            if (!have) return std::nullopt;
            return std::make_pair(std::move(content), raw);
        };
        struct Builder
        {   std::string  name;
            std::int64_t length     = 0;
            std::int64_t offset     = 0;
            std::int32_t linebases  = 0;
            std::int32_t linewidth  = 0;
            bool         first_line = true;
        };
        std::optional<Builder> cur;
        auto flush = [&]()
        {   if (cur && !cur->name.empty())
                out << cur->name  << '\t'
                    << cur->length    << '\t'
                    << cur->offset    << '\t'
                    << cur->linebases << '\t'
                    << cur->linewidth << '\n';
            cur.reset();
        };
        while (true)
        {   auto res = read_line();
            if (!res) break;
            auto& [content, rawlen] = *res;
            if (!content.empty() && content[0] == '>')
            {   flush();
                cur.emplace();
                std::size_t ws = content.find_first_of(" \t", 1);
                cur->name   = content.substr(1, ws - 1);
                cur->offset = pos;  // first base is right after the header '\n'
            }
            else if (cur && !content.empty())
            {   if (cur->first_line)
                {   cur->linebases  = static_cast<std::int32_t>(content.size());
                    cur->linewidth  = rawlen;
                    cur->first_line = false;
                }
                cur->length += static_cast<std::int64_t>(content.size());
            }
        }
        flush();
        std::fclose(fa);
    }

    void load_fai(const std::string& path)
    {   std::ifstream in(path);
        if (!in)
            throw std::runtime_error
            (   fmt::format
                (   "gnx::virtual_vector: could not open FAI index -> {}"
                ,   path
                )
            );
        std::string line;
        while (std::getline(in, line))
        {   if (line.empty()) continue;
            std::istringstream ss(line);
            fai_entry e;
            if (!(ss >> e.name >> e.length >> e.offset >> e.linebases >> e.linewidth))
                throw std::runtime_error
                (   fmt::format
                    (   "gnx::virtual_vector: malformed FAI line -> {}"
                    ,   line
                    )
                );
            _index.push_back(std::move(e));
        }
    }

    /// @brief Seeks to the byte offset of @p e and reads its bases line by line.
    [[nodiscard]] SequenceType read_at(const fai_entry& e) const
    {
#if defined(_WIN32)
        if (0 != ::_fseeki64(_fp, e.offset, SEEK_SET))
#else
        if (0 != ::fseeko(_fp, static_cast<::off_t>(e.offset), SEEK_SET))
#endif
            throw std::runtime_error
            (   fmt::format
                (   "gnx::virtual_vector: seek failed for sequence -> {}"
                ,   e.name
                )
            );

        std::string seq;
        seq.reserve(static_cast<std::size_t>(e.length));

        std::vector<char> buf(static_cast<std::size_t>(e.linebases));
        std::int64_t remaining = e.length;
        while (remaining > 0)
        {   auto to_read = static_cast<std::size_t>
            (   std::min(remaining, static_cast<std::int64_t>(e.linebases))
            );
            std::size_t n = std::fread(buf.data(), 1, to_read, _fp);
            if (n == 0) break;
            seq.append(buf.data(), n);
            remaining -= static_cast<std::int64_t>(n);
            if (remaining > 0)
            {   // skip the newline byte(s) between lines
#if defined(_WIN32)
                ::_fseeki64(_fp, e.linewidth - e.linebases, SEEK_CUR);
#else
                ::fseeko
                (   _fp
                ,   static_cast<::off_t>(e.linewidth - e.linebases)
                ,   SEEK_CUR
                );
#endif
            }
        }

        SequenceType s(seq.c_str());
        s["_id"] = e.name;
        return s;
    }
};

} // namespace gnx
