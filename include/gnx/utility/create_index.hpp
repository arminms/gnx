// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <optional>

#include <fmt/core.h>

namespace gnx {

// -- index utilities ----------------------------------------------------------

/// @brief Scans @p fasta_path and writes a SAMtools-compatible .fai index
///        to @p fai_path.
///
/// The FASTA file is read in binary mode so byte offsets are exact on all
/// platforms. Both LF and CRLF line endings are handled correctly.
///
/// @param fasta_path  Path to the plain FASTA file.
/// @param fai_path    Output path for the .fai index file.
/// @throws std::runtime_error  If @p fasta_path cannot be read or
///                             @p fai_path cannot be created.
inline void create_fai
(   const std::string& fasta_path
,   const std::string& fai_path
)
{   FILE* fa = std::fopen(fasta_path.c_str(), "rb");
    if (!fa)
        throw std::runtime_error
        (   fmt::format
            (   "gnx::create_fai(): cannot open FASTA to build index -> {}"
            ,   fasta_path
            )
        );
    std::ofstream out(fai_path);
    if (!out)
    {   std::fclose(fa);
        throw std::runtime_error
        (   fmt::format
            (   "gnx::create_fai(): cannot create FAI index -> {}"
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

/// @brief Scans a bgzip-compressed file and writes a SAMtools-compatible
///        .gzi block index to @p gzi_path.
///
/// Only the 18-byte BGZF block header and the 4-byte ISIZE footer field
/// are read per block — no decompression is performed. The output binary
/// format is identical to the one produced by `samtools faidx`:
///   - `uint64_t N` (little-endian) — number of entries.
///   - N × (`uint64_t compressed_offset`, `uint64_t uncompressed_offset`)
///     both little-endian. The implicit block 0 at (0, 0) is **not** stored.
///
/// @param bgzf_path  Path to the bgzip-compressed file.
/// @param gzi_path   Output path for the .gzi index file.
/// @throws std::runtime_error  If @p bgzf_path cannot be read,
///                             @p gzi_path cannot be created, or the
///                             bgzip file is truncated or malformed.
inline void create_gzi
(   const std::string& bgzf_path
,   const std::string& gzi_path
)
{   if (bgzf_is_bgzf(bgzf_path.c_str()) != 1)
        throw std::runtime_error
        (   fmt::format
            (   "gnx::create_gzi(): not a block-compressed gzip file -> {}"
            ,   bgzf_path
            )
        );
    FILE* fp = std::fopen(bgzf_path.c_str(), "rb");
    if (!fp)
        throw std::runtime_error
        (   fmt::format
            (   "gnx::create_gzi(): cannot open bgzip file to build GZI -> {}"
            ,   bgzf_path
            )
        );
    std::ofstream out(gzi_path, std::ios::binary);
    if (!out)
    {   std::fclose(fp);
        throw std::runtime_error
        (   fmt::format
            (   "gnx::create_gzi(): cannot create GZI index -> {}"
            ,   gzi_path
            )
        );
    }
    // Little-endian uint64 writer.
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
        if (!out.write(reinterpret_cast<const char*>(buf), 8))
            throw std::runtime_error
            (   fmt::format
                (   "gnx::virtual_vector: I/O error writing GZI index -> {}"
                ,   gzi_path
                )
            );
    };
    struct Block { std::uint64_t coff, uoff; };
    std::vector<Block> entries;
    std::uint64_t cumulative_upos = 0;
    bool first = true;
    while (true)
    {
#if defined(_WIN32)
        std::int64_t cstart = ::_ftelli64(fp);
#else
        std::int64_t cstart = ::ftello(fp);
#endif
        if (cstart < 0)
        {   std::fclose(fp);
            throw std::runtime_error
            (   fmt::format
                (   "gnx::create_gzi(): ftell failed for bgzip file -> {}"
                ,   bgzf_path
                )
            );
        }
        // Read 18-byte BGZF block header.
        std::uint8_t header[18];
        const std::size_t n = std::fread(header, 1, 18, fp);
        if (n == 0) break;   // clean EOF
        if (n != 18)
        {   std::fclose(fp);
            throw std::runtime_error
            (   fmt::format
                (   "gnx::create_gzi(): truncated BGZF block header -> {}"
                ,   bgzf_path
                )
            );
        }
        // BSIZE at bytes 16-17 (little-endian uint16); block_size = BSIZE + 1.
        const std::int64_t block_size =
            1
        +   static_cast<std::int64_t>(header[16])
        +  (static_cast<std::int64_t>(header[17]) << 8);
        // Seek to ISIZE: last 4 bytes of the block.
        const std::int64_t isize_pos = cstart + block_size - 4;
#if defined(_WIN32)
        if (0 != ::_fseeki64(fp, isize_pos, SEEK_SET))
#else
        if (0 != ::fseeko(fp, static_cast<::off_t>(isize_pos), SEEK_SET))
#endif
        {   std::fclose(fp);
            throw std::runtime_error
            (   fmt::format
                (   "gnx::create_gzi(): seek failed in bgzip file -> {}"
                ,   bgzf_path
                )
            );
        }
        std::uint8_t ibuf[4];
        if (std::fread(ibuf, 1, 4, fp) != 4)
        {   std::fclose(fp);
            throw std::runtime_error
            (   fmt::format
                (   "gnx::create_gzi(): truncated BGZF block (ISIZE missing) -> {}"
                ,   bgzf_path
                )
            );
        }
        const std::uint32_t usize =
                static_cast<std::uint32_t>(ibuf[0])
            | (static_cast<std::uint32_t>(ibuf[1]) << 8)
            | (static_cast<std::uint32_t>(ibuf[2]) << 16)
            | (static_cast<std::uint32_t>(ibuf[3]) << 24);
        if (usize == 0) break;  // bgzip EOF marker block
        // Block 0 at (0,0) is implicit in the .gzi format — skip first entry.
        if (!first)
            entries.push_back({static_cast<std::uint64_t>(cstart), cumulative_upos});
        first = false;
        cumulative_upos += static_cast<std::uint64_t>(usize);
        // Advance to the next block.
#if defined(_WIN32)
        if (0 != ::_fseeki64(fp, cstart + block_size, SEEK_SET))
#else
        if (0 != ::fseeko(fp, static_cast<::off_t>(cstart + block_size), SEEK_SET))
#endif
        {   std::fclose(fp);
            throw std::runtime_error
            (   fmt::format
                (   "gnx::create_gzi(): seek to next block failed -> {}"
                ,   bgzf_path
                )
            );
        }
    }
    std::fclose(fp);
    // Write count then entries.
    write_u64(static_cast<std::uint64_t>(entries.size()));
    for (const auto& e : entries)
    {   write_u64(e.coff);
        write_u64(e.uoff);
    }
}

} // namespace gnx