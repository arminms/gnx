// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <gnx/utility/detail/wget.hpp>

namespace gnx {

inline detail::wget_result wget
(   std::string_view url
,   size_t buffer_size = 1 << 20 // 1 MiB
)
{   if (!detail::is_valid_url(url))
        throw std::invalid_argument
        (   fmt::format
            (   "Unsupported URL scheme in '{}'"
            ,   url
            )
        );
    std::string url_str(url);
    if (url.starts_with("genome://"))
        url_str = detail::construct_genome_url(url);
    if (url.starts_with("sra://"))
        url_str = detail::construct_sra_url(url);
#ifdef _WIN32
    knet_win32_init();
#endif
    auto fp = knet_open(url_str.data(), "r");
    if (fp == nullptr)
        throw std::runtime_error(fmt::format("Failed to open URL: {}", url_str));
    auto temp_file_path
    =   std::filesystem::temp_directory_path()
    /   std::filesystem::path(url_str).filename();
    FILE* out_fp = fopen(temp_file_path.string().c_str(), "wb");
    if (out_fp == nullptr)
    {   knet_close(fp);
        throw std::runtime_error
        (   fmt::format
            (   "Failed to create temporary file: {}"
            ,   temp_file_path.string()
            )
        );
    }
    fmt::memory_buffer buffer;
    buffer.reserve(buffer_size);
    size_t bytes_read;
    while ((bytes_read = knet_read(fp, buffer.data(), buffer_size)) > 0)
    {   fwrite(buffer.data(), 1, bytes_read, out_fp);
        buffer.clear();
    }
    fclose(out_fp);
    knet_close(fp);
    return detail::wget_result(temp_file_path);
}

} // namespace gnx