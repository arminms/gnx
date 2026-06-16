// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <gnx/utility/detail/wget.hpp>

#if defined(__CLING__)
#   include <xwidgets/xprogress.hpp>
#endif //__CLING__

namespace gnx {

inline detail::wget_result wget
(   std::string_view url
,   size_t buffer_size = 65536
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

#if defined(__CLING__)
    double file_size = 0, downloaded = 0;
    xw::progress<double> progress;
    if (fp->type == KNF_TYPE_FTP)
    {   file_size = static_cast<double>(fp->file_size);
        progress.description = "Starting...";
        progress.style().bar_color = "#4CAF50";
        progress.style().description_width = "180px";
        progress.layout().width = "50%";
        progress.layout().height = "12px";
        xcpp::display(progress);
    }
#endif // __CLING__

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
#if defined(__CLING__)
        auto hrs = detail::human_readable_size(file_size);
        downloaded += static_cast<double>(bytes_read);
        progress.value = downloaded / file_size * 100;
        progress.description = fmt::format
        (   "{:>8} of {:>8} ({}%)"
        ,   detail::human_readable_size(downloaded)
        ,   hrs
        ,   static_cast<int>(progress.value)
        );
#endif // __CLING__
    }
    fclose(out_fp);
    knet_close(fp);
    return detail::wget_result(temp_file_path);
}

} // namespace gnx