// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <gnx/utility/detail/wget.hpp>

namespace gnx {

inline wget_result wget(std::string_view url)
{   auto temp_file_path
    =   std::filesystem::temp_directory_path()
    /   std::filesystem::path(url).filename()
    ;
#ifdef _WIN32
    knet_win32_init();
#endif

    auto fp = knet_open(url.data(), "r");
    if (fp == nullptr)
        throw std::runtime_error(fmt::format("Failed to open URL: {}", url));
    FILE* out_fp = fopen(temp_file_path.c_str(), "wb");
    if (out_fp == nullptr)
    {   knet_close(fp);
        throw std::runtime_error
        (   fmt::format
            (   "Failed to create temporary file: {}"
            ,   temp_file_path.string()
            )
        );
    }
    char buffer[8192];
    size_t bytes_read;
    while ((bytes_read = knet_read(fp, buffer, sizeof(buffer))) > 0)
        fwrite(buffer, 1, bytes_read, out_fp);
    fclose(out_fp);
    knet_close(fp);
    return wget_result(temp_file_path);
}

} // namespace gnx