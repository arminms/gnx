// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <gnx/utility/detail/wget.hpp>
#include <gnx/concepts.hpp>

namespace gnx {

#if defined(GNX_USE_ASIO) && !defined(__CLING__)
inline detail::wget_result wget
(   std::string_view url
,   size_t buffer_size = 65536
)
{   auto [fp, out_fp, temp_file_path] = detail::wget_init(url);
    detail::download(fp, out_fp, temp_file_path, buffer_size);
    return detail::wget_result(temp_file_path);
}
#else
inline detail::wget_result wget
(   std::string_view url
,   size_t buffer_size = 65536
)
{   auto [fp, out_fp, temp_file_path] = detail::wget_init(url);
#if defined(__CLING__)
    std::thread t
    (   detail::async_download
    ,   fp
    ,   out_fp
    ,   temp_file_path
    ,   buffer_size
    );
    t.detach();
#else
    detail::download(fp, out_fp, temp_file_path, buffer_size);
#endif // __CLING__
    return detail::wget_result(temp_file_path);
}
#endif

#if defined(__CLING__)
template <sequence_container SequenceType>
void wget
(   std::string_view url
,   SequenceType& seq
,   typename SequenceType::size_type ndx
)
{   auto [fp, out_fp, temp_file_path] = detail::wget_init(url);
    std::thread t
    (   detail::async_load_ndx<SequenceType>
    ,   std::ref(seq)
    ,   ndx
    ,   fp
    ,   out_fp
    ,   temp_file_path
    ,   65535
    );
    t.detach();
}

template <sequence_container SequenceType>
void wget
(   std::string_view url
,   SequenceType& seq
,   std::string_view id
)
{   auto [fp, out_fp, temp_file_path] = detail::wget_init(url);
    std::thread t
    (   detail::async_load_id<SequenceType>
    ,   std::ref(seq)
    ,   id
    ,   fp
    ,   out_fp
    ,   temp_file_path
    ,   65535
    );
    t.detach();
}
#endif // __CLING__

} // namespace gnx