// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#if defined(GNX_USE_ASIO)
#   if defined(__CLING__)
#       error "Asio networking support is not available in cling mode. Please disable GNX_USE_ASIO."
#   else
#       include <gnx/utility/detail/net_ftp.hpp>
#       include <gnx/utility/detail/net_http.hpp>
#   endif
#else
#   include <gnx/utility/detail/knetfile.h>
#endif

#include <fmt/base.h>
#include <fmt/format.h>

#include <string_view>
#include <filesystem>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <utility>

#if defined(__CLING__)
#   include <xwidgets/xbox.hpp>
#   include <xwidgets/xlabel.hpp>
#   include <xwidgets/xprogress.hpp>
#   include <xwidgets/xbutton.hpp>
#endif //__CLING__

namespace gnx::detail
{

struct wget_result
{   wget_result() = default;
    wget_result(wget_result const&) = delete;
    wget_result& operator=(wget_result const&) = delete;
    wget_result(wget_result&&) = default;
    wget_result& operator=(wget_result&&) = default;
    wget_result(std::filesystem::path const& temp_file_path)
    :   _temp_file_path(temp_file_path)
    ,   _file_path(temp_file_path.string())
    {}
    ~wget_result()
    {   if (std::filesystem::exists(_temp_file_path))
            std::filesystem::remove(_temp_file_path);
    }
    std::string_view operator() () const noexcept
    {   return _file_path;
    }
    std::filesystem::path file_path() const noexcept
    {   return _temp_file_path;
    }
    void close() noexcept
    {   if (std::filesystem::exists(_temp_file_path))
            std::filesystem::remove(_temp_file_path);
        _file_path.clear();
    }

private:
    std::filesystem::path _temp_file_path;
    std::string _file_path;
};

[[nodiscard]]
inline bool is_valid_url(std::string_view url)
{   return
    (   url.starts_with("ftp://")
    ||  url.starts_with("http://")
    ||  url.starts_with("genome://")
    ||  url.starts_with("sra://")
#if defined(GNX_USE_ASIO)
    ||  url.starts_with("https://")
#endif
    );  
}

[[nodiscard]]
inline std::string construct_genome_url(std::string_view genome_acn_n_assembly)
{   genome_acn_n_assembly.remove_prefix(9); // remove "genome://"
    return fmt::format
    (   "ftp://ftp.ncbi.nlm.nih.gov/genomes/all/{}/{}/{}/{}/{}/{}_genomic.fna.gz"
    ,   genome_acn_n_assembly.substr(0, 3)
    ,   genome_acn_n_assembly.substr(4, 3)
    ,   genome_acn_n_assembly.substr(7, 3)
    ,   genome_acn_n_assembly.substr(10, 3)
    ,   genome_acn_n_assembly
    ,   genome_acn_n_assembly
    );
}

[[nodiscard]]
inline std::string construct_sra_url(std::string_view run_acn)
{   std::string result, base{"ftp://ftp.sra.ebi.ac.uk/vol1/fastq"};
    run_acn.remove_prefix(6); // remove "sra://"
    auto pos = run_acn.rfind('_');
    auto accession
    =   pos == std::string_view::npos
    ?   std::string(run_acn)
    :   std::string(run_acn.substr(0, pos));
    if (accession.size() == 9)
        result = fmt::format
        (   "{}/{}/{}/{}.fastq.gz"
        ,   base
        ,   accession.substr(0, 6)
        ,   accession
        ,   run_acn
        );
    else if (accession.size() == 10)
        result = fmt::format
        (   "{}/{}/{:0>3}/{}/{}.fastq.gz"
        ,   base
        ,   accession.substr(0, 6)
        ,   accession.substr(9, 1)
        ,   accession
        ,   run_acn
        );
    else if (accession.size() == 11)
        result = fmt::format
        (   "{}/{}/{:0>3}/{}/{}.fastq.gz"
        ,   base
        ,   accession.substr(0, 6)
        ,   accession.substr(9, 2)
        ,   accession
        ,   run_acn
        );
    else if (accession.size() >= 12)
        result = fmt::format
        (   "{}/{}/{}/{}/{}.fastq.gz"
        ,   base
        ,   accession.substr(0, 6)
        ,   accession.substr(9, 3)
        ,   accession
        ,   run_acn
        );
    else
        throw std::invalid_argument
        (   fmt::format
            (   "Invalid SRA accession in '{}'. Expected format: "
                "'sra://SRR12345678' or 'sra://SRR12345678_1'"
            ,   run_acn
            )
        );
    
    return result;
}

[[nodiscard]]
inline std::string human_readable_size(double size)
{   int suffix_index{0};
    for (; size >= 1024.; size /= 1024., ++suffix_index);
    return fmt::format
    (   "{:.1f} {}B"
    ,   std::ceil(size * 10.) / 10.
    ,   "BKMGTPE"[suffix_index]
    );
};

#if defined(GNX_USE_ASIO)
    [[nodiscard]]
    inline std::tuple<std::unique_ptr<net_session>, FILE*, std::filesystem::path>
    wget_init(std::string_view url)
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

        std::unique_ptr<net_session> session;
        try
        {   session
            =   url_str.starts_with("ftp://")
            ?   std::unique_ptr<net_session>(std::make_unique<ftp_session>(url_str))
            :   detail::open_http(url_str);
        }
        catch (std::exception const& e)
        {   throw std::runtime_error
            (   fmt::format("Failed to open URL: {} ({})", url_str, e.what())
            );
        }

        auto temp_file_path
        =   std::filesystem::temp_directory_path()
        /   std::filesystem::path(url_str).filename();
        auto out_fp = fopen(temp_file_path.string().c_str(), "wb");
        if (out_fp == nullptr)
            throw std::runtime_error
            (   fmt::format
                (   "Failed to create temporary file: {}"
                ,   temp_file_path.string()
                )
            );
        return std::make_tuple(std::move(session), out_fp, temp_file_path);
    }

    inline void download
    (   std::unique_ptr<net_session>& fp
    ,   FILE* out_fp
    ,   std::filesystem::path temp_file_path
    ,   size_t buffer_size
    )
    {   fmt::memory_buffer buffer;
        buffer.reserve(buffer_size);
        size_t bytes_read;
        while ((bytes_read = fp->read(buffer.data(), buffer_size)) > 0)
            fwrite(buffer.data(), 1, bytes_read, out_fp);
        fclose(out_fp);
        fp->close();
    }
#else
    [[nodiscard]]
    inline std::tuple<knetFile*, FILE*, std::filesystem::path>
    wget_init(std::string_view url)
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
        auto out_fp = fopen(temp_file_path.string().c_str(), "wb");
        if (out_fp == nullptr)
        {   knet_close(fp);
            throw std::runtime_error
            (   fmt::format
                (   "Failed to create temporary file: {}"
                ,   temp_file_path.string()
                )
            );
        }
        return std::make_tuple(fp, out_fp, temp_file_path);
    }

inline void download
(   knetFile* fp
,   FILE* out_fp
,   std::filesystem::path temp_file_path
,   size_t buffer_size
)
{   fmt::memory_buffer buffer;
    buffer.reserve(buffer_size);
    size_t bytes_read;
    while ((bytes_read = knet_read(fp, buffer.data(), buffer_size)) > 0)
        fwrite(buffer.data(), 1, bytes_read, out_fp);
    fclose(out_fp);
    knet_close(fp);
}
#endif

#if defined(__CLING__)
    inline void async_download
    (   knetFile* fp
    ,   FILE* out_fp
    ,   std::filesystem::path temp_file_path
    ,   size_t buffer_size
    )
    {   double file_size = 0, downloaded = 0;
        bool cancel_download = false;
        xw::button cancel_button;
        xw::label downloaded_label, speed_label;
        xw::progress<double> progress;
        xw::hbox box;
        box.layout().width = "50%";
        box.add(cancel_button);
        box.add(progress);
        box.add(downloaded_label);
        box.add(speed_label);
        cancel_button.on_click
        (   [&cancel_button, &cancel_download]()
            {   cancel_download = true;
                cancel_button.button_style = "success";
            }
        );
        if (fp->type == KNF_TYPE_FTP)
        {   file_size = static_cast<double>(fp->file_size);
            // cancel_button.layout().height = "25px";
            cancel_button.layout().margin = "0 0 0 auto";
            cancel_button.layout().width = "40px"; 
            cancel_button.button_style = "danger";
            cancel_button.icon = "close";
            cancel_button.tooltip = "Cancel the download";
            downloaded_label.layout().margin = "0 0 0 auto";
            downloaded_label.layout().width = "25%";
            speed_label.layout().margin = "0 0 0 auto";
            speed_label.layout().width = "5%";
            progress.style().bar_color = "#4CAF50";
            progress.layout().width = "67%";
            xcpp::display(box);
        }
        fmt::memory_buffer buffer;
        buffer.reserve(buffer_size);
        size_t bytes_read;
        auto timer_start = std::chrono::steady_clock::now();
        while ((bytes_read = knet_read(fp, buffer.data(), buffer_size)) > 0)
        {   if (cancel_download)
                break;
            fwrite(buffer.data(), 1, bytes_read, out_fp);
            auto hrs_file_size = detail::human_readable_size(file_size);
            downloaded += static_cast<double>(bytes_read);
            double elapsed_seconds = std::chrono::duration<double>
            (   std::chrono::steady_clock::now()
            -   timer_start
            ).count();
            double speed = downloaded / elapsed_seconds;
            progress.value = downloaded / file_size * 100;
            progress.description = fmt::format
            (   "{}%"
            ,   static_cast<int>(progress.value)
            );
            downloaded_label.value = fmt::format
            (   "{:>8} / {:>8}"
            ,   detail::human_readable_size(downloaded)
            ,   hrs_file_size
            );
            speed_label.value = fmt::format
            (   "{:>8}/s"
            ,   detail::human_readable_size(speed)
            );
        }
        fclose(out_fp);
        knet_close(fp);
        if (cancel_download)
            std::filesystem::remove(temp_file_path);
    }

    template <sequence_container SequenceType>
    void async_load_ndx
    (   SequenceType& seq
    ,   typename SequenceType::size_type ndx
    ,   knetFile* fp
    ,   FILE* out_fp
    ,   std::filesystem::path temp_file_path
    ,   size_t buffer_size
    )
    {   detail::async_download(fp, out_fp, temp_file_path, buffer_size);
        if (std::filesystem::exists(temp_file_path))
            seq.load(temp_file_path.string(), ndx);
        std::filesystem::remove(temp_file_path);
    }

    template <sequence_container SequenceType>
    void async_load_id
    (   SequenceType& seq
    ,   std::string_view id
    ,   knetFile* fp
    ,   FILE* out_fp
    ,   std::filesystem::path temp_file_path
    ,   size_t buffer_size
    )
    {   detail::async_download(fp, out_fp, temp_file_path, buffer_size);
        if (std::filesystem::exists(temp_file_path))
            seq.load(temp_file_path.string(), id);
        std::filesystem::remove(temp_file_path);
    }
#endif // __CLING__

} // namespace gnx::detail