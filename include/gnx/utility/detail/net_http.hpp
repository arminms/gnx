// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <gnx/utility/detail/net_ftp.hpp> // for gnx::detail::net_session

#include <asio.hpp>
#include <asio/ssl.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <istream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
    #include <wincrypt.h>
    #if defined(_MSC_VER)
        #pragma comment(lib, "crypt32.lib")
    #endif
#endif

namespace gnx::detail
{

/// Result of parsing an `http://` or `https://` URL.
struct http_url
{   bool tls{false};
    std::string host;
    std::string port;
    std::string path;
};

[[nodiscard]]
inline http_url parse_http_url(std::string_view url)
{   http_url result;
    if (url.starts_with("https://"))
    {   result.tls = true;
        result.port = "443";
        url.remove_prefix(8);
    }
    else if (url.starts_with("http://"))
    {   result.tls = false;
        result.port = "80";
        url.remove_prefix(7);
    }
    else
        throw std::invalid_argument(fmt::format("Not an HTTP(S) URL: '{}'", url));

    auto slash = url.find('/');
    auto authority = slash == std::string_view::npos ? url : url.substr(0, slash);
    result.path = slash == std::string_view::npos ? "/" : std::string(url.substr(slash));

    auto colon = authority.find(':');
    if (colon == std::string_view::npos)
        result.host = std::string(authority);
    else
    {   result.host = std::string(authority.substr(0, colon));
        result.port = std::string(authority.substr(colon + 1));
    }
    return result;
}

/// Parsed HTTP response status line + headers of interest.
struct http_response_info
{   int status_code{0};
    int64_t content_length{-1};
    std::string location;
};

[[nodiscard]]
inline bool iequals(std::string_view a, std::string_view b) noexcept
{   return std::ranges::equal
    (   a, b
    ,   [](char x, char y)
        {   return std::tolower(static_cast<unsigned char>(x))
            ==  std::tolower(static_cast<unsigned char>(y));
        }
    );
}

template <typename SyncReadWriteStream>
inline void send_http_get(SyncReadWriteStream& stream, http_url const& url)
{   auto request = fmt::format
    (   "GET {} HTTP/1.1\r\n"
        "Host: {}\r\n"
        "Connection: close\r\n"
        "User-Agent: gnx-wget/1.0\r\n"
        "\r\n"
    ,   url.path
    ,   url.host
    );
    asio::write(stream, asio::buffer(request));
}

template <typename SyncReadStream>
[[nodiscard]]
inline http_response_info read_http_headers(SyncReadStream& stream, asio::streambuf& buf)
{   asio::error_code ec;
    asio::read_until(stream, buf, "\r\n\r\n", ec);
    if (ec)
        throw std::runtime_error(fmt::format("HTTP header read failed: {}", ec.message()));

    std::istream is(&buf);
    std::string status_line;
    std::getline(is, status_line);
    if (!status_line.empty() && status_line.back() == '\r')
        status_line.pop_back();
    auto first_space = status_line.find(' ');
    if (first_space == std::string::npos)
        throw std::runtime_error(fmt::format("Malformed HTTP status line: '{}'", status_line));

    http_response_info info;
    info.status_code = std::stoi(status_line.substr(first_space + 1));

    std::string header_line;
    while (std::getline(is, header_line))
    {   if (!header_line.empty() && header_line.back() == '\r')
            header_line.pop_back();
        if (header_line.empty())
            break; // blank line marks end of headers
        auto colon = header_line.find(':');
        if (colon == std::string::npos)
            continue;
        auto name = header_line.substr(0, colon);
        auto value_start = header_line.find_first_not_of(' ', colon + 1);
        auto value
        =   value_start == std::string::npos
        ?   std::string{}
        :   header_line.substr(value_start);
        if (iequals(name, "Content-Length"))
            info.content_length = std::stoll(value);
        else if (iequals(name, "Location"))
            info.location = value;
    }
    return info;
}

template <typename SyncReadStream>
[[nodiscard]]
inline size_t http_read(SyncReadStream& stream, asio::streambuf& buf, char* out, size_t len)
{   if (buf.size() > 0)
    {   size_t n = buf.sgetn(out, static_cast<std::streamsize>(len));
        if (n > 0)
            return n;
    }
    asio::error_code ec;
    size_t n = stream.read_some(asio::buffer(out, len), ec);
    if (ec && ec != asio::error::eof)
        throw std::runtime_error(fmt::format("HTTP read failed: {}", ec.message()));
    return n;
}

/// Blocking, plain-text HTTP GET session. Connects and parses the response
/// headers on construction, leaving the socket positioned at the start of
/// the body, ready for `read()`.
class http_session final : public net_session
{
public:
    explicit http_session(http_url const& url)
    :   _socket(_io_context)
    {   asio::ip::tcp::resolver resolver(_io_context);
        auto endpoints = resolver.resolve(url.host, url.port);
        asio::connect(_socket, endpoints);
        send_http_get(_socket, url);
        _info = read_http_headers(_socket, _buf);
    }

    http_session(http_session const&) = delete;
    http_session& operator=(http_session const&) = delete;

    ~http_session() override
    {   close();
    }

    [[nodiscard]] size_t read(char* buf, size_t len) override
    {   return http_read(_socket, _buf, buf, len);
    }

    [[nodiscard]] int64_t file_size() const noexcept override
    {   return _info.content_length;
    }

    void close() noexcept override
    {   asio::error_code ec;
        _socket.close(ec);
    }

    [[nodiscard]] int status_code() const noexcept { return _info.status_code; }
    [[nodiscard]] std::string const& location() const noexcept { return _info.location; }

private:
    asio::io_context _io_context;
    asio::ip::tcp::socket _socket;
    asio::streambuf _buf;
    http_response_info _info;
};

#ifdef _WIN32
/// Loads the OS-trusted root certificates from the Windows "ROOT"
/// certificate store (via CryptoAPI) into an Asio/OpenSSL SSL context.
///
/// OpenSSL has no knowledge of the Windows certificate store, so
/// `asio::ssl::context::set_default_verify_paths()` looks for Unix-style CA
/// bundle files/env vars that don't exist on Windows, causing every HTTPS
/// handshake to fail with "certificate verify failed". This function feeds
/// the OS-trusted roots directly into OpenSSL's certificate store instead.
inline void load_windows_root_certificates(asio::ssl::context& ctx)
{   HCERTSTORE cert_store = CertOpenSystemStoreW(0, L"ROOT");
    if (!cert_store)
        throw std::runtime_error("Failed to open Windows ROOT certificate store");

    X509_STORE* x509_store = SSL_CTX_get_cert_store(ctx.native_handle());
    PCCERT_CONTEXT cert_ctx = nullptr;
    while ((cert_ctx = CertEnumCertificatesInStore(cert_store, cert_ctx)) != nullptr)
    {   const unsigned char* encoded = cert_ctx->pbCertEncoded;
        X509* x509 = d2i_X509(nullptr, &encoded, cert_ctx->cbCertEncoded);
        if (x509)
        {   X509_STORE_add_cert(x509_store, x509);
            X509_free(x509);
        }
    }
    CertCloseStore(cert_store, 0);
}
#endif // _WIN32

/// Blocking HTTPS GET session over `asio::ssl::stream`, with SNI and peer
/// hostname verification enabled.
class https_session final : public net_session
{
public:
    explicit https_session(http_url const& url)
    :   _stream(_io_context, _ssl_context)
    {
#ifdef _WIN32
        load_windows_root_certificates(_ssl_context);
#else
        _ssl_context.set_default_verify_paths();
#endif
        asio::ip::tcp::resolver resolver(_io_context);
        auto endpoints = resolver.resolve(url.host, url.port);
        asio::connect(_stream.lowest_layer(), endpoints);
        if (!SSL_set_tlsext_host_name(_stream.native_handle(), url.host.c_str()))
            throw std::runtime_error("Failed to set SNI hostname");
        _stream.set_verify_mode(asio::ssl::verify_peer);
        _stream.set_verify_callback(asio::ssl::host_name_verification(url.host));
        _stream.handshake(asio::ssl::stream_base::client);
        send_http_get(_stream, url);
        _info = read_http_headers(_stream, _buf);
    }

    https_session(https_session const&) = delete;
    https_session& operator=(https_session const&) = delete;

    ~https_session() override
    {   close();
    }

    [[nodiscard]] size_t read(char* buf, size_t len) override
    {   return http_read(_stream, _buf, buf, len);
    }

    [[nodiscard]] int64_t file_size() const noexcept override
    {   return _info.content_length;
    }

    void close() noexcept override
    {   asio::error_code ec;
        _stream.lowest_layer().close(ec);
    }

    [[nodiscard]] int status_code() const noexcept { return _info.status_code; }
    [[nodiscard]] std::string const& location() const noexcept { return _info.location; }

private:
    asio::io_context _io_context;
    asio::ssl::context _ssl_context{asio::ssl::context::tlsv12_client};
    asio::ssl::stream<asio::ip::tcp::socket> _stream;
    asio::streambuf _buf;
    http_response_info _info;
};

/// Opens an `http://` or `https://` URL, following redirects (up to
/// `max_redirects` hops), and returns a session ready to stream the
/// response body via `read()`.
[[nodiscard]]
inline std::unique_ptr<net_session> open_http(std::string_view url, int max_redirects = 5)
{   std::string current_url(url);
    for (int hop = 0; hop <= max_redirects; ++hop)
    {   auto parsed = parse_http_url(current_url);
        std::unique_ptr<net_session> session;
        int status{};
        std::string location;
        if (parsed.tls)
        {   auto https = std::make_unique<https_session>(parsed);
            status = https->status_code();
            location = https->location();
            session = std::move(https);
        }
        else
        {   auto http = std::make_unique<http_session>(parsed);
            status = http->status_code();
            location = http->location();
            session = std::move(http);
        }
        if (status >= 300 && status < 400 && !location.empty())
        {   current_url = location;
            continue;
        }
        if (status != 200 && status != 206)
            throw std::runtime_error
            (   fmt::format("HTTP request failed with status {} for '{}'", status, current_url)
            );
        return session;
    }
    throw std::runtime_error(fmt::format("Too many redirects for URL: '{}'", url));
}

} // namespace gnx::detail
