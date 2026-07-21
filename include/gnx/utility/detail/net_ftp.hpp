// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <asio.hpp>

#include <fmt/format.h>

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace gnx::detail
{

/// Abstract interface shared by all network transfer sessions (FTP, HTTP,
/// HTTPS). `gnx::detail::download()` reads from this interface without
/// caring about the underlying protocol.
struct net_session
{   virtual ~net_session() = default;
    [[nodiscard]] virtual size_t read(char* buf, size_t len) = 0;
    [[nodiscard]] virtual int64_t file_size() const noexcept = 0;
    virtual void close() noexcept = 0;
};

/// Result of parsing an `ftp://host/path` URL.
struct ftp_url
{   std::string host;
    std::string path;
};

[[nodiscard]]
inline ftp_url parse_ftp_url(std::string_view url)
{   constexpr std::string_view prefix{"ftp://"};
    if (!url.starts_with(prefix))
        throw std::invalid_argument(fmt::format("Not an FTP URL: '{}'", url));
    url.remove_prefix(prefix.size());
    auto slash = url.find('/');
    if (slash == std::string_view::npos)
        throw std::invalid_argument(fmt::format("Malformed FTP URL: '{}'", url));
    return ftp_url
    {   std::string(url.substr(0, slash))
    ,   std::string(url.substr(slash))
    };
}

/// Blocking, anonymous FTP client. Connects on construction, negotiates a
/// passive-mode data connection and issues `RETR`, leaving the data socket
/// ready for streaming via `read()`. Ports the control-channel handshake
/// and `SIZE`/`PASV`/`RETR` sequence from the `kftp_*` routines in
/// knetfile.h.
class ftp_session final : public net_session
{
public:
    explicit ftp_session(std::string_view url)
    :   _control_socket(_io_context)
    ,   _data_socket(_io_context)
    {   auto [host, path] = parse_ftp_url(url);
        _host = std::move(host);
        _path = std::move(path);
        connect_control();
        open_data_connection();
    }

    ftp_session(ftp_session const&) = delete;
    ftp_session& operator=(ftp_session const&) = delete;

    ~ftp_session() override
    {   close();
    }

    [[nodiscard]] size_t read(char* buf, size_t len) override
    {   asio::error_code ec;
        size_t n = _data_socket.read_some(asio::buffer(buf, len), ec);
        if (ec && ec != asio::error::eof)
            throw std::runtime_error(fmt::format("FTP data read failed: {}", ec.message()));
        return n;
    }

    [[nodiscard]] int64_t file_size() const noexcept override
    {   return _file_size;
    }

    void close() noexcept override
    {   asio::error_code ec;
        if (_data_socket.is_open())
            _data_socket.close(ec);
        if (_control_socket.is_open())
        {   try { send_cmd("QUIT\r\n", false); } catch (...) {}
            _control_socket.close(ec);
        }
    }

private:
    [[nodiscard]]
    std::string read_line()
    {   asio::error_code ec;
        asio::read_until(_control_socket, _control_buffer, "\n", ec);
        if (ec && ec != asio::error::eof)
            throw std::runtime_error(fmt::format("FTP control read failed: {}", ec.message()));
        std::istream is(&_control_buffer);
        std::string line;
        std::getline(is, line);
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        return line;
    }

    // Reads a (possibly multi-line) FTP response and returns its 3-digit code.
    int get_response()
    {   std::string line;
        for (;;)
        {   line = read_line();
            if
            (   line.size() >= 4
            &&  std::isdigit(static_cast<unsigned char>(line[0]))
            &&  std::isdigit(static_cast<unsigned char>(line[1]))
            &&  std::isdigit(static_cast<unsigned char>(line[2]))
            &&  line[3] != '-'
            )
            {   _response = line;
                return std::stoi(line.substr(0, 3));
            }
        }
    }

    int send_cmd(std::string_view cmd, bool expect_response = true)
    {   asio::write(_control_socket, asio::buffer(cmd));
        return expect_response ? get_response() : 0;
    }

    void connect_control()
    {   asio::ip::tcp::resolver resolver(_io_context);
        auto endpoints = resolver.resolve(_host, "21");
        asio::connect(_control_socket, endpoints);
        get_response(); // welcome banner
        send_cmd("USER anonymous\r\n");
        send_cmd("PASS gnx@anonymous\r\n");
        send_cmd("TYPE I\r\n");
    }

    // Parses a "227 Entering Passive Mode (h1,h2,h3,h4,p1,p2)." style response.
    [[nodiscard]]
    static std::pair<std::string, std::string> parse_pasv(std::string const& response)
    {   auto lparen = response.find('(');
        if (lparen == std::string::npos)
            throw std::runtime_error(fmt::format("Malformed PASV response: '{}'", response));
        int v[6]{};
        std::sscanf
        (   response.c_str() + lparen + 1
        ,   "%d,%d,%d,%d,%d,%d"
        ,   &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]
        );
        return
        {   fmt::format("{}.{}.{}.{}", v[0], v[1], v[2], v[3])
        ,   std::to_string((v[4] << 8) + v[5])
        };
    }

    void open_data_connection()
    {   auto size_code = send_cmd(fmt::format("SIZE {}\r\n", _path));
        if (size_code == 213)
        {   std::istringstream iss(_response);
            std::string code_str;
            iss >> code_str >> _file_size;
        }
        send_cmd("PASV\r\n");
        auto [pasv_host, pasv_port] = parse_pasv(_response);
        asio::ip::tcp::resolver resolver(_io_context);
        auto endpoints = resolver.resolve(pasv_host, pasv_port);
        asio::connect(_data_socket, endpoints);
        auto retr_code = send_cmd(fmt::format("RETR {}\r\n", _path));
        if (retr_code != 150 && retr_code != 125)
            throw std::runtime_error(fmt::format("FTP RETR failed: {}", _response));
    }

    asio::io_context _io_context;
    asio::ip::tcp::socket _control_socket;
    asio::ip::tcp::socket _data_socket;
    asio::streambuf _control_buffer;
    std::string _host;
    std::string _path;
    std::string _response;
    int64_t _file_size{-1};
};

} // namespace gnx::detail
