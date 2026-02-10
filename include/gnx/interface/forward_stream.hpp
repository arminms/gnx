// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
#pragma once

#include <gnx/concepts.hpp>
#include <gnx/io/fastaqz.hpp>

namespace gnx {

template <gnx::sequence_container SequenceType>
class forward_stream
{   gzFile _fp;
    kseq_t* _seq;
    int _read_result;

public:
    struct iterator
    {   forward_stream* _stream;
        bool _end;

        using value_type = SequenceType;
        using difference_type = std::ptrdiff_t;
        using pointer = const forward_stream*;
        using reference = const forward_stream&;
        using iterator_category = std::input_iterator_tag;

        iterator()
        :   _stream(nullptr)
        ,   _end(true)
        {}
        explicit iterator(forward_stream* stream)
        :   _stream(stream)
        ,   _end(false)
        {   if ((_stream->_read_result = kseq_read(_stream->_seq)) < 0)
            {   _end = true;
                _stream = nullptr;
            }
        }

        reference operator*() const
        {   return *_stream;
        }
        pointer operator->() const
        {   return _stream;
        }

        iterator& operator++()
        {   if ((_stream->_read_result = kseq_read(_stream->_seq)) < 0)
            {   _end = true;
                _stream = nullptr;
            }
            return *this;
        }

        iterator operator++(int)
        {   iterator temp = *this;
            ++(*this);
            return temp;
        }

        friend bool operator==(const iterator& a, const iterator& b)
        {   return a._end == b._end && a._stream == b._stream;
        }

        friend bool operator!=(const iterator& a, const iterator& b)
        {   return !(a == b);
        }
    };

    using value_type = forward_stream;

    forward_stream() = delete;
    forward_stream(std::string_view filename)
    :   _fp(filename == "-"
        ?   gzdopen(fileno(stdin), "r")
        :   gzopen(std::string(filename).c_str(), "r"))
    ,   _seq(kseq_init(_fp))
    ,   _read_result(0)
    {   if (nullptr == _fp)
            throw std::runtime_error
            (   "gnx::forward_stream: could not open file -> "
            +   std::string(filename)
            );
    }
    forward_stream(const forward_stream&) = delete;
    forward_stream& operator=(const forward_stream&) = delete;
    forward_stream(forward_stream&& other) noexcept
    :   _fp(other._fp)
    ,   _seq(other._seq)
    ,   _read_result(other._read_result)
    {   other._fp = nullptr;
        other._seq = nullptr;
        other._read_result = 0;
    }
    forward_stream& operator=(forward_stream&& other) noexcept
    {   if (this != &other)
        {   if (_seq)
                kseq_destroy(_seq);
            if (_fp)
                gzclose(_fp);
            _fp = other._fp;
            _seq = other._seq;
            _read_result = other._read_result;
            other._fp = nullptr;
            other._seq = nullptr;
            other._read_result = 0;
        }
        return *this;
    }
    ~forward_stream()
    {   if (_seq)
            kseq_destroy(_seq);
        if (_fp)
            gzclose(_fp);
    }
    std::string_view id() const
    {   return std::string_view(_seq->name.s, _seq->name.l);
    }
    std::string_view description() const
    {   return std::string_view(_seq->comment.s, _seq->comment.l);
    }
    std::string_view quality() const
    {   return std::string_view(_seq->qual.s, _seq->qual.l);
    }
    std::string_view sequence() const
    {   return std::string_view(_seq->seq.s, _seq->seq.l);
    }

    SequenceType operator() () const
    {   SequenceType seq = (_read_result > 0)
        ?   SequenceType(_seq->seq.s)
        :   SequenceType();
        if (_seq->name.l)
            seq["_id"] = std::string(_seq->name.s);
        if (_seq->comment.l)
            seq["_desc"] = std::string(_seq->comment.s);
        if (_seq->qual.l)
            seq["_qs"] = std::string(_seq->qual.s);
        return seq;
    }

    iterator begin() { return iterator(this); }
    iterator end() { return iterator(); }
};

} // namespace gnx