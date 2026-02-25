// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace gnx {

// forward declaration of generic_sequence
template<typename Container, typename Map>
class generic_sequence;

// forward declaration of packed_generic_sequence_2bit
template<typename ByteContainer, typename Map>
class packed_generic_sequence_2bit;

/// @brief A non-owning view over a sequence, similar to std::basic_string_view.
/// @tparam Container The container type used by the owning generic_sequence.
template<typename Container>
class generic_sequence_view
:   public std::ranges::view_interface<generic_sequence_view<Container>>
{
public:
    using value_type = typename Container::value_type;
    using size_type = typename Container::size_type;
    using difference_type = typename Container::difference_type;
    using const_pointer = const value_type*;
    using const_reference = const value_type&;
    using const_iterator = const_pointer;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using container_type = Container;

    static constexpr size_type npos = static_cast<size_type>(-1);

// -- constructors -------------------------------------------------------------
    constexpr generic_sequence_view() noexcept = default;

    constexpr generic_sequence_view
    (const value_type* data, size_type count) noexcept
    :   _data(data)
    ,   _size(count)
    {}

    template<typename Map>
    constexpr generic_sequence_view
    (const generic_sequence<Container, Map>& seq) noexcept
    :   _data(seq.data())
    ,   _size(seq.size())
    {}

// -- iterators ----------------------------------------------------------------
    constexpr const_iterator begin() const noexcept
    {   return _data;
    }
    constexpr const_iterator end() const noexcept
    {   return _data + _size;
    }
    constexpr const_iterator cbegin() const noexcept
    {   return begin();
    }
    constexpr const_iterator cend() const noexcept
    {   return end();
    }
    constexpr const_reverse_iterator rbegin() const noexcept
    {   return const_reverse_iterator(end());
    }
    constexpr const_reverse_iterator rend() const noexcept
    {   return const_reverse_iterator(begin());
    }
    constexpr const_reverse_iterator crbegin() const noexcept
    {   return rbegin();
    }
    constexpr const_reverse_iterator crend() const noexcept
    {   return rend();
    }

// -- capacity -----------------------------------------------------------------
    constexpr size_type size() const noexcept
    {   return _size;
    }
    [[nodiscard]] constexpr bool empty() const noexcept
    {   return _size == 0;
    }

// -- element access -----------------------------------------------------------
    constexpr const_reference operator[] (size_type pos) const
    {   return _data[pos];
    }
    constexpr const_reference at(size_type pos) const
    {   if (pos >= _size)
            throw std::out_of_range("gnx::sq_view: pos >= size()");
        return _data[pos];
    }
    constexpr const_reference front() const
    {   return _data[0];
    }
    constexpr const_reference back() const
    {   return _data[_size - 1];
    }

    constexpr const_pointer data() const noexcept
    {   return _data;
    }

// -- modifiers ----------------------------------------------------------------

    constexpr void remove_prefix(size_type n)
    {   if (n > _size)
            throw std::out_of_range("gnx::sq_view: remove_prefix overflow");
        _data += n;
        _size -= n;
    }
    constexpr void remove_suffix(size_type n)
    {   if (n > _size)
            throw std::out_of_range("gnx::sq_view: remove_suffix overflow");
        _size -= n;
    }

// -- operations ---------------------------------------------------------------

    constexpr generic_sequence_view subseq
    (size_type pos, size_type count = npos) const
    {   if (pos > _size)
            throw std::out_of_range("gnx::sq_view: pos > size()");
        const size_type rlen = std::min(count, static_cast<size_type>(_size - pos));
        return generic_sequence_view(_data + pos, rlen);
    }

private:
    const_pointer _data = nullptr;
    size_type _size = 0;
};

// -- comparison operators -----------------------------------------------------
    template<typename Container>
    constexpr bool operator==
    (   const generic_sequence_view<Container>& lhs
    ,   const generic_sequence_view<Container>& rhs
    )
    {   return lhs.size() == rhs.size()
        && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    // Note: comparisons with generic_sequence are provided generically in sq.hpp.
    // Avoid defining overlapping overloads here to prevent ambiguity.

    template<typename Container>
    constexpr bool operator!=
    (   const generic_sequence_view<Container>& lhs
    ,   const generic_sequence_view<Container>& rhs
    )
    {   return ! (lhs == rhs);
    }

    // Inequality with generic_sequence is covered by the generic operators in sq.hpp.

    template<typename Container>
    constexpr
    std::enable_if_t<std::is_same_v<typename Container::value_type, char>, bool>
    operator==
    (   const generic_sequence_view<Container>& lhs
    ,   std::string_view rhs
    )
    {   return lhs.size()
    ==  rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template<typename Container>
    constexpr
    std::enable_if_t<std::is_same_v<typename Container::value_type, char>, bool>
    operator==
    (std::string_view lhs, const generic_sequence_view<Container>& rhs)
    {   return rhs == lhs;
    }

    template<typename Container>
    constexpr
    std::enable_if_t<std::is_same_v<typename Container::value_type, char>, bool>
    operator!=
    (const generic_sequence_view<Container>& lhs, std::string_view rhs)
    {   return ! (lhs == rhs);
    }

    template<typename Container>
    constexpr
    std::enable_if_t<std::is_same_v<typename Container::value_type, char>, bool>
    operator!=
    (std::string_view lhs, const generic_sequence_view<Container>& rhs)
    {   return ! (lhs == rhs);
    }

    // Convenience overloads for C-string literals
    template<typename Container>
    constexpr
    std::enable_if_t<std::is_same_v<typename Container::value_type, char>, bool>
    operator==
    (const generic_sequence_view<Container>& lhs, const char* rhs)
    {   return lhs == std::string_view(rhs);
    }

    template<typename Container>
    constexpr
    std::enable_if_t<std::is_same_v<typename Container::value_type, char>, bool>
    operator==
    (const char* lhs, const generic_sequence_view<Container>& rhs)
    {   return std::string_view(lhs) == rhs;
    }

    template<typename Container>
    constexpr
    std::enable_if_t<std::is_same_v<typename Container::value_type, char>, bool>
    operator!=
    (const generic_sequence_view<Container>& lhs, const char* rhs)
    {   return ! (lhs == rhs);
    }

    template<typename Container>
    constexpr
    std::enable_if_t<std::is_same_v<typename Container::value_type, char>, bool>
    operator!=
    (const char* lhs, const generic_sequence_view<Container>& rhs)
    {   return ! (lhs == rhs);
    }

// -- aliases ------------------------------------------------------------------
    using sq_view = generic_sequence_view<std::vector<char>>;

// =============================================================================
// packed_generic_sequence_2bit_view
// =============================================================================

/// @brief A non-owning, zero-copy view over a 2-bit packed sequence.
///
/// Provides read-only access to a contiguous sub-range of a
/// packed_generic_sequence_2bit without copying the underlying packed bytes.
/// The view stores a raw pointer to the packed byte array, a base offset, and
/// the number of bases, so subsequence views are O(1) to construct.
///
/// @tparam ByteContainer The byte container type used by the owning
///                       packed_generic_sequence_2bit (determines the
///                       pointer type used for data access).
template<typename ByteContainer = std::vector<uint8_t>>
class packed_generic_sequence_2bit_view
:   public std::ranges::view_interface<packed_generic_sequence_2bit_view<ByteContainer>>
{
public:
    using value_type          = char;
    using byte_type           = uint8_t;
    using size_type           = std::size_t;
    using difference_type     = std::ptrdiff_t;
    using byte_container_type = ByteContainer;

    static constexpr size_type npos = static_cast<size_type>(-1);

    // ---- 2-bit encode/decode (mirrors packed_generic_sequence_2bit) ---------

    /// Encode a nucleotide character to a 2-bit value.
    static constexpr uint8_t encode(char c) noexcept
    {   switch (c)
        {   case 'C': case 'c': return 0b01u;
            case 'G': case 'g': return 0b10u;
            case 'T': case 't': return 0b11u;
            default:            return 0b00u;
        }
    }

    /// Decode a 2-bit value to the corresponding nucleotide character.
    static constexpr char decode(uint8_t bits) noexcept
    {   constexpr char table[4] = {'A', 'C', 'G', 'T'};
        return table[bits & 0x03u];
    }

    // =========================================================================
    // const_reference proxy
    // =========================================================================

    /// @brief Read-only proxy reference to a single packed base.
    class const_reference
    {   const uint8_t* _data;
        size_type       _pos;  ///< global base position inside the byte array
    public:
        constexpr const_reference(const uint8_t* data, size_type pos) noexcept
        :   _data(data)
        ,   _pos(pos)
        {}

        constexpr operator char() const noexcept
        {   const size_type byte_idx = _pos >> 2u;
            const int shift = 6 - ((_pos & 3u) << 1u);
            return decode((_data[byte_idx] >> shift) & 0x03u);
        }

        [[nodiscard]]
        constexpr bool operator==(char rhs) const noexcept
        {   return char(*this) == rhs;
        }
        [[nodiscard]]
        constexpr bool operator==(const const_reference& rhs) const noexcept
        {   return char(*this) == char(rhs);
        }
        [[nodiscard]]
        constexpr bool operator!=(char rhs) const noexcept
        {   return char(*this) != rhs;
        }
        [[nodiscard]]
        constexpr bool operator!=(const const_reference& rhs) const noexcept
        {   return char(*this) != char(rhs);
        }
    };

    // =========================================================================
    // const_iterator
    // =========================================================================

    /// @brief Random-access const iterator for packed_generic_sequence_2bit_view.
    class const_iterator
    {   const uint8_t* _data;
        size_type       _pos;  ///< global base position inside the byte array
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = char;
        using difference_type   = std::ptrdiff_t;
        using pointer           = void;
        using reference         = packed_generic_sequence_2bit_view::const_reference;

        constexpr const_iterator() noexcept
        :   _data(nullptr)
        ,   _pos(0)
        {}

        constexpr const_iterator(const uint8_t* data, size_type pos) noexcept
        :   _data(data)
        ,   _pos(pos)
        {}

        constexpr reference operator*()  const noexcept
        {   return const_reference(_data, _pos);
        }
        constexpr reference operator[](difference_type n) const noexcept
        {   return const_reference(_data, static_cast<size_type>(_pos + n));
        }

        constexpr const_iterator& operator++() noexcept
        {   ++_pos; return *this;
        }
        constexpr const_iterator operator++(int) noexcept
        {   auto t = *this; ++_pos; return t;
        }
        constexpr const_iterator& operator--() noexcept
        {   --_pos; return *this;
        }
        constexpr const_iterator operator--(int) noexcept
        {   auto t = *this; --_pos; return t;
        }
        constexpr const_iterator& operator+=(difference_type n) noexcept
        {   _pos = static_cast<size_type>(_pos + n); return *this;
        }
        constexpr const_iterator& operator-=(difference_type n) noexcept
        {   _pos = static_cast<size_type>(_pos - n); return *this;
        }
        constexpr const_iterator operator+(difference_type n) const noexcept
        {   return const_iterator(_data, static_cast<size_type>(_pos + n));
        }
        constexpr const_iterator operator-(difference_type n) const noexcept
        {   return const_iterator(_data, static_cast<size_type>(_pos - n));
        }
        constexpr difference_type operator-(const const_iterator& rhs) const noexcept
        {   return static_cast<difference_type>(_pos)
        -   static_cast<difference_type>(rhs._pos);
        }

        constexpr bool operator==(const const_iterator& rhs) const noexcept
        {   return _pos == rhs._pos;
        }
        constexpr bool operator!=(const const_iterator& rhs) const noexcept
        {   return _pos != rhs._pos;
        }
        constexpr bool operator< (const const_iterator& rhs) const noexcept
        {   return _pos <  rhs._pos;
        }
        constexpr bool operator<=(const const_iterator& rhs) const noexcept
        {   return _pos <= rhs._pos;
        }
        constexpr bool operator> (const const_iterator& rhs) const noexcept
        {   return _pos >  rhs._pos;
        }
        constexpr bool operator>=(const const_iterator& rhs) const noexcept
        {   return _pos >= rhs._pos;
        }
    };

    friend constexpr const_iterator operator+
    (   typename const_iterator::difference_type n
    ,   const const_iterator& it
    ) noexcept
    {   return it + n;
    }

    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    // =========================================================================
    // constructors
    // =========================================================================

    /// Default constructor — empty view.
    constexpr packed_generic_sequence_2bit_view() noexcept = default;

    /// Constructs a view over @a count bases starting at global base @a offset
    /// in the packed byte array pointed to by @a data.
    constexpr packed_generic_sequence_2bit_view
    (   const uint8_t* data
    ,   size_type       offset
    ,   size_type       count
    )   noexcept
    :   _data(data)
    ,   _offset(offset)
    ,   _size(count)
    {}

    /// Constructs a full view over an existing packed_generic_sequence_2bit.
    template<typename Map>
    constexpr packed_generic_sequence_2bit_view
    (   const packed_generic_sequence_2bit<ByteContainer, Map>& seq
    )   noexcept
    :   _data(seq.data())
    ,   _offset(0)
    ,   _size(seq.size())
    {}

    // =========================================================================
    // iterators
    // =========================================================================

    constexpr const_iterator begin()  const noexcept
    {   return const_iterator(_data, _offset);
    }
    constexpr const_iterator end()    const noexcept
    {   return const_iterator(_data, _offset + _size);
    }
    constexpr const_iterator cbegin() const noexcept
    {   return begin();
    }
    constexpr const_iterator cend() const noexcept
    {   return end();
    }

    constexpr const_reverse_iterator rbegin()  noexcept
    {   return const_reverse_iterator(end());
    }
    constexpr const_reverse_iterator rend() const noexcept
    {   return const_reverse_iterator(begin());
    }
    constexpr const_reverse_iterator crbegin() const noexcept
    {   return rbegin();
    }
    constexpr const_reverse_iterator crend() const noexcept
    {   return rend();
    }

    // =========================================================================
    // capacity
    // =========================================================================

    constexpr size_type size() const noexcept { return _size; }

    [[nodiscard]] constexpr bool empty() const noexcept { return _size == 0; }

    // =========================================================================
    // element access
    // =========================================================================

    constexpr const_reference operator[](size_type pos) const
    {   return const_reference(_data, _offset + pos);
    }
    constexpr const_reference at(size_type pos) const
    {   if (pos >= _size)
            throw std::out_of_range("gnx::psq2_view: pos >= size()");
        return const_reference(_data, _offset + pos);
    }
    constexpr const_reference front() const
    {   return const_reference(_data, _offset);
    }
    constexpr const_reference back() const
    {   return const_reference(_data, _offset + _size - 1u);
    }

    /// Returns the decoded base at position @a pos
    /// (same as @c char(operator[](pos))).
    [[nodiscard]] constexpr char get_base(size_type pos) const noexcept
    {   return char(const_reference(_data, _offset + pos));
    }

    /// Returns a raw pointer to the beginning of the underlying packed byte array.
    /// @note The first base of this view starts at global base @c offset(),
    ///       not necessarily at byte 0 of the returned pointer.
    [[nodiscard]]
    constexpr const uint8_t* data() const noexcept { return _data; }

    /// Returns the global base-index offset of the first base of this view.
    [[nodiscard]]
    constexpr size_type offset() const noexcept { return _offset; }

    // =========================================================================
    // operations
    // =========================================================================

    /// Returns a sub-view starting at @a pos with at most @a count bases.
    constexpr packed_generic_sequence_2bit_view subseq
    (   size_type pos
    ,   size_type count = npos
    )   const
    {   if (pos > _size)
            throw std::out_of_range("gnx::psq2_view: pos > size()");
        const size_type rlen = std::min(count, _size - pos);
        return packed_generic_sequence_2bit_view(_data, _offset + pos, rlen);
    }

private:
    const uint8_t* _data   = nullptr;
    size_type      _offset = 0;
    size_type      _size   = 0;
};

// -- comparison operators for packed_generic_sequence_2bit_view ---------------

    template<typename ByteContainer>
    constexpr bool operator==
    (   const packed_generic_sequence_2bit_view<ByteContainer>& lhs
    ,   const packed_generic_sequence_2bit_view<ByteContainer>& rhs
    )
    {   if (lhs.size() != rhs.size()) return false;
        for (std::size_t i = 0; i < lhs.size(); ++i)
            if (char(lhs[i]) != char(rhs[i])) return false;
        return true;
    }

    template<typename ByteContainer>
    constexpr bool operator!=
    (   const packed_generic_sequence_2bit_view<ByteContainer>& lhs
    ,   const packed_generic_sequence_2bit_view<ByteContainer>& rhs
    )
    {   return !(lhs == rhs);
    }

    template<typename ByteContainer>
    constexpr bool operator==
    (   const packed_generic_sequence_2bit_view<ByteContainer>& lhs
    ,   std::string_view rhs
    )
    {   if (lhs.size() != rhs.size()) return false;
        for (std::size_t i = 0; i < lhs.size(); ++i)
            if (char(lhs[i]) != rhs[i]) return false;
        return true;
    }

    template<typename ByteContainer>
    constexpr bool operator==
    (   std::string_view lhs
    ,   const packed_generic_sequence_2bit_view<ByteContainer>& rhs
    )
    {   return rhs == lhs;
    }

    template<typename ByteContainer>
    constexpr bool operator!=
    (   const packed_generic_sequence_2bit_view<ByteContainer>& lhs
    ,   std::string_view rhs
    )
    {   return !(lhs == rhs);
    }

    template<typename ByteContainer>
    constexpr bool operator!=
    (   std::string_view lhs
    ,   const packed_generic_sequence_2bit_view<ByteContainer>& rhs
    )
    {   return !(rhs == lhs);
    }

    template<typename ByteContainer>
    constexpr bool operator==
    (   const packed_generic_sequence_2bit_view<ByteContainer>& lhs
    ,   const char* rhs
    )
    {   return lhs == std::string_view(rhs);
    }

    template<typename ByteContainer>
    constexpr bool operator==
    (   const char* lhs
    ,   const packed_generic_sequence_2bit_view<ByteContainer>& rhs
    )
    {   return std::string_view(lhs) == rhs;
    }

    template<typename ByteContainer>
    constexpr bool operator!=
    (   const packed_generic_sequence_2bit_view<ByteContainer>& lhs
    ,   const char* rhs
    )
    {   return !(lhs == rhs);
    }

    template<typename ByteContainer>
    constexpr bool operator!=
    (   const char* lhs
    ,   const packed_generic_sequence_2bit_view<ByteContainer>& rhs
    )
    {   return !(rhs == lhs);
    }

// -- aliases ------------------------------------------------------------------
    using psq2_view = packed_generic_sequence_2bit_view<>;

}   // end gnx namespace

// Enable std::ranges::view concept for generic_sequence_view
namespace std::ranges
{   template<typename Container>
    inline constexpr bool enable_view<gnx::generic_sequence_view<Container>> = true;

    template<typename ByteContainer>
    inline constexpr bool enable_view<gnx::packed_generic_sequence_2bit_view<ByteContainer>> = true;
}
