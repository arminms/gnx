// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
#pragma once

#include <fmt/core.h>
#include <fmt/format.h>
#include <concepts>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>

#include <gnx/sq.hpp>
#include <gnx/io/fastaqz.hpp>
#include <gnx/memory.hpp>

namespace gnx {
//
/// @brief A 2-bit packed sequence class template.
///
/// Each nucleotide is stored using 2 bits (A=00, C=01, G=10, T=11), packing
/// 4 bases per byte.  Bit layout within each byte is MSB-first:
///   byte[i] = [base 4i | base 4i+1 | base 4i+2 | base 4i+3]
///             bits 7-6       5-4         3-2         1-0
///
/// @tparam ByteContainer  Container that stores packed bytes (value_type must
///                        be uint8_t).
/// @tparam Map            Map type used for tagged metadata storage.
template
<   typename ByteContainer = std::vector<uint8_t>
,   typename Map = std::unordered_map<std::string, std::any>
>
class packed_generic_sequence_2bit
{
    static_assert
    (   std::is_same_v<typename ByteContainer::value_type, uint8_t>
    ,   "ByteContainer::value_type must be uint8_t"
    );

    ByteContainer             _bytes; // packed bytes
    std::size_t                _size; // number of bases (not bytes)
    std::unique_ptr<Map>     _ptr_td; // pointer to tagged data

public:
    // ---- type aliases -------------------------------------------------------
    using value_type          = char; ///< Logical element type (nucleotide)
    using byte_type           = uint8_t;
    using size_type           = std::size_t;
    using difference_type     = std::ptrdiff_t;
    using byte_container_type = ByteContainer;
    using map_type            = Map;

    static constexpr size_type npos = static_cast<size_type>(-1);

    // ---- 2-bit encode/decode ------------------------------------------------

    /// Encode a nucleotide character to a 2-bit value.
    /// Unknown characters map to 0 (A).
    static constexpr uint8_t encode(char c) noexcept
    {   switch (c)
        {   case 'C': case 'c': return 0b01u;
            case 'G': case 'g': return 0b10u;
            case 'T': case 't': return 0b11u;
            default:            return 0b00u; // A / fallback
        }
    }

    /// Decode a 2-bit value to the corresponding nucleotide character.
    static constexpr char decode(uint8_t bits) noexcept
    {   constexpr char table[4] = {'A', 'C', 'G', 'T'};
        return table[bits & 0x03u];
    }

    // ---- helper: number of bytes needed for n bases -------------------------
    static constexpr size_type num_bytes(size_type n) noexcept
    {   return (n + 3u) >> 2u;   }

    // =========================================================================
    // proxy reference types
    // =========================================================================

    /// @brief Mutable proxy reference to a single packed base.
    class reference
    {   ByteContainer&   _c;
        size_type      _pos;
    public:
        constexpr reference(ByteContainer& c, size_type pos) noexcept
        :   _c(c)
        ,   _pos(pos)
        {}

        /// Read: returns the decoded nucleotide character.
        constexpr operator char() const noexcept
        {   const size_type byte_idx = _pos >> 2u;
            const int shift = 6 - ((_pos & 3u) << 1u);
            return decode((_c[byte_idx] >> shift) & 0x03u);
        }
        /// Write: encodes @a ch and stores the 2-bit value.
        constexpr reference& operator=(char ch) noexcept
        {   const size_type byte_idx = _pos >> 2u;
            const int shift = 6 - ((_pos & 3u) << 1u);
            _c[byte_idx] = static_cast<uint8_t>
            (   (_c[byte_idx] & ~(0x03u << shift))
            |   (encode(ch) << shift)
            );
            return *this;
        }
        /// Copy-assign from another proxy.
        constexpr reference& operator=(const reference& other) noexcept
        {   return *this = char(other);
        }
        [[nodiscard]]
        constexpr bool operator==(char rhs) const noexcept
        {   return char(*this) == rhs;
        }
        [[nodiscard]]
        constexpr bool operator==(const reference& rhs) const noexcept
        {   return char(*this) == char(rhs);
        }
    };

    /// @brief Read-only proxy reference to a single packed base.
    class const_reference
    {   const ByteContainer& _c;
        size_type          _pos;
    public:
        constexpr const_reference(const ByteContainer& c, size_type pos) noexcept
        :   _c(c)
        ,   _pos(pos)
        {}
        constexpr operator char() const noexcept
        {   const size_type byte_idx = _pos >> 2u;
            const int shift = 6 - ((_pos & 3u) << 1u);
            return decode((_c[byte_idx] >> shift) & 0x03u);
        }
        [[nodiscard]]
        constexpr bool operator==(char rhs) const noexcept
        {   return char(*this) == rhs;
        }
        [[nodiscard]]
        constexpr bool operator==(const const_reference& rhs) const noexcept
        {   return char(*this) == char(rhs);
        }
    };

    // =========================================================================
    // iterators
    // =========================================================================

    /// @brief Random-access iterator for packed_generic_sequence_2bit.
    class iterator
    {   ByteContainer* _c;
        size_type    _pos;
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = char;
        using difference_type   = std::ptrdiff_t;
        using pointer           = void;
        using reference         = packed_generic_sequence_2bit::reference;

        constexpr iterator() noexcept
        :   _c(nullptr)
        ,   _pos(0)
        {}
        constexpr iterator(ByteContainer* c, size_type pos) noexcept
        :   _c(c)
        ,   _pos(pos)
        {}

        constexpr reference operator*() const noexcept
        {   return reference(*_c, _pos);
        }
        constexpr reference operator[](difference_type n) const noexcept
        {   return reference(*_c, static_cast<size_type>(_pos + n));
        }

        constexpr iterator& operator++() noexcept
        {   ++_pos;
            return *this;
        }
        constexpr iterator  operator++(int) noexcept
        {   auto tmp = *this;
            ++_pos;
            return tmp;
        }
        constexpr iterator& operator--() noexcept
        {   --_pos;
            return *this;
        }
        constexpr iterator operator--(int) noexcept
        {   auto tmp = *this;
            --_pos;
            return tmp;
        }
        constexpr iterator& operator+=(difference_type n) noexcept
        {   _pos += n;
            return *this;
        }
        constexpr iterator& operator-=(difference_type n) noexcept
        {   _pos -= n;
            return *this;
        }
        constexpr iterator operator+(difference_type n) const noexcept
        {   return iterator(_c, static_cast<size_type>(_pos + n));
        }
        constexpr iterator operator-(difference_type n) const noexcept
        {   return iterator(_c, static_cast<size_type>(_pos - n));
        }
        constexpr difference_type operator-(const iterator& rhs) const noexcept
        {   return static_cast<difference_type>(_pos)
        -   static_cast<difference_type>(rhs._pos);
        }

        constexpr bool operator==(const iterator& rhs) const noexcept
        {   return _pos == rhs._pos;
        }
        constexpr bool operator!=(const iterator& rhs) const noexcept
        {   return _pos != rhs._pos;
        }
        constexpr bool operator< (const iterator& rhs) const noexcept
        {   return _pos <  rhs._pos;
        }
        constexpr bool operator<=(const iterator& rhs) const noexcept
        {   return _pos <= rhs._pos;
        }
        constexpr bool operator> (const iterator& rhs) const noexcept
        {   return _pos >  rhs._pos;
        }
        constexpr bool operator>=(const iterator& rhs) const noexcept
        {   return _pos >= rhs._pos;
        }
    };

    friend constexpr iterator operator+
    (   iterator::difference_type n
    ,   const iterator& it
    )   noexcept
    {   return it + n;
    }

    /// @brief Random-access const iterator for packed_generic_sequence_2bit.
    class const_iterator
    {   const ByteContainer* _c;
        size_type          _pos;
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = char;
        using difference_type   = std::ptrdiff_t;
        using pointer           = void;
        using reference         = packed_generic_sequence_2bit::const_reference;

        constexpr const_iterator () noexcept
        :   _c(nullptr)
        ,   _pos(0)
        {}
        constexpr const_iterator (const ByteContainer* c, size_type pos) noexcept
        :   _c(c)
        ,   _pos(pos)
        {}
        // implicit conversion from iterator
        constexpr const_iterator (const iterator& it) noexcept
        :   _c(it._c)
        ,   _pos(it._pos)
        {}

        constexpr reference operator* () const noexcept
        {   return const_reference(*_c, _pos);
        }
        constexpr reference operator[] (difference_type n) const noexcept
        {   return const_reference(*_c, static_cast<size_type>(_pos + n));
        }

        constexpr const_iterator& operator++ () noexcept
        {   ++_pos;
            return *this;
        }
        constexpr const_iterator operator++ (int) noexcept
        {   auto t = *this;
            ++_pos;
            return t;
        }
        constexpr const_iterator& operator-- () noexcept
        {   --_pos;
            return *this;
        }
        constexpr const_iterator operator-- (int) noexcept
        {   auto t = *this;
            --_pos;
            return t;
        }
        constexpr const_iterator& operator+= (difference_type n) noexcept
        {   _pos += n;
            return *this;
        }
        constexpr const_iterator& operator-= (difference_type n) noexcept
        {   _pos -= n;
            return *this;
        }
        constexpr const_iterator operator+ (difference_type n) const noexcept
        {   return const_iterator(_c, static_cast<size_type>(_pos + n));
        }
        constexpr const_iterator operator- (difference_type n) const noexcept
        {   return const_iterator(_c, static_cast<size_type>(_pos - n));
        }
        constexpr difference_type operator- (const const_iterator& rhs)
        const noexcept
        {   return static_cast<difference_type>(_pos)
        -   static_cast<difference_type>(rhs._pos);
        }

        constexpr bool operator== (const const_iterator& rhs) const noexcept
        {   return _pos == rhs._pos;
        }
        constexpr bool operator!= (const const_iterator& rhs) const noexcept
        {   return _pos != rhs._pos;
        }
        constexpr bool operator<  (const const_iterator& rhs) const noexcept
        {   return _pos <  rhs._pos;
        }
        constexpr bool operator<= (const const_iterator& rhs) const noexcept
        {   return _pos <= rhs._pos;
        }
        constexpr bool operator>  (const const_iterator& rhs) const noexcept
        {   return _pos >  rhs._pos;
        }
        constexpr bool operator>= (const const_iterator& rhs) const noexcept
        {   return _pos >= rhs._pos;
        }
    };

    friend constexpr const_iterator operator+
    (   const_iterator::difference_type n
    ,   const const_iterator& it
    ) noexcept
    {   return it + n;
    }

    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    // =========================================================================
    // constructors
    // =========================================================================

    /// Default constructor. Constructs an empty packed sequence.
    packed_generic_sequence_2bit() noexcept
    :   _bytes()
    ,   _size(0)
    ,   _ptr_td()
    {}
    /// Constructs a packed sequence from a string view (only ACGT recognized).
    explicit packed_generic_sequence_2bit(std::string_view sv)
    :   _bytes(num_bytes(sv.size()), uint8_t{0})
    ,   _size(sv.size())
    ,   _ptr_td()
    {   for (size_type i = 0; i < _size; ++i)
            _set_base(i, sv[i]);
    }
    /// Constructs a packed sequence with @a count bases, all initialized to 'A'.
    explicit packed_generic_sequence_2bit(size_type count)
    :   _bytes(num_bytes(count)
    ,   uint8_t{0})
    ,   _size(count)
    ,   _ptr_td()
    {}
    /// Constructs a packed sequence with @a count bases each initialized to @a base.
    packed_generic_sequence_2bit(size_type count, char base)
    :   _bytes(num_bytes(count)
    ,   uint8_t{0})
    ,   _size(count)
    ,   _ptr_td()
    {   const uint8_t bits = encode(base);
        // fill all full bytes with replicated 2-bit pattern
        if (bits != 0)
        {   const uint8_t fill =
                static_cast<uint8_t>((bits << 6) | (bits << 4) | (bits << 2) | bits);
            std::fill(_bytes.begin(), _bytes.end(), fill);
            // clear any padding bits in the last byte
            _clear_padding();
        }
    }

    /// Constructs a packed sequence from a range of input iterator of chars.
    template<typename InputIt>
    requires std::input_iterator<InputIt>
    &&  std::is_convertible_v<typename std::iterator_traits<InputIt>::value_type, char>
    packed_generic_sequence_2bit(InputIt first, InputIt last)
    :   _bytes()
    ,   _size(0)
    ,   _ptr_td()
    {   for (auto it = first; it != last; ++it, ++_size)
        {   if (_size % 4 == 0)
                _bytes.push_back(0);
            _set_base(_size, *it);
        }
    }
    /// Constructs a packed sequence from an initializer list.
    packed_generic_sequence_2bit(std::initializer_list<char> init)
    :   packed_generic_sequence_2bit(init.begin(), init.end())
    {}
    /// Constructs a packed sequence from a @a generic_sequence (any Container/Map).
    template<typename SqContainer, typename SqMap>
    explicit packed_generic_sequence_2bit(const generic_sequence<SqContainer, SqMap>& sq)
    :   _bytes(num_bytes(sq.size()), uint8_t{0})
    ,   _size(sq.size())
    ,   _ptr_td()
    {   for (size_type i = 0; i < _size; ++i)
            _set_base(i, sq[i]);
        // copy tagged data
        if (sq._has_td())
            _ptr_td = std::make_unique<Map>(*sq._get_td());
    }
    /// Copy constructor.
    packed_generic_sequence_2bit(const packed_generic_sequence_2bit& other)
    :   _bytes(other._bytes)
    ,   _size(other._size)
    ,   _ptr_td(other._ptr_td ? std::make_unique<Map>(*other._ptr_td) : nullptr)
    {}
    /// Move constructor.
    packed_generic_sequence_2bit(packed_generic_sequence_2bit&& other) noexcept
    :   _bytes(std::move(other._bytes))
    ,   _size(other._size)
    ,   _ptr_td(std::move(other._ptr_td))
    {   other._size = 0;
    }

    // =========================================================================
    // assignment operators
    // =========================================================================

    packed_generic_sequence_2bit& operator= (const packed_generic_sequence_2bit& other)
    {   _bytes  = other._bytes;
        _size   = other._size;
        _ptr_td = other._ptr_td ? std::make_unique<Map>(*other._ptr_td) : nullptr;
        return *this;
    }

    packed_generic_sequence_2bit& operator=(packed_generic_sequence_2bit&& other) noexcept
    {   _bytes  = std::move(other._bytes);
        _size   = other._size;
        _ptr_td = std::move(other._ptr_td);
        other._size = 0;
        return *this;
    }

    packed_generic_sequence_2bit& operator=(std::string_view sv)
    {   _size = sv.size();
        _bytes.assign(num_bytes(_size), uint8_t{0});
        for (size_type i = 0; i < _size; ++i)
            _set_base(i, sv[i]);
        return *this;
    }

    packed_generic_sequence_2bit& operator=(std::initializer_list<char> init)
    {   *this = packed_generic_sequence_2bit(init.begin(), init.end());
        return *this;
    }

    // =========================================================================
    // conversion to generic_sequence
    // =========================================================================

    /// Converts this packed sequence to a @a generic_sequence<SqContainer, SqMap>.
    template
    <   typename SqContainer = std::vector<char>
    ,   typename SqMap = Map
    >
    [[nodiscard]] generic_sequence<SqContainer, SqMap> to_sq() const
    {   generic_sequence<SqContainer, SqMap> result(_size);
        for (size_type i = 0; i < _size; ++i)
            result[i] = get_base(i);
        if (_ptr_td)
            for (const auto& [tag, val] : *_ptr_td)
                result[tag] = val;
        return result;
    }

    // =========================================================================
    // iterators
    // =========================================================================

    iterator begin()  noexcept { return iterator(&_bytes, 0);      }
    iterator end()    noexcept { return iterator(&_bytes, _size);   }
    const_iterator begin()  const noexcept { return const_iterator(&_bytes, 0);     }
    const_iterator end()    const noexcept { return const_iterator(&_bytes, _size); }
    const_iterator cbegin() const noexcept { return const_iterator(&_bytes, 0);     }
    const_iterator cend()   const noexcept { return const_iterator(&_bytes, _size); }
    reverse_iterator rbegin()  noexcept { return reverse_iterator(end());   }
    reverse_iterator rend()    noexcept { return reverse_iterator(begin()); }
    const_reverse_iterator rbegin()  const noexcept { return const_reverse_iterator(end());   }
    const_reverse_iterator rend()    const noexcept { return const_reverse_iterator(begin()); }
    const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend());  }
    const_reverse_iterator crend()   const noexcept { return const_reverse_iterator(cbegin());}

    // =========================================================================
    // capacity
    // =========================================================================

    /// Returns true if the sequence has no bases and no tagged data.
    [[nodiscard]] bool empty() const noexcept
    {   return (_size == 0 && (!_ptr_td || _ptr_td->empty()));
    }
    /// Returns the number of bases in the sequence.
    [[nodiscard]] size_type size() const noexcept
    {   return _size;
    }
    /// Returns the number of bytes used to store the packed bases.
    [[nodiscard]] size_type byte_size() const noexcept
    {   return _bytes.size();
    }
    /// Returns the approximate size in memory (in bytes) of the packed sequence
    /// including its tagged data.
    [[nodiscard]] size_type size_in_memory() const noexcept
    {   size_type mem = sizeof(ByteContainer) + _bytes.capacity();
        if (_ptr_td)
        {   mem += sizeof(Map);
            for (const auto& [tag, data] : *_ptr_td)
                mem += tag.capacity() * sizeof(char);
        }
        return mem;
    }

    // =========================================================================
    // element access
    // =========================================================================

    /// Returns a mutable proxy reference to the base at position @a pos.
    reference operator[] (size_type pos)
    {   return reference(_bytes, pos);
    }
    /// Returns a read-only proxy reference to the base at position @a pos.
    const_reference operator[](size_type pos) const
    {   return const_reference(_bytes, pos);
    }
    /// Returns a mutable proxy with bounds checking.
    reference at(size_type pos)
    {   if (pos >= _size)
            throw std::out_of_range("gnx::psq2: pos >= size()");
        return reference(_bytes, pos);
    }
    /// Returns a read-only proxy with bounds checking.
    const_reference at(size_type pos) const
    {   if (pos >= _size)
            throw std::out_of_range("gnx::psq2: pos >= size()");
        return const_reference(_bytes, pos);
    }
    /// Returns the decoded base at position @a pos (equivalent to char(operator[](pos))).
    [[nodiscard]] char get_base(size_type pos) const noexcept
    {   return char(const_reference(_bytes, pos));
    }
    /// Returns a raw pointer to the underlying packed byte array.
#if defined(__CUDACC__) || defined(__HIPCC__)
    [[nodiscard]] const uint8_t* data() const noexcept
    {   return thrust::raw_pointer_cast(_bytes.data());
    }
    uint8_t* data() noexcept
    {   return thrust::raw_pointer_cast(_bytes.data());
    }
#else
    [[nodiscard]] const uint8_t* data() const noexcept
    {   return _bytes.data();
    }
    uint8_t* data() noexcept
    {   return _bytes.data();
    }
#endif

    // =========================================================================
    // managing tagged data
    // =========================================================================

    /// Returns true if tagged data with the specified @a tag exists.
    [[nodiscard]] bool has(std::string_view tag) const
    {   return
        (   _ptr_td
        ?   _ptr_td->find(std::string(tag)) != _ptr_td->end()
        :   false
        );
    }
    /// Returns a reference to the tagged data for @a tag.
    /// Creates a new entry if it does not exist.
    std::any& operator[](const std::string& tag)
    {   if (!_ptr_td) _ptr_td = std::make_unique<Map>();
        return (*_ptr_td)[tag];
    }
    std::any& operator[](std::string&& tag)
    {   if (!_ptr_td) _ptr_td = std::make_unique<Map>();
        return (*_ptr_td)[std::move(tag)];
    }
    /// Returns a const reference to the tagged data for @a tag.
    /// Throws std::out_of_range if the tag does not exist.
    const std::any& operator[](const std::string& tag) const
    {   if (!_ptr_td || _ptr_td->find(tag) == _ptr_td->end())
            throw std::out_of_range("gnx::psq2: tag not found -> " + tag);
        return _ptr_td->at(tag);
    }

    // =========================================================================
    // comparison operators
    // =========================================================================

    bool operator== (const packed_generic_sequence_2bit& rhs) const noexcept
    {   return _size == rhs._size && _bytes == rhs._bytes;
    }
    bool operator!= (const packed_generic_sequence_2bit& rhs) const noexcept
    {   return !(*this == rhs);
    }

    bool operator== (std::string_view sv) const noexcept
    {   if (_size != sv.size()) return false;
        for (size_type i = 0; i < _size; ++i)
            if (get_base(i) != sv[i]) return false;
        return true;
    }
    bool operator!=(std::string_view sv) const noexcept
    {   return !(*this == sv);
    }

    template<typename SqContainer, typename SqMap>
    bool operator== (const generic_sequence<SqContainer, SqMap>& rhs) const
    {   if (_size != rhs.size()) return false;
        for (size_type i = 0; i < _size; ++i)
            if (get_base(i) != rhs[i]) return false;
        return true;
    }
    template<typename SqContainer, typename SqMap>
    bool operator!= (const generic_sequence<SqContainer, SqMap>& rhs) const
    {   return !(*this == rhs);
    }

    // =========================================================================
    // file i/o
    // =========================================================================
    ///
    /// Loads a sequence from a file by its index using the provided
    /// @a read function object
    void load
    (   std::string_view filename
    ,   size_type ndx = 0
    ,   in::fast_aqz<packed_generic_sequence_2bit> read
    =   in::fast_aqz<packed_generic_sequence_2bit>()
    )
    {   *this = read(filename, ndx);
    }
    ///
    /// Loads a sequence from a file by its identifier using the provided
    /// @a read function object
    void load
    (   std::string_view filename
    ,   std::string_view id
    ,   in::fast_aqz<packed_generic_sequence_2bit> read
    =   in::fast_aqz<packed_generic_sequence_2bit>()
    )
    {   *this = read(filename, id);
    }
    ///
    /// Saves the sequence to a file using the provided write function object
    template<typename WriteFunc>
    void save
    (   std::string_view filename
    ,   WriteFunc write
    )
    {   write(filename, *this);
    }

    // =========================================================================
    // print / scan (binary stream format)
    // =========================================================================

    /// Returns the packed sequence and its tagged data as a formatted string.
    /// Format: <num_bases><packed_bytes>[#tag#|type|value...]
    [[nodiscard]] std::string print() const
    {   fmt::memory_buffer buf;
        fmt::format_to(std::back_inserter(buf), "{}", _size);
#if defined(__CUDACC__) || defined(__HIPCC__)
        if constexpr
        (   std::is_same_v<byte_container_type, thrust::device_vector<byte_type>>
        )
        {   universal_host_pinned_vector<byte_type> uhpv(_bytes);
            buf.append
            (   thrust::raw_pointer_cast(uhpv.data())
            ,   thrust::raw_pointer_cast(uhpv.data()) + uhpv.size()
            );
        }
        else if constexpr
        (   std::is_same_v<byte_container_type, thrust::universal_vector<byte_type>>
        )
        {   buf.append
            (   thrust::raw_pointer_cast(_bytes.data())
            ,   thrust::raw_pointer_cast(_bytes.data()) + _bytes.size()
            );
        }
#if defined(__HIPCC__)
        else if constexpr
        (   std::is_same_v<byte_container_type, gnx::unified_vector<byte_type>>
        )
        {   buf.append
            (   thrust::raw_pointer_cast(_bytes.data())
            ,   thrust::raw_pointer_cast(_bytes.data()) + _bytes.size()
            );
        }
#endif //__HIPCC__
        else
#endif  //__CUDACC__
            buf.append(_bytes.data(), _bytes.data() + _bytes.size());
        if (_ptr_td)
            for (const auto& [tag, data] : *_ptr_td)
            {   fmt::format_to(std::back_inserter(buf), "#{0}#", tag);
                if (const auto it = td_print_visitor.find(std::type_index(data.type()))
                ;   it != td_print_visitor.cend())
                    it->second(buf, data);
                else
                {   quote_with_delimiter(buf, "UNREGISTERED TYPE");
                    fmt::format_to(std::back_inserter(buf), "{{}}");
                }
            }
        return fmt::to_string(buf);
    }

    /// Prints to an output stream (for backward compatibility).
    void print(std::ostream& os) const
    {   os << print();
    }
    /// Scans the packed sequence and tagged data from the input stream @a is.
    void scan(std::istream& is)
    {   is >> _size;
        _bytes.resize(num_bytes(_size));
#if defined(__CUDACC__) || defined(__HIPCC__)
        if constexpr
        (   std::is_same_v<byte_container_type, thrust::device_vector<byte_type>>
        )
        {   universal_host_pinned_vector<byte_type> uhpv(_bytes.size());
            is.read
            (   reinterpret_cast<char*>(thrust::raw_pointer_cast(uhpv.data()))
            ,   static_cast<std::streamsize>(_bytes.size())
            );
            thrust::copy(uhpv.begin(), uhpv.end(), _bytes.begin());
        }
        else if constexpr
        (   std::is_same_v<byte_container_type, thrust::universal_vector<byte_type>>
        )
        {   is.read
            (   reinterpret_cast<char*>(thrust::raw_pointer_cast(_bytes.data()))
            ,   static_cast<std::streamsize>(_bytes.size())
            );
        }
#if defined(__HIPCC__)
        else if constexpr
        (   std::is_same_v<byte_container_type, gnx::unified_vector<byte_type>>
        )
        {   is.read
            (   reinterpret_cast<char*>(thrust::raw_pointer_cast(_bytes.data()))
            ,   static_cast<std::streamsize>(_bytes.size())
            );
        }
#endif //__HIPCC__
        else
#endif  //__CUDACC__
        is.read
        (   reinterpret_cast<char*>(_bytes.data())
        ,   static_cast<std::streamsize>(_bytes.size())
        );
        if (is.peek() == '#' && !_ptr_td)
            _ptr_td = std::make_unique<Map>();
        while (is.peek() == '#')
        {   std::string tag, type;
            std::any a;
            is >> std::quoted(tag, '#') >> std::quoted(type, '|');
            if
            (   const auto it = td_scan_visitor.find(type)
            ;   it != td_scan_visitor.cend()
            )
                it->second(is, a);
            else
                throw std::runtime_error
                (   fmt::format("gnx::psq2: unregistered type -> {}"
                ,   type)
                );
            (*_ptr_td)[tag] = a;
        }
    }

    // =========================================================================
    // internal helpers (exposed for generic_sequence interop)
    // =========================================================================

    /// @cond INTERNAL
    bool _has_td() const noexcept
    {   return static_cast<bool>(_ptr_td);
    }
    const Map* _get_td() const noexcept
    {   return _ptr_td.get();
    }
    /// @endcond

private:
    /// Set a single base at index @a pos to nucleotide @a ch.
    void _set_base(size_type pos, char ch) noexcept
    {   const size_type byte_idx = pos >> 2u;
        const int shift = 6 - (static_cast<int>(pos & 3u) << 1);
        _bytes[byte_idx] = static_cast<uint8_t>
        (   (_bytes[byte_idx] & ~(0x03u << shift))
        |   (encode(ch) << shift)
        );
    }
    /// Clear the unused padding bits in the last byte.
    void _clear_padding() noexcept
    {   if (_size == 0 || _bytes.empty()) return;
        const size_type remainder = _size & 3u;
        if (remainder == 0) return;
        const uint8_t mask = static_cast<uint8_t>(0xFFu << ((4u - remainder) << 1u));
        _bytes.back() &= mask;
    }
};

// -- free comparison operators ------------------------------------------------

template<typename ByteContainer, typename Map>
bool operator==
(   std::string_view lhs
,   const packed_generic_sequence_2bit<ByteContainer, Map>& rhs
)
{   return rhs == lhs;
}
template<typename ByteContainer, typename Map>
bool operator==
(   const char* lhs
,   const packed_generic_sequence_2bit<ByteContainer, Map>& rhs
)
{   return rhs == std::string_view(lhs);
}
template<typename ByteContainer, typename Map>
bool operator==
(   const packed_generic_sequence_2bit<ByteContainer, Map>& lhs
,   const char* rhs
)
{   return lhs == std::string_view(rhs);
}
template
<   typename SqContainer
,   typename SqMap
,   typename ByteContainer
,   typename PMap
>
bool operator==
(   const generic_sequence<SqContainer, SqMap>& lhs
,   const packed_generic_sequence_2bit<ByteContainer, PMap>& rhs
)
{   return rhs == lhs;
}

// -- i/o stream operators -----------------------------------------------------

template<typename ByteContainer, typename Map>
std::ostream& operator<<
(   std::ostream& os
,   const packed_generic_sequence_2bit<ByteContainer, Map>& s
)
{   s.print(os); return os;
}

template<typename ByteContainer, typename Map>
std::istream& operator>>
(   std::istream& is
,   packed_generic_sequence_2bit<ByteContainer, Map>& s
)
{   s.scan(is); return is;
}

// -- convenience alias --------------------------------------------------------

/// @brief Default 2-bit packed sequence over @c std::vector<uint8_t>.
using psq2 = packed_generic_sequence_2bit<>;

}   // namespace gnx

// -- string literal operator --------------------------------------------------

[[nodiscard]] inline
gnx::psq2 operator""_psq2 (const char* str, std::size_t)
{   return gnx::psq2(str);
}

// -- fmt formatter ------------------------------------------------------------

template<>
struct fmt::formatter<gnx::psq2> : fmt::formatter<std::string>
{   auto format(const gnx::psq2& s, format_context& ctx) const
    {   // Render as a plain nucleotide string for fmt output
        std::string out(s.size(), '\0');
        for (std::size_t i = 0; i < s.size(); ++i)
            out[i] = s.get_base(i);
        return fmt::formatter<std::string>::format(out, ctx);
    }
};

template<typename ByteContainer, typename Map>
struct fmt::formatter<gnx::packed_generic_sequence_2bit<ByteContainer, Map>>
:   fmt::formatter<std::string>
{   auto format
    (   const gnx::packed_generic_sequence_2bit<ByteContainer, Map>& s
    ,   format_context& ctx
    )   const
    {   std::string out(s.size(), '\0');
        for (std::size_t i = 0; i < s.size(); ++i)
            out[i] = s.get_base(i);
        return fmt::formatter<std::string>::format(out, ctx);
    }
};
