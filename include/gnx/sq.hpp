// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
#pragma once

#include <concepts>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <initializer_list>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <any>
#include <memory>
#include <typeindex>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
    #include <thrust/host_vector.h>
    #include <thrust/device_vector.h>
    #include <thrust/universal_vector.h>
    #include <thrust/memory.h>
#endif // __CUDACC__

#include <gnx/commons.hpp>
#include <gnx/concepts.hpp>
#include <gnx/views.hpp>
#include <gnx/visitor.hpp>
#include <gnx/backend/forward_stream.hpp>
#include <gnx/backend/virtual_vector.hpp>
#include <gnx/memory.hpp>

namespace gnx {
//
/// @brief A generic sequence class template with tagged data support.
/// @tparam Container The underlying container type to hold the sequence.
/// @tparam Map The type of the map used for tagged data storage.
template
<   typename Container
,   typename Map = std::unordered_map<std::string, std::any>
>
class generic_sequence
{   Container                _sq;  // sequence
    std::unique_ptr<Map> _ptr_td;  // pointer to tagged data

public:
    using value_type = typename Container::value_type;
    using size_type = typename Container::size_type;
    using difference_type = typename Container::difference_type;
    using reference = typename Container::reference;
    using const_reference = typename Container::const_reference;
    using iterator = typename Container::iterator;
    using const_iterator = typename Container::const_iterator;
    using reverse_iterator = typename Container::reverse_iterator;
    using const_reverse_iterator = typename Container::const_reverse_iterator;
    using container_type = Container;
    using map_type = Map;
    using self_type = generic_sequence<Container, Map>;

    static constexpr size_type npos = static_cast<size_type>(-1);

// -- constructors -------------------------------------------------------------
    ///
    /// Default constructor. Constructs an empty sequence.
    generic_sequence() noexcept
    :   _sq()
    ,   _ptr_td()
    {}
    ///
    /// @brief Constructs a sequence from a string view. If the string view is a
    /// file path, the constructor attempts to read the sequence from the file.
    /// If the string view is not a file path, it is treated as a direct sequence
    /// input.
    /// @param sv The string view representing the sequence or a file path.
    /// @param ndx The index of the sequence to read from the file if the string
    /// view is a file path. Default is 0 (the first sequence).
    explicit generic_sequence(std::string_view sv, size_type ndx = 0)
    {   if  // if the sv is a view over a filename, attempt to read it as such
        (std::filesystem::path(std::string(sv)).has_extension())
        {   if (std::filesystem::exists(std::string(sv) + ".fai"))
            {   gnx::virtual_vector<self_type> vv{sv};
                if (ndx < vv.size())
                    *this = vv[ndx];
                else
                    log_error("invalid sequence index");
            }
            else
            {   size_t count = 0;
                gnx::forward_stream<self_type> stream(sv);
                auto it = stream.begin();
                for (; it != stream.end() && count < ndx; ++it, ++count);
                if (it != stream.end())
                    *this = stream();
                else
                    log_error("invalid sequence index");
            }
        }
        else // otherwise, treat it as a view over a sequence
        {   _sq.assign(std::begin(sv), std::end(sv));
        }
    }
    ///
    /// @brief Constructs a sequence from a string view and an ID. If the string
    /// view is a file path, the constructor attempts to read the sequence with
    /// the matching ID from the file. If the string view is not a file path, it
    /// is treated as a direct sequence input.
    /// @param sv The string view representing the sequence or a file path.
    /// @param id The ID of the sequence to read from the file if the string is
    /// a file path.
    generic_sequence(std::string_view sv, std::string_view id)
    {   if  // if the sv is a view over a filename, attempt to read it as such
        (std::filesystem::path(std::string(sv)).has_extension())
        {   if (std::filesystem::exists(std::string(sv) + ".fai"))
            {   gnx::virtual_vector<self_type> vv{sv};
                for (size_type i = 0; i < vv.size(); ++i)
                {   if (vv.entry(i).name == id)
                    {   *this = vv[i];
                        return;
                    }
                }
                log_error("no sequence with matching ID found");
            }
            else
            {   gnx::forward_stream<self_type> stream(sv);
                auto it = stream.begin();
                for (; it != stream.end() && stream.id() != id; ++it);
                if (it != stream.end())
                    *this = stream();
                else
                    log_error("no sequence with matching ID found");
            }
        }
        else // otherwise, treat it as a view over a sequence
        {   _sq.assign(std::begin(sv), std::end(sv));
        }
    }
    ///
    /// Constructs a sequence with @a count residues.
    /// @param count The number of residues in the sequence.
    generic_sequence(size_type count)
    :   _sq(count)
    ,   _ptr_td()
    {}
    ///
    /// Constructs a sequence with @a count residues, each initialized to
    /// @a value (default is 'A' (ASCII 65)).
    /// @param count The number of residues in the sequence.
    /// @param value The residue value to initialize each position with.
#if defined(__CUDACC__) || defined(__HIPCC__)
    generic_sequence(size_type count, const_reference value)
    requires (!std::is_same_v<Container, thrust::device_vector<value_type>>)
#if defined(__HIPCC__)
    || (!std::is_same_v<Container, gnx::unified_vector<value_type>>)
#endif //__HIPCC__
    :   _sq(count, value)
    ,   _ptr_td()
    {}
#else
    generic_sequence(size_type count, const_reference value)
    :   _sq(count, value)
    ,   _ptr_td()
    {}
#endif //__CUDACC__

// test

//<-------------------
    ///
    /// @brief Constructs a sequence from a sequence view.
    /// @param sv The sequence view to construct the sequence from.
#if defined(__CUDACC__) || defined(__HIPCC__)
    explicit generic_sequence(generic_sequence_view<Container> sv)
    requires (!std::is_same_v<Container, thrust::device_vector<value_type>>)
    :   _sq(std::begin(sv), std::end(sv))
    ,   _ptr_td()
    {}
#else
    explicit generic_sequence(generic_sequence_view<Container> sv)
    :   _sq(std::begin(sv), std::end(sv))
    ,   _ptr_td()
    {}
#endif //__CUDACC__
//<-------------------

    ///
    template<typename InputIt>
    requires std::input_iterator<InputIt>
    &&  std::is_same_v<typename std::iterator_traits<InputIt>::value_type, value_type>
    &&  (!std::is_convertible_v<InputIt, std::string_view>)
    /// @brief Constructs a sequence from a pair of iterators.
    /// @tparam InputIt The type of the input iterators.
    /// @param first The beginning iterator of the sequence.
    /// @param last The ending iterator of the sequence.
    generic_sequence(InputIt first, InputIt last)
    :   _sq(first, last)
    ,   _ptr_td()
    {}
    ///
    /// Copy constructor.
    generic_sequence(const generic_sequence& other)
    :   _sq(other._sq)
    ,   _ptr_td(other._ptr_td ? std::make_unique<Map>(*other._ptr_td) : nullptr)
    {}
    ///
    /// Move constructor.
    generic_sequence(generic_sequence&& other) noexcept
    :   _sq(std::move(other._sq))
    ,   _ptr_td(std::move(other._ptr_td))
    {}
    ///
    /// @brief Constructs a sequence from an initializer list.
    /// @param init The initializer list containing the residues.
    generic_sequence(std::initializer_list<value_type> init)
    :   _sq(init)
    ,   _ptr_td()
    {}

// -- copy assignment operators ------------------------------------------------
    ///
    /// Copy assignment operator.
    generic_sequence& operator= (const generic_sequence& other)
    {   _sq = other._sq;
        _ptr_td = other._ptr_td ? std::make_unique<Map>(*other._ptr_td) : nullptr;
        return *this;
    }
    ///
    /// Move assignment operator.
    generic_sequence& operator= (generic_sequence&& other)
    {   _sq = std::move(other._sq);
        _ptr_td = std::move(other._ptr_td);
        return *this;
    }
    ///
    /// Assignment operator from an initializer list.
    generic_sequence& operator= (std::initializer_list<value_type> init)
    {   _sq = init;
        return *this;
    }

// -- iterators ----------------------------------------------------------------
    ///
    /// Returns a read/write iterator that points to the first residue in the
    /// @a sq. Iteration is done in ordinary residue order.
    iterator begin() noexcept
    {   return _sq.begin();   }
    ///
    /// Returns a read-only (constant) iterator that points to the first
    /// residue in the @a sq. Iteration is done in ordinary residue order.
    const_iterator begin() const noexcept
    {   return _sq.begin();   }
    ///
    /// Returns a read-only (constant) iterator that points to the first
    /// residue in the @a sq. Iteration is done in ordinary residue order.
    const_iterator cbegin() const noexcept
    {   return _sq.cbegin();   }
    ///
    /// Returns a read/write iterator that points one past the last residue in
    /// the @a sq. Iteration is done in ordinary residue order.
    iterator end() noexcept
    {   return _sq.end();   }
    ///
    /// Returns a read-only (constant) iterator that points one past the last
    /// residue in the @a sq. Iteration is done in ordinary residue order.
    const_iterator end() const noexcept
    {   return _sq.end();   }
    ///
    /// Returns a read-only (constant) iterator that points one past the last
    /// residue in the @a sq. Iteration is done in ordinary residue order.
    const_iterator cend() const noexcept
    {   return _sq.cend();   }
    ///
    /// Returns a read/write iterator that points to the last residue in the
    /// @a sq. Iteration is done in reverse residue order.
    reverse_iterator rbegin() noexcept
    {   return _sq.rbegin();   }
    ///
    /// Returns a read-only (constant) iterator that points to the last residue
    /// in the @a sq. Iteration is done in reverse residue order.
    const_reverse_iterator rbegin() const noexcept
    {   return _sq.rbegin();   }
    ///
    /// Returns a read-only (constant) iterator that points to the last residue
    /// in the @a sq. Iteration is done in reverse residue order.
    const_reverse_iterator crbegin() const noexcept
    {   return _sq.crbegin();   }
    ///
    /// Returns a read/write iterator that points to one before the first
    /// residue in the @a sq. Iteration is done in reverse residue order.
    reverse_iterator rend() noexcept
    {   return _sq.rend();   }
    ///
    /// Returns a read-only (constant) iterator that points to one before the
    /// first residue in the @a sq. Iteration is done in reverse residue order.
    const_reverse_iterator rend() const noexcept
    {   return _sq.rend();   }
    ///
    /// Returns a read-only (constant) iterator that points to one before the
    /// first residue in the @a sq. Iteration is done in reverse residue order.
    const_reverse_iterator crend() const noexcept
    {   return _sq.crend();   }

// -- capacity -----------------------------------------------------------------
    ///
    /// Returns true if the @a sq is empty. (Thus begin() would equal end().)
    bool empty() const noexcept
    {   return (_sq.empty() && (!_ptr_td || _ptr_td->empty()));   }
    ///
    /// Returns the number of residues in the @a sq.
    size_type size() const noexcept
    {   return _sq.size();   }
    ///
    /// Returns the size in memory (in bytes) used by the @a sq including its
    /// tagged data.
    size_type size_in_memory() const noexcept
    {   size_type mem = sizeof(Container) + (_sq.capacity() * sizeof(value_type));
        if (_ptr_td)
        {   mem += sizeof(Map);
            for (const auto& [tag, data] : *_ptr_td)
            {   mem += tag.capacity() * sizeof(char);
            // Note: estimating size of std::any content is not straightforward.
            // Here we just add sizeof of the contained type as a rough estimate.
                mem += data.has_value() ? sizeof(data.type()) : 0;
            }
        }
        return mem;
    }
    ///
    /// Reserves storage for at least @a new_cap residues. Does not change the
    /// size of the sequence.
    void reserve(size_type new_cap)
    {   _sq.reserve(new_cap);
    }

// -- modifiers ----------------------------------------------------------------
    ///
    /// Appends residues in the range [s, s + count) to the end of the @a sq.
    void append(const value_type* s, size_type count)
    {   _sq.insert(_sq.end(), s, s + count);
    }

// -- subscript operator -------------------------------------------------------
    ///
    /// Returns a reference to the residue at position @a pos in the @a sq.
    reference operator[] (size_type pos)
    {   return _sq[pos];   }
    ///
    /// Returns a const reference to the residue at position @a pos in the @a sq.
    const_reference operator[] (size_type pos) const
    {   return _sq[pos];   }

// -- view operator ------------------------------------------------------------
    ///
    /// Returns a non-owning view (subsequence) starting at position @a pos with
    /// length @a count.
    /// If @a count is gnx::sq::npos or exceeds the sequence length from
    /// @a pos, the subsequence extends to the end of the sequence.
#if defined(__CUDACC__) || defined(__HIPCC__)
    generic_sequence<Container> operator()
    (   size_type pos = 0
    ,   size_type count = npos
    )   const
    requires
    (   std::is_same_v<Container, thrust::device_vector<value_type>>
    ||  std::is_same_v<Container, thrust::universal_vector<value_type>>
    )
    {   if (pos > _sq.size())
            throw std::out_of_range("gnx::sq: pos > this->size()");
        return generic_sequence
        (   _sq.begin() + pos
        ,   (count > _sq.size() - pos) ? _sq.end() : _sq.begin() + pos + count
        );
    }
#endif
    generic_sequence_view<Container> operator()
    (   size_type pos = 0
    ,   size_type count = npos
    )   const
    {   generic_sequence_view<Container> sv(*this);
        return sv.subseq(pos, count);
    }

// -- managing tagged data -----------------------------------------------------
    ///
    /// Returns true if the tagged data with the specified @a tag exists.
    bool has(std::string_view tag) const
    {   return
        (   _ptr_td
        ?   _ptr_td->find(std::string(tag)) == _ptr_td->end()
        ?   false
        :   true
        :   false
        );
    }
    ///
    /// Returns a reference to the tagged data associated with the specified
    /// @a tag. If the tagged data does not exist, a new entry is created.
    std::any& operator[] (const std::string& tag)
    {   if (!_ptr_td) _ptr_td = std::make_unique<Map>();
        return (*_ptr_td)[tag];
    }
    ///
    /// Returns a reference to the tagged data associated with the specified
    /// @a tag. If the tagged data does not exist, a new entry is created.
    std::any& operator[] (std::string&& tag)
    {   if (!_ptr_td) _ptr_td = std::make_unique<Map>();
        return (*_ptr_td)[std::move(tag)];
    }
    ///
    /// Returns a const reference to the tagged data associated with the specified
    /// @a tag. Throws std::out_of_range if the tag does not exist.
    const std::any& operator[] (const std::string& tag) const
    {   if (!_ptr_td || _ptr_td->find(tag) == _ptr_td->end())
            throw std::out_of_range("gnx::sq: tag not found -> " + tag);
        return _ptr_td->at(tag);
    }
#if defined(__CUDACC__) || defined(__HIPCC__)
    value_type* data()
    {   return thrust::raw_pointer_cast(_sq.data());
    }
    const value_type* data() const
    {   return thrust::raw_pointer_cast(_sq.data());
    }
#else
    ///
    /// Returns a reference to the underlying container's data.
    value_type* data() noexcept
    {   return _sq.data();
    }
    ///
    /// Returns a const reference to the underlying container's data.
    const value_type* data() const noexcept
    {   return _sq.data();
    }
#endif

    // -- internal helpers (exposed for psq2 interop) --------------------------
    /// @cond INTERNAL
    bool       _has_td() const noexcept { return static_cast<bool>(_ptr_td); }
    const Map* _get_td() const noexcept { return _ptr_td.get(); }
    /// @endcond

    // -- comparison operators -----------------------------------------------------
    ///
    /// Equality operator with another generic_sequence of possibly different Container/Map.
    template<typename Container2, typename Map2>
    bool operator==(const generic_sequence<Container2, Map2>& rhs) const
    {   return _sq == rhs._sq;
    }
    ///
    /// Equality operator with std::string_view.
    bool operator==(std::string_view sv) const
    {   if (size() != sv.size())
            return false;
        return std::equal(_sq.begin(), _sq.end(), sv.begin());
    }
    ///
    /// Inequality operator with another generic_sequence of possibly different Container/Map.
    template<typename Container2, typename Map2>
    bool operator!=(const generic_sequence<Container2, Map2>& rhs) const
    {   return _sq != rhs._sq;
    }

// -- file i/o -----------------------------------------------------------------
    ///
    /// Loads a sequence from a file by its index using the provided
    /// @a read function object
    void load
    (   std::string_view filename
    ,   size_type ndx = 0
    ,   in::fast_aqz<generic_sequence> read = in::fast_aqz<generic_sequence>()
    )
    {   *this = read(filename, ndx);
    }
    ///
    /// Loads a sequence from a file by its identifier using the provided
    /// @a read function object
    void load
    (   std::string_view filename
    ,   std::string_view id
    ,   in::fast_aqz<generic_sequence> read = in::fast_aqz<generic_sequence>()
    )
    {   *this = read(filename, id);
    }
    ///
    /// Saves the sequence to a file using the provided write function object
    template<typename WriteFunc = out::fasta_gz>
    requires write_functor<WriteFunc, generic_sequence>
    void save
    (   std::string_view filename
    ,   WriteFunc write = WriteFunc()
    )
    {   write(filename, *this);
    }
    ///
    /// Returns the sequence and its tagged data as a formatted string.
    std::string print() const
    {   fmt::memory_buffer buf;
        fmt::format_to(std::back_inserter(buf), "{}", _sq.size());
        
        // Write raw sequence data
#if defined(__CUDACC__) || defined(__HIPCC__)
        if constexpr
        (   std::is_same_v<Container, thrust::device_vector<value_type>>
        )
        {   universal_host_pinned_vector<value_type> uhpv(_sq);
            buf.append(thrust::raw_pointer_cast(uhpv.data()), 
                      thrust::raw_pointer_cast(uhpv.data()) + uhpv.size());
        }
        else if constexpr
        (   std::is_same_v<Container, thrust::universal_vector<value_type>>
        )
        {   buf.append(thrust::raw_pointer_cast(_sq.data()), 
                      thrust::raw_pointer_cast(_sq.data()) + _sq.size());
        }
#if defined(__HIPCC__)
        else if constexpr
        (   std::is_same_v<Container, gnx::unified_vector<value_type>>
        )
        {   buf.append(thrust::raw_pointer_cast(_sq.data()), 
                      thrust::raw_pointer_cast(_sq.data()) + _sq.size());
        }
#endif //__HIPCC__
        else
#endif  //__CUDACC__
            buf.append(_sq.data(), _sq.data() + _sq.size());

        // Write tagged data
        if (_ptr_td)
            for (const auto& [tag, data] : *_ptr_td)
            {   // Print tag in quoted format: #tag#
                fmt::format_to(std::back_inserter(buf), "#{0}#", tag);
                if
                (   const auto it = td_print_visitor.find(std::type_index(data.type()))
                ;    it != td_print_visitor.cend()
                )
                    it->second(buf, data);
                else
                {   quote_with_delimiter(buf, "UNREGISTERED TYPE");
                    fmt::format_to(std::back_inserter(buf), "{{}}");
                }
            }
        return fmt::to_string(buf);
    }
    ///
    /// Prints the sequence to an output stream (for backward compatibility).
    void print(std::ostream& os) const
    {   os << print();
    }
    ///
    /// Scans the sequence and its tagged data from the input stream @a is.
    void scan(std::istream& is)
    {   size_type n;
        is >> std::boolalpha >> n;
        _sq.resize(n);
#if defined(__CUDACC__) || defined(__HIPCC__)
        if constexpr
        (   std::is_same_v<Container, thrust::device_vector<value_type>>
        )
        {   universal_host_pinned_vector<value_type> uhpv(n);
            is.read(thrust::raw_pointer_cast(uhpv.data()), n);
            thrust::copy(uhpv.begin(), uhpv.end(), _sq.begin());
        }
        else if constexpr
        (   std::is_same_v<Container, thrust::universal_vector<value_type>>
        )
        {   is.read(thrust::raw_pointer_cast(_sq.data()), n);
        }
#if defined(__HIPCC__)
        else if constexpr
        (   std::is_same_v<Container, gnx::unified_vector<value_type>>
        )
        {   is.read(thrust::raw_pointer_cast(_sq.data()), n);
        }
#endif //__HIPCC__
        else
#endif  //__CUDACC__
            is.read(_sq.data(), n);
        if (is.peek() == '#' && !_ptr_td)
            _ptr_td = std::make_unique<Map>();
        while (is.peek() == '#')
        {   std::string tag, type;
            std::any a;
            is >> std::quoted(tag, '#')
               >> std::quoted(type, '|');
            if
            (   const auto it = td_scan_visitor.find(type)
            ;    it != td_scan_visitor.cend()
            )
                it->second(is, a);
            else
                throw std::runtime_error
                (   fmt::format("gnx::sq: unregistered type -> {}", type)
                );
            (*_ptr_td)[tag] = a;
        }
    }
};

// -- comparison operators (external) -----------------------------------------
    ///
    /// Symmetric operator for "literal" == generic_sequence
    template<typename Container>
    bool operator==(std::string_view lhs, const generic_sequence<Container>& rhs)
    {   return rhs == lhs; 
    }
    ///
    /// Compare generic_sequence with generic_sequence_view
    template<typename Container1, typename Map1, typename Container2>
    bool operator==(const generic_sequence<Container1, Map1>& lhs, const generic_sequence_view<Container2>& rhs)
    {   if (lhs.size() != rhs.size()) return false;
        return std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }
    template<typename Container1, typename Map1, typename Container2>
    bool operator==(const generic_sequence_view<Container2>& lhs, const generic_sequence<Container1, Map1>& rhs)
    {   return rhs == lhs;
    }
    template<typename Container1, typename Map1, typename Container2>
    bool operator!=(const generic_sequence<Container1, Map1>& lhs, const generic_sequence_view<Container2>& rhs)
    {   return ! (lhs == rhs);
    }
    template<typename Container1, typename Map1, typename Container2>
    bool operator!=(const generic_sequence_view<Container2>& lhs, const generic_sequence<Container1, Map1>& rhs)
    {   return ! (lhs == rhs);
    }
    ///
    /// Specialization for C-string literal comparisons (const char*)
    template<typename Container>
    bool operator==(const char* lhs, const generic_sequence<Container>& rhs)
    {   return std::string_view(lhs) == rhs;
    }
    template<typename Container>
    bool operator==(const generic_sequence<Container>& lhs, const char* rhs)
    {   return lhs == std::string_view(rhs);
    }

// -- i/o stream operators -----------------------------------------------------
    ///
    /// Output stream operator for generic_sequence.
    template<typename T>
    std::ostream& operator<< (std::ostream& os, const generic_sequence<T>& s)
    {   s.print(os);
        return os;
    }
    ///
    /// Input stream operator for generic_sequence.
    template<typename T>
    std::istream& operator>> (std::istream& is, generic_sequence<T>& s)
    {   s.scan(is);
        return is;
    }
    ///
    /// A sequence of @a char
    using sq = generic_sequence<std::vector<char>>;
    ///
    /// A sequence of @a char stored in a GPU device vector.
#if defined(__CUDACC__) || defined(__HIPCC__)
    using dsq = generic_sequence<thrust::device_vector<char>>;
#endif // __CUDACC__ || __HIPCC__

// -- Jupyter integration ------------------------------------------------------
#ifdef __CLING__
    template<typename Container, typename Map>
    nlohmann::json mime_bundle_repr(generic_sequence<Container, Map> const& s)
    {   return detail::print_to_bundle(s);
    }
#endif //__CLING__

}   // end gnx namespace

// -- string literal operator --------------------------------------------------

    inline gnx::sq operator""_sq (const char* str, std::size_t len)
    {   return gnx::sq(str);   }

// -- fmt formatter for gnx::sq ------------------------------------------------

template <>
struct fmt::formatter<gnx::sq> : fmt::formatter<std::string>
{   auto format(const gnx::sq& s, format_context& ctx) const
    {   return fmt::formatter<std::string>::format(s.print(), ctx);
    }
};

template <typename Container, typename Map>
struct fmt::formatter<gnx::generic_sequence<Container, Map>> : fmt::formatter<std::string>
{   auto format(const gnx::generic_sequence<Container, Map>& s, format_context& ctx) const
    {   return fmt::formatter<std::string>::format(s.print(), ctx);
    }
};
