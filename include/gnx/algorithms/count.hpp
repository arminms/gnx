// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <ranges>
#include <utility>
#include <cstddef>

#if defined(__CUDACC__)
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#elif defined(__HIPCC__)
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#endif // __CUDACC__

#include <gnx/concepts.hpp>
#include <gnx/execution.hpp>
#include <gnx/lut/case_fold.hpp>

namespace gnx {

namespace detail {

#if !defined(_WIN32)
#pragma omp declare simd uniform(v, table) linear(i:1)
#endif
template<typename T, typename SizeT, typename LutT>
inline void count_func(const T* v, SizeT i, LutT* table)
{   ++table[static_cast<uint8_t>(v[i])];
}

#if defined(__CUDACC__) || defined(__HIPCC__)

template<typename ExecPolicy, device_resident_iterator Iterator>
inline std::map<char, std::size_t> count_device
(   const ExecPolicy& policy
,   Iterator first
,   Iterator last
)
{   typedef typename std::iterator_traits<Iterator>::value_type value_type;
    typedef typename std::iterator_traits<Iterator>::difference_type difference_type;

    difference_type n = last - first;
    if (n <= 0)
        return {};

    // Create device vectors for normalized characters
    thrust::device_vector<char> d_normalized(n);
    thrust::device_vector<uint8_t> d_lut(lut::case_fold.begin(), lut::case_fold.end());

    // Normalize all characters using the lookup table
    auto lut_ptr = thrust::raw_pointer_cast(d_lut.data());
    thrust::transform
    (   first
    ,   last
    ,   d_normalized.begin()
    ,   [lut_ptr] __host__ __device__ (value_type c)
        {   return static_cast<char>(lut_ptr[static_cast<uint8_t>(c)]);
        }
    );

    // Sort normalized characters
    thrust::sort(d_normalized.begin(), d_normalized.end());

    // Count occurrences using reduce_by_key
    thrust::device_vector<char> d_unique_chars(n);
    thrust::device_vector<std::size_t> d_counts(n);

    auto end_pair = thrust::reduce_by_key
    (   d_normalized.begin()
    ,   d_normalized.end()
    ,   thrust::constant_iterator<std::size_t>(1)
    ,   d_unique_chars.begin()
    ,   d_counts.begin()
    );

    // Copy results back to host and build the map
    std::size_t num_unique = end_pair.first - d_unique_chars.begin();
    std::vector<char> h_chars(num_unique);
    std::vector<std::size_t> h_counts(num_unique);

    thrust::copy(d_unique_chars.begin(), end_pair.first, h_chars.begin());
    thrust::copy(d_counts.begin(), end_pair.second, h_counts.begin());

    std::map<char, std::size_t> result;
    for (std::size_t i = 0; i < num_unique; ++i)
        result[h_chars[i]] = h_counts[i];

    return result;
}
#endif // __CUDACC__

} // end detail namespace

/// @brief Count all bases/amino acids in a sequence (case-insensitive).
/// @tparam Iterator Forward iterator type
/// @param first Iterator to the beginning of the sequence
/// @param last Iterator to the end of the sequence
/// @return std::map<char, std::size_t> with uppercase characters as keys and counts as values
#if defined(__CUDACC__)
template<device_resident_iterator Iterator>
inline std::map<char, std::size_t> count
(   Iterator first
,   Iterator last
)
{   return detail::count_device(thrust::cuda::par, first, last);
}
template<host_resident_iterator Iterator>
#elif defined(__HIPCC__)
template<device_resident_iterator Iterator>
inline std::map<char, std::size_t> count
(   Iterator first
,   Iterator last
)
{   return detail::count_device(thrust::hip::par, first, last);
}
template<host_resident_iterator Iterator>
#else
template<typename Iterator>
#endif
inline std::map<char, std::size_t> count
(   Iterator first
,   Iterator last
)
{   std::map<char, std::size_t> result;
    std::array<std::size_t, 256> counts{};
    std::fill(counts.begin(), counts.end(), 0);

    while (first != last)
        ++counts[static_cast<uint8_t>(*first++)];

    // add lowercase ASCII letters's counts to uppercase
    for (char c = 'a'; c <= 'z'; ++c)
    {   counts[static_cast<uint8_t>(c - 32)] += counts[static_cast<uint8_t>(c)];
        counts[static_cast<uint8_t>(c)] = 0;
    }

    // fill the result map with non-zero counts
    for (char c = ' '; c <= '~'; ++c)
        if (counts[static_cast<uint8_t>(c)] > 0)
            result[c] = counts[static_cast<uint8_t>(c)];

    return result;
}

/// @brief Parallel-enabled count using an execution policy.
/// @tparam ExecPolicy Execution policy type (e.g., gnx::execution::seq)
/// @tparam Iterator Forward iterator type
/// @param policy Execution policy controlling algorithm execution
/// @param first Iterator to the beginning of the sequence
/// @param last Iterator to the end of the sequence
/// @return std::map<char, std::size_t> with uppercase characters as keys and counts as values
#if defined(__CUDACC__) || defined(__HIPCC__)
template<typename ExecPolicy, device_resident_iterator Iterator>
inline std::map<char, std::size_t> count
(   ExecPolicy&& policy
,   Iterator first
,   Iterator last
)
{   return detail::count_device
    (   std::forward<ExecPolicy>(policy)
    ,   first
    ,   last
    );
}
template<typename ExecPolicy, host_resident_iterator Iterator>
#else
template<typename ExecPolicy, std::random_access_iterator Iterator>
#endif
requires gnx::is_execution_policy_v<std::decay_t<ExecPolicy>>
inline std::map<char, std::size_t> count
(   ExecPolicy&& policy
,   Iterator first
,   Iterator last
)
{   typedef typename std::iterator_traits<Iterator>::difference_type difference_type;

    difference_type n = last - first;
    if (n <= 0)
        return {};

    const auto vptr = &first[0];

    // compile-time dispatch based on execution policy
    if constexpr
    (   std::is_same_v<std::decay_t<ExecPolicy>
    ,   gnx::execution::sequenced_policy>
    )
    {   return count(first, last);
    }
    else if constexpr
    (   std::is_same_v<std::decay_t<ExecPolicy>
    ,   gnx::execution::unsequenced_policy>
    )
    {   std::array<std::size_t, 256> counts{};
        std::fill(counts.begin(), counts.end(), 0);

        #pragma omp simd
        for (difference_type i = 0; i < n; ++i)
            detail::count_func(vptr, i, counts.data());

        for (char c = 'a'; c <= 'z'; ++c)
        {   counts[static_cast<uint8_t>(c - 32)] += counts[static_cast<uint8_t>(c)];
            counts[static_cast<uint8_t>(c)] = 0;
        }
        std::map<char, std::size_t> result;
        for (char c = ' '; c <= '~'; ++c)
            if (counts[static_cast<uint8_t>(c)] > 0)
                result[c] = counts[static_cast<uint8_t>(c)];
        return result;
    }
    else if constexpr
    (   std::is_same_v<std::decay_t<ExecPolicy>
    ,   gnx::execution::parallel_policy>
    )
    {   // parallel approach: each thread builds local count, then merge
        std::array<std::size_t, 256> counts{};
        std::fill(counts.begin(), counts.end(), 0);
        #pragma omp parallel
        {   std::array<std::size_t, 256> local_counts{};
            std::fill(local_counts.begin(), local_counts.end(), 0);
            #pragma omp for
            for (difference_type i = 0; i < n; ++i)
            {   detail::count_func(vptr, i, local_counts.data());
            }
            for (char c = 'a'; c <= 'z'; ++c)
            {   local_counts[static_cast<uint8_t>(c - 32)]
            +=  local_counts[static_cast<uint8_t>(c)];
                local_counts[static_cast<uint8_t>(c)] = 0;
            }
            #pragma omp critical
            {   for (char c = ' '; c <= '~'; ++c)
                {   counts[static_cast<uint8_t>(c)]
                +=  local_counts[static_cast<uint8_t>(c)];
                }
            }
        }
        std::map<char, std::size_t> result;
        for (char c = ' '; c <= '~'; ++c)
            if (counts[static_cast<uint8_t>(c)] > 0)
                result[c] = counts[static_cast<uint8_t>(c)];
        return result;
    }
    else
        return count(first, last);
}

/// @brief Count all bases/amino acids in a sequence range (case-insensitive).
/// @tparam Range Range type
/// @param seq The sequence range
/// @return std::map<char, std::size_t> with uppercase characters as keys and counts as values
template<std::ranges::input_range Range>
inline std::map<char, std::size_t> count(const Range& seq)
{   return count
    (   std::begin(seq)
    ,   std::end(seq)
    );
}

/// @brief Parallel-enabled count of bases/amino acids in sequence range.
/// @tparam ExecPolicy Execution policy type (e.g., gnx::execution::par)
/// @tparam Range Range type
/// @param policy Execution policy controlling algorithm execution
/// @param seq The sequence range
/// @return std::map<char, std::size_t> with uppercase characters as keys and counts as values
template<typename ExecPolicy, std::ranges::input_range Range>
inline std::map<char, std::size_t> count
(   ExecPolicy&& policy
,   const Range& seq
)
{   return count
    (   std::forward<ExecPolicy>(policy)
    ,   std::begin(seq)
    ,   std::end(seq)
    );
}

} // namespace gnx
