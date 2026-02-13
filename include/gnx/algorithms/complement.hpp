// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <utility>
#include <cstddef>

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#endif // __CUDACC__ || __HIPCC__

#include <gnx/concepts.hpp>
#include <gnx/execution.hpp>
#include <gnx/lut/complement.hpp>

namespace gnx {

namespace detail {

#if !defined(_WIN32)
#pragma omp declare simd uniform(v, table) linear(i:1)
#endif
template<typename T, typename SizeT, typename LutT>
inline void complement_func(T* v, SizeT i, const LutT* table)
{   v[i] = table[static_cast<uint8_t>(v[i])];
}

#if defined(__CUDACC__) || defined(__HIPCC__)

namespace kernel {

template<typename T, typename SizeT, typename TableT>
__global__ void complement_kernel
(   T* d_data
,   SizeT n
,   const TableT* lut
)
{   // Allocate shared memory for the lookup table
    __shared__ TableT shared_lut[256];
    int tid = threadIdx.x;
    shared_lut[tid] = lut[tid];
    __syncthreads();

    auto idx = blockDim.x * blockIdx.x + tid;
    if (idx < n)
    {   uint8_t c = static_cast<uint8_t>(d_data[idx]);
        d_data[idx] = shared_lut[c];
    }
}

} // end kernel namespace

template<typename ExecPolicy, device_resident_iterator Iterator>
inline void complement_device
(   const ExecPolicy& policy
,   Iterator first
,   Iterator last
)
{   typedef typename std::iterator_traits<Iterator>::value_type value_type;
    typedef typename std::iterator_traits<Iterator>::difference_type difference_type;

    difference_type n = last - first;
    if (n <= 0)
        return;

    // create static device-resident copy of the complement lookup table
    auto d_lut = lut::get_static_device(lut::complement);

    difference_type threads_per_block{256};
    difference_type grid_size{(n + threads_per_block - 1) / threads_per_block};

#if defined(__HIPCC__)
    hipStream_t stream = 0;
#else
    cudaStream_t stream = 0;
#endif
    if constexpr (has_stream_member<ExecPolicy>)
       stream = policy.stream();

    kernel::complement_kernel<<<grid_size, threads_per_block, 0, stream>>>
    (   thrust::raw_pointer_cast(&first[0])
    ,   n
    ,   thrust::raw_pointer_cast(d_lut.data())
    );
}

#endif // __CUDACC__ || __HIPCC__

} // end detail namespace

/// @brief Convert a nucleotide sequence to its complement (in-place).
/// @tparam Iterator Random access iterator type
/// @param first Iterator to the beginning of the sequence
/// @param last Iterator to the end of the sequence
///
/// This function modifies the sequence in-place, converting each nucleotide to its
/// Watson-Crick complement using a lookup table. Supports both DNA (A, T, G, C) and
/// RNA (A, U, G, C) nucleotides, as well as IUPAC ambiguity codes.
///
/// @note The operation is case-preserving (uppercase remains uppercase, lowercase remains lowercase)
#if defined(__CUDACC__)
template<device_resident_iterator Iterator>
constexpr void complement
(   Iterator first
,   Iterator last
)
{   detail::complement_device(thrust::cuda::par, first, last);
}
template<host_resident_iterator Iterator>
#elif defined(__HIPCC__)
template<device_resident_iterator Iterator>
constexpr void complement
(   Iterator first
,   Iterator last
)
{   detail::complement_device(thrust::hip::par, first, last);
}
template<host_resident_iterator Iterator>
#else
template<std::random_access_iterator Iterator>
#endif
constexpr void complement
(   Iterator first
,   Iterator last
)
{   typedef typename std::iterator_traits<Iterator>::difference_type difference_type;
    
    difference_type n = last - first;
    if (n <= 0)
        return;

    const auto vptr = &first[0];
    for (difference_type i = 0; i < n; ++i)
        vptr[i] = lut::complement[static_cast<uint8_t>(vptr[i])];
}

/// @brief Parallel-enabled complement using an execution policy (in-place).
/// @tparam ExecPolicy Execution policy type (e.g., gnx::execution::seq)
/// @tparam Iterator Random access iterator type
/// @param policy Execution policy controlling algorithm execution
/// @param first Iterator to the beginning of the sequence
/// @param last Iterator to the end of the sequence
///
/// This overload allows explicit control over execution strategy using execution policies.
/// Supports sequential, parallel, vectorized, and GPU execution.
#if defined(__CUDACC__) || defined(__HIPCC__)
template<typename ExecPolicy, device_resident_iterator Iterator>
inline void complement
(   ExecPolicy&& policy
,   Iterator first
,   Iterator last
)
{   detail::complement_device
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
inline void complement
(   ExecPolicy&& policy
,   Iterator first
,   Iterator last
)
{   typedef typename std::iterator_traits<Iterator>::difference_type difference_type;

    difference_type n = last - first;
    if (n <= 0)
        return;

    const auto vptr = &first[0];

    // compile-time dispatch based on execution policy
    if constexpr
    (   std::is_same_v<std::decay_t<ExecPolicy>
    ,   gnx::execution::sequenced_policy>
    )   // sequential execution
    {   complement(first, last);
    }
    else if constexpr
    (   std::is_same_v<std::decay_t<ExecPolicy>
    ,   gnx::execution::unsequenced_policy>
    )
    {   // unsequenced (vectorized) execution
        #pragma omp simd
        for (difference_type i = 0; i < n; ++i)
            detail::complement_func(vptr, i, lut::complement.data());
    }
    else if constexpr
    (   std::is_same_v<std::decay_t<ExecPolicy>
    ,   gnx::execution::parallel_policy>
    )
    {   // parallel execution with OpenMP
        #pragma omp parallel for
        for (difference_type i = 0; i < n; ++i)
            detail::complement_func(vptr, i, lut::complement.data());
    }
    else if constexpr
    (   std::is_same_v<std::decay_t<ExecPolicy>
    ,   gnx::execution::parallel_unsequenced_policy>
    )
    {   // parallel unsequenced execution (vectorized + parallel)
#if defined(_WIN32)
        #pragma omp parallel for
#else
        #pragma omp parallel for simd
#endif // _WIN32
        for (difference_type i = 0; i < n; ++i)
            detail::complement_func(vptr, i, lut::complement.data());
    }
    else
    {   // default to sequential execution
        complement(first, last);
    }
}

/// @brief Convert a nucleotide sequence to its complement (in-place, range-based).
/// @tparam Range Range type
/// @param seq The sequence range to complement
///
/// This is a convenience overload that accepts any range type supporting
/// std::begin() and std::end().
template<std::ranges::input_range Range>
inline void complement(Range& seq)
{   complement
    (   std::begin(seq)
    ,   std::end(seq)
    );
}

/// @brief Parallel-enabled complement of a sequence range (in-place).
/// @tparam ExecPolicy Execution policy type (e.g., gnx::execution::par)
/// @tparam Range Range type
/// @param policy Execution policy controlling algorithm execution
/// @param seq The sequence range to complement
///
/// This overload combines execution policy control with range-based syntax
/// for maximum convenience and flexibility.
template<typename ExecPolicy, std::ranges::input_range Range>
inline void complement
(   ExecPolicy&& policy
,   Range& seq
)
{   complement
    (   std::forward<ExecPolicy>(policy)
    ,   std::begin(seq)
    ,   std::end(seq)
    );
}

} // namespace gnx

