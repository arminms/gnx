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
#include <gnx/lut/codon.hpp>

namespace gnx {

namespace detail {

/// @brief Translate a single codon (three bases) to an amino acid character.
/// @param b1 First base character
/// @param b2 Second base character
/// @param b3 Third base character
/// @return Amino acid single-letter code, or 'X' for invalid/unknown codons
inline char translate_codon(char b1, char b2, char b3) noexcept
{   uint8_t i1 = lut::base_enc[static_cast<uint8_t>(b1)];
    uint8_t i2 = lut::base_enc[static_cast<uint8_t>(b2)];
    uint8_t i3 = lut::base_enc[static_cast<uint8_t>(b3)];
    if (i1 == 0xFF || i2 == 0xFF || i3 == 0xFF)
        return 'X'; // unknown / invalid codon
    return lut::codon_table[static_cast<uint8_t>(i1 * 16 + i2 * 4 + i3)];
}

#if defined(__CUDACC__) || defined(__HIPCC__)

namespace kernel {

template<typename InT, typename OutT, typename SizeT>
__global__ void translate_kernel
(   const InT* d_in
,   OutT* d_out
,   SizeT in_n          // number of input characters (must be divisible by 3)
,   const uint8_t* d_base_enc
,   const char*    d_codon_table
)
{   // Load both LUTs into shared memory
    __shared__ uint8_t s_base_enc[256];
    __shared__ char    s_codon[64];

    int tid = threadIdx.x;
    s_base_enc[tid] = d_base_enc[tid];
    if (tid < 64)
        s_codon[tid] = d_codon_table[tid];
    __syncthreads();

    SizeT codon_idx = static_cast<SizeT>(blockDim.x) * blockIdx.x + tid;
    SizeT num_codons = in_n / 3;
    if (codon_idx < num_codons)
    {   SizeT base_idx = codon_idx * 3;
        uint8_t i1 = s_base_enc[static_cast<uint8_t>(d_in[base_idx    ])];
        uint8_t i2 = s_base_enc[static_cast<uint8_t>(d_in[base_idx + 1])];
        uint8_t i3 = s_base_enc[static_cast<uint8_t>(d_in[base_idx + 2])];
        if (i1 == 0xFF || i2 == 0xFF || i3 == 0xFF)
            d_out[codon_idx] = static_cast<OutT>('X');
        else
            d_out[codon_idx] = static_cast<OutT>(s_codon[i1 * 16 + i2 * 4 + i3]);
    }
}

} // end kernel namespace

/// @brief GPU-resident translate: device input → device output.
/// @tparam ExecPolicy Execution policy type with optional stream member
/// @tparam InIterator Device-resident random access iterator over input chars
/// @tparam OutIterator Device-resident random access iterator over output chars
/// @param policy  Execution policy (may carry a CUDA/HIP stream)
/// @param first   Iterator to beginning of nucleotide sequence
/// @param last    Iterator past the end of the nucleotide sequence
/// @param d_first Output iterator for the protein sequence
/// @return Iterator past the last written amino acid
template
<   typename ExecPolicy
,   device_resident_iterator InIterator
,   device_resident_iterator OutIterator
>
inline OutIterator translate_device
(   const ExecPolicy& policy
,   InIterator first
,   InIterator last
,   OutIterator d_first
)
{   typedef typename std::iterator_traits<InIterator>::difference_type difference_type;

    difference_type in_n = last - first;
    difference_type num_codons = in_n / 3;
    if (num_codons <= 0)
        return d_first;

    auto d_base_enc    = lut::get_static_device(lut::base_enc);
    // codon_table has 64 entries; wrap in a 256-element helper array for the
    // generic get_static_device template. We use a fixed-size copy instead.
    static std::unique_ptr
    <   thrust::device_vector<char>
    ,   lut::device_safe_deleter<char>
    >
    d_codon_vec(new thrust::device_vector<char>
    (   lut::codon_table.begin()
    ,   lut::codon_table.end()
    ));

    difference_type threads_per_block{256};
    difference_type grid_size{(num_codons + threads_per_block - 1) / threads_per_block};

#if defined(__HIPCC__)
    hipStream_t stream = 0;
#else
    cudaStream_t stream = 0;
#endif
    if constexpr (has_stream_member<ExecPolicy>)
        stream = policy.stream();

    kernel::translate_kernel<<<grid_size, threads_per_block, 0, stream>>>
    (   thrust::raw_pointer_cast(&first[0])
    ,   thrust::raw_pointer_cast(&d_first[0])
    ,   in_n
    ,   thrust::raw_pointer_cast(d_base_enc.data())
    ,   thrust::raw_pointer_cast(d_codon_vec->data())
    );

    return d_first + num_codons;
}

#endif // __CUDACC__ || __HIPCC__

} // end detail namespace

// ---------------------------------------------------------------------------
// Policy-free overloads (host)
// ---------------------------------------------------------------------------

/// @brief Translate a nucleotide sequence to a protein sequence.
/// @tparam InputIterator  Random access iterator over the nucleotide input
/// @tparam OutputIterator Output iterator for amino acid characters
/// @param first   Iterator to the beginning of the nucleotide sequence
/// @param last    Iterator past the end of the nucleotide sequence
/// @param d_first Output iterator for the resulting protein sequence
/// @return Iterator past the last written amino acid
///
/// Reads the input three characters at a time (codons) starting at @p first,
/// translates each codon using the NCBI standard genetic code (table 1), and
/// writes the corresponding single-letter amino acid code to @p d_first.
/// Incomplete trailing codons (< 3 bases) are silently ignored.
/// Invalid or ambiguous bases within a codon produce 'X'.
/// Stop codons are written as '*'.
///
/// @note Both DNA (T) and RNA (U) are supported.
/// @note The output range must accommodate at least `(last - first) / 3` elements.
#if defined(__CUDACC__)
template<device_resident_iterator InIterator, device_resident_iterator OutIterator>
OutIterator translate
(   InIterator first
,   InIterator last
,   OutIterator d_first
)
{   return detail::translate_device(thrust::cuda::par, first, last, d_first);
}
template<host_resident_iterator InIterator, std::output_iterator<char> OutIterator>
#elif defined(__HIPCC__)
template<device_resident_iterator InIterator, device_resident_iterator OutIterator>
OutIterator translate
(   InIterator first
,   InIterator last
,   OutIterator d_first
)
{   return detail::translate_device(thrust::hip::par, first, last, d_first);
}
template<host_resident_iterator InIterator, std::output_iterator<char> OutIterator>
#else
template<std::random_access_iterator InIterator, std::output_iterator<char> OutIterator>
#endif
[[nodiscard]] OutIterator translate
(   InIterator first
,   InIterator last
,   OutIterator d_first
)
{   typedef typename std::iterator_traits<InIterator>::difference_type difference_type;
    difference_type n = last - first;
    difference_type num_codons = n / 3;

    const auto vptr = &first[0];
    for (difference_type i = 0; i < num_codons; ++i, ++d_first)
        *d_first = detail::translate_codon
        (   static_cast<char>(vptr[i * 3    ])
        ,   static_cast<char>(vptr[i * 3 + 1])
        ,   static_cast<char>(vptr[i * 3 + 2])
        );
    return d_first;
}

// ---------------------------------------------------------------------------
// Policy-accepting overloads
// ---------------------------------------------------------------------------

/// @brief Parallel-enabled translate using an execution policy.
/// @tparam ExecPolicy  Execution policy type (e.g., gnx::execution::par)
/// @tparam InIterator  Random access iterator over the nucleotide input
/// @tparam OutIterator Output iterator for amino acid characters
/// @param policy  Execution policy controlling algorithm execution
/// @param first   Iterator to the beginning of the nucleotide sequence
/// @param last    Iterator past the end of the nucleotide sequence
/// @param d_first Output iterator for the resulting protein sequence
/// @return Iterator past the last written amino acid
///
/// Supports sequential, parallel, vectorized, and GPU execution.
/// The output range must accommodate at least `(last - first) / 3` elements.
#if defined(__CUDACC__) || defined(__HIPCC__)
template
<   typename ExecPolicy
,   device_resident_iterator InIterator
,   device_resident_iterator OutIterator
>
inline OutIterator translate
(   ExecPolicy&& policy
,   InIterator first
,   InIterator last
,   OutIterator d_first
)
{   return detail::translate_device
    (   std::forward<ExecPolicy>(policy)
    ,   first, last, d_first
    );
}
template
<   typename ExecPolicy
,   host_resident_iterator InIterator
,   std::output_iterator<char> OutIterator
>
#else
template
<   typename ExecPolicy
,   std::random_access_iterator InIterator
,   std::output_iterator<char> OutIterator
>
#endif
requires gnx::is_execution_policy_v<std::decay_t<ExecPolicy>>
[[nodiscard]] inline OutIterator translate
(   ExecPolicy&& policy
,   InIterator first
,   InIterator last
,   OutIterator d_first
)
{   typedef typename std::iterator_traits<InIterator>::difference_type difference_type;
    difference_type n = last - first;
    difference_type num_codons = n / 3;
    if (num_codons <= 0)
        return d_first;

    const auto vptr = &first[0];
    auto optr = &(*d_first);

    if constexpr
    (   std::is_same_v<std::decay_t<ExecPolicy>
    ,   gnx::execution::sequenced_policy>
    )
    {   return translate(first, last, d_first);
    }
    else if constexpr
    (   std::is_same_v<std::decay_t<ExecPolicy>
    ,   gnx::execution::unsequenced_policy>
    )
    {   // Vectorized: each codon is independent, so SIMD over codon indices is safe
#pragma omp simd
        for (difference_type i = 0; i < num_codons; ++i)
            optr[i] = detail::translate_codon
            (   static_cast<char>(vptr[i * 3    ])
            ,   static_cast<char>(vptr[i * 3 + 1])
            ,   static_cast<char>(vptr[i * 3 + 2])
            );
        return d_first + num_codons;
    }
    else if constexpr
    (   std::is_same_v<std::decay_t<ExecPolicy>
    ,   gnx::execution::parallel_policy>
    )
    {   // parallel execution with OpenMP
#pragma omp parallel for
        for (difference_type i = 0; i < num_codons; ++i)
            optr[i] = detail::translate_codon
            (   static_cast<char>(vptr[i * 3    ])
            ,   static_cast<char>(vptr[i * 3 + 1])
            ,   static_cast<char>(vptr[i * 3 + 2])
            );
        return d_first + num_codons;
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
        for (difference_type i = 0; i < num_codons; ++i)
            optr[i] = detail::translate_codon
            (   static_cast<char>(vptr[i * 3    ])
            ,   static_cast<char>(vptr[i * 3 + 1])
            ,   static_cast<char>(vptr[i * 3 + 2])
            );
        return d_first + num_codons;
    }
    else
    {   return translate(first, last, d_first);
    }
}

// ---------------------------------------------------------------------------
// Range overloads
// ---------------------------------------------------------------------------

/// @brief Translate a nucleotide range to a protein sequence.
/// @tparam InputRange  Input range of nucleotides
/// @tparam OutputRange Output range for amino acids (resized to fit if possible)
/// @param seq     The nucleotide sequence to translate
/// @param out     The output protein sequence
/// @return Iterator past the last written amino acid
///
/// Convenience overload accepting any range types. The caller is responsible
/// for ensuring @p out has sufficient capacity.
template<std::ranges::random_access_range InputRange, std::ranges::range OutputRange>
[[nodiscard]] auto translate(const InputRange& seq, OutputRange& out)
{   return translate
    (   std::begin(seq)
    ,   std::end(seq)
    ,   std::begin(out)
    );
}

/// @brief Parallel-enabled translate of a sequence range.
/// @tparam ExecPolicy Execution policy type (e.g., gnx::execution::par)
/// @tparam InputRange Input range of nucleotides
/// @tparam OutputRange Output range for amino acids
/// @param policy Execution policy controlling algorithm execution
/// @param seq    The nucleotide sequence to translate
/// @param out    The output protein sequence
/// @return Iterator past the last written amino acid
template
<   typename ExecPolicy
,   std::ranges::random_access_range InputRange
,   std::ranges::range OutputRange
>
[[nodiscard]] inline auto translate
(   ExecPolicy&& policy
,   const InputRange& seq
,   OutputRange& out
)
{   return translate
    (   std::forward<ExecPolicy>(policy)
    ,   std::begin(seq)
    ,   std::end(seq)
    ,   std::begin(out)
    );
}

} // namespace gnx
