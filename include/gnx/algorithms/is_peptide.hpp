// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
#pragma once

#include <algorithm>
#include <iterator>
#include <ranges>

#if defined(__CUDACC__) || defined(__HIPCC__)
    #include <thrust/host_vector.h>
    #include <thrust/device_vector.h>
    #include <thrust/universal_vector.h>
    #include <thrust/memory.h>
#endif // __CUDACC__

#include <gnx/concepts.hpp>
#include <gnx/lut/peptide.hpp>

namespace gnx {

#if defined(__CUDACC__) || defined(__HIPCC__)
template<device_resident_iterator Iterator>
constexpr bool is_peptide
(   Iterator first
,   Iterator last
)
{   using value_type = typename std::iterator_traits<Iterator>::value_type;
    auto& is_peptide_device = lut::get_static_device<bool, 0>(gnx::lut::is_peptide);
    return thrust::any_of
    (   first
    ,   std::distance(first, last) < 100'000 ? last : first + 100'000
    ,   [] __device__ (value_type c)
        {   return is_peptide_device[static_cast<uint8_t>(c)];
        }
    );
}
template<host_resident_iterator Iterator>
#else
template<typename Iterator>
#endif
constexpr bool is_peptide
(   Iterator first
,   Iterator last
)
{   using value_type = typename std::iterator_traits<Iterator>::value_type;
    return std::any_of
    (   first
    ,   std::distance(first, last) < 10'000 ? last : first + 10'000
    ,   [](value_type c)
        {   return gnx::lut::is_peptide[static_cast<uint8_t>(c)];
        }
    );
}

template<std::ranges::input_range Range>
constexpr bool is_peptide(const Range& seq)
{   return is_peptide(std::begin(seq), std::end(seq));
}

} // namespace gnx