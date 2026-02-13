// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <memory>

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/device_vector.h>

namespace gnx::lut {

template <typename T>
struct device_safe_deleter
{   void operator()(thrust::device_vector<T>* ptr) const
    {   // only attempt cleanup if we can still communicate with the GPU
#if defined(__CUDACC__)
        cudaError_t err = cudaFree(0);
        if (err == cudaSuccess)
#elif defined(__HIPCC__)
        hipError_t err = hipFree(0);
        if (err == hipSuccess)
#endif
            delete ptr;

        // If err != cudaSuccess, the driver is likely already down.
        // We let the OS reclaim the memory to avoid a crash.
    }
};

template <typename T>
thrust::device_vector<T>& get_static_device(std::array<T, 256> const& table)
{   static std::unique_ptr<thrust::device_vector<T>, device_safe_deleter<T>>
        vec(new thrust::device_vector<T>(table.begin(), table.end()));
    return *vec;
}

} // namespace gnx::lut

#endif // __CUDACC__ || __HIPCC__