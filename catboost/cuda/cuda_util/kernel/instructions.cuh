#pragma once
#if defined(__HIP_PLATFORM_AMD__)
#include <hipcub/thread/thread_load.hpp>
namespace cub = hipcub;
#else
#include <cub/thread/thread_load.cuh>
#endif


__forceinline__ __device__ ui32 bfe(ui32 a, ui32 start, ui32 length)
{
#if defined(__HIP_PLATFORM_AMD__)
    // HIP fallback: extract bits manually
    return (a >> start) & ((1u << length) - 1);
#else
    ui32 res;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(start), "r"(length));
    return res;
#endif
}

__forceinline__ __device__ ui32 load_noncached(const ui32 * ptr)
{
#if defined(__HIP_PLATFORM_AMD__)
    // HIP: use volatile load as fallback
    return __builtin_nontemporal_load(ptr);
#else
    ui32 res;
    asm("ld.global.cg.u32 %0, [%1];" : "=r"(res) : "l"(ptr));
    return res;
#endif
}

__forceinline__ __device__ uint2 load_noncached(const uint2* ptr)
{
#if defined(__HIP_PLATFORM_AMD__)
    // HIP: regular load (no direct non-temporal uint2 intrinsic)
    return *ptr;
#else
    uint2 res;
    asm("ld.global.cg.v2.u32 {%0, %1}, [%2];" : "=r"(res.x), "=r"(res.y) : "l"(ptr));
    return res;
#endif
}

__forceinline__ __device__ float load_noncached(const float * ptr)
{
#if defined(__HIP_PLATFORM_AMD__)
    // HIP: use volatile load as fallback
    return __builtin_nontemporal_load(ptr);
#else
    float res;
    asm("ld.global.cg.f32 %0, [%1];" : "=f"(res) : "l"(ptr));
    return res;
#endif
}

__device__ __forceinline__ void PrefetchL1(void *ptr) {
#if defined(__HIP_PLATFORM_AMD__)
    // HIP: prefetch using AMD intrinsic
    __builtin_prefetch(ptr, 0, 3);
#else
    asm("prefetch.global.L1 [%0];"::"l"(ptr));
#endif
}


template <typename T>
__forceinline__ __device__ T const_load(const T* ptr) {
    return cub::ThreadLoad<cub::LOAD_CS>(ptr);
}


