/*!
 * Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef CATBOOST_INCLUDE_CATBOOST_CUDA_CUDA_ROCM_INTEROP_H_
#define CATBOOST_INCLUDE_CATBOOST_CUDA_CUDA_ROCM_INTEROP_H_


#if defined(__HIP_PLATFORM_AMD__)

// ROCm doesn't have atomicAdd_block, but it should be semantically the same as atomicAdd
#define atomicAdd_block atomicAdd

// hipify
#include <hip/hip_runtime.h>
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaFree hipFree
#define cudaFreeHost hipFreeHost
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaHostAlloc hipHostAlloc
#define cudaHostAllocPortable hipHostAllocPortable
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemoryTypeHost hipMemoryTypeHost
#define cudaMemset hipMemset
#define cudaPointerAttributes hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaSetDevice hipSetDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
//#define cudaErrorCudartUnloading hipErrorCudartUnloading
#define cudaEvent_t hipEvent_t
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaEventDestroy hipEventDestroy
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaEventQuery hipEventQuery
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaMemoryTypeDevice hipMemoryTypeDevice
#define cudaEventRecord hipEventRecord
#define cudaStreamPerThread hipStreamPerThread
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemoryTypeManaged hipMemoryTypeManaged
#define cudaMemoryType hipMemoryType
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorUnknown hipErrorUnknown
#define cudaErrorNotYetImplemented hipErrorNotSupported
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventBlockingSync hipEventBlockingSync
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaGraphExec_t hipGraphExec_t
#define cudaGraph_t hipGraph_t
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy 
#define cudaGraphInstantiate hipGraphInstantiate
#define cudaGraphLaunch hipGraphLaunch 
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaMemsetAsync hipMemsetAsync

// ROCm now ships real CUDA-compatible inline templates for __shfl_*_sync in
// <hip/amd_detail/amd_warp_sync_functions.h> (since ~ROCm 7.0; mandatory on
// ROCm 7.2+ because newer ROCm headers like amd_hip_bf16.h call these with
// the full 4-arg form `__shfl_*_sync(mask, var, delta, width)`).
//
// We intentionally do NOT macro-redefine __shfl_down_sync / __shfl_up_sync
// here anymore: a 3-arg macro shim shadows the real inline templates and
// breaks the 4-arg call sites inside ROCm headers ("too many arguments
// provided to function-like macro invocation"). The real templates default
// `width = warpSize`, so existing 3-arg call sites in CatBoost still work.
//
// Portable full-warp mask. ROCm 7.2+ enforces a static_assert that
// sizeof(MaskT)==8 (because AMD wavefronts can be 64 lanes wide), so passing
// the literal 0xFFFFFFFF (unsigned int, 4 bytes) is now a hard compile error.
// CUDA's __shfl_*_sync mask is always 32-bit unsigned. Use the wide literal
// for HIP and the narrow one for CUDA so each platform's signature is
// satisfied without warnings.

// warpSize is only allowed for device code.
// HIP header used to define warpSize as a constexpr that was either 32 or 64
// depending on the target device, and then always set it to 64 for host code.
static inline constexpr int WARP_SIZE_INTERNAL() {
#if defined(__GFX9__)
  return 64;
#else  // __GFX9__
  return 32;
#endif  // __GFX9__
}
#define WARPSIZE (WARP_SIZE_INTERNAL())

#define CATBOOST_FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFULL

#else  // __HIP_PLATFORM_AMD__
// CUDA warpSize is not a constexpr, but always 32
#define WARPSIZE 32

#define CATBOOST_FULL_WARP_MASK 0xFFFFFFFFu
#endif  // defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)



#endif  // CATBOOST_INCLUDE_CATBOOST_CUDA_CUDA_ROCM_INTEROP_H_

