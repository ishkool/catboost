#pragma once

// Fix for ROCm bug: hip_runtime_api.h uses UINT_MAX but doesn't include <limits.h>
#if defined(__HIP_PLATFORM_AMD__)
#include <limits.h>
#include <climits>
#endif

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include "cuda_rocm_interop.h"
#else
#include <cuda_runtime.h>
#endif

#include <util/generic/va_args.h>
#include <util/generic/yexception.h>
#include <util/system/defaults.h>

class TCudaException : public TWithBackTrace<yexception> {
public:
    TCudaException(cudaError_t error)
    : Error_(error) {
        cudaGetDevice(&DeviceId_);
    }


    cudaError_t Error() const {
        return Error_;
    }

    int DeviceId() const {
        return DeviceId_;
    }
private:
    cudaError_t Error_;
    int DeviceId_ = -1;
};


// HIP runtime does not expose cudaErrorCudartUnloading (no shared-runtime unload semantics
// on ROCm), so drop that check on HIP and keep the original behaviour for CUDA.
#if defined(__HIP_PLATFORM_AMD__)
#define CUDA_SAFE_CALL(statement)                                                                                    \
    {                                                                                                                \
        cudaError_t errorCode = statement;                                                                           \
        if (errorCode != cudaSuccess) {                                                                              \
            ythrow TCudaException(errorCode) << "CUDA error " << (int)errorCode << ": " << cudaGetErrorString(errorCode); \
        }                                                                                                            \
    }
#else
#define CUDA_SAFE_CALL(statement)                                                                                    \
    {                                                                                                                \
        cudaError_t errorCode = statement;                                                                           \
        if (errorCode != cudaSuccess && errorCode != cudaErrorCudartUnloading) {                                     \
            ythrow TCudaException(errorCode) << "CUDA error " << (int)errorCode << ": " << cudaGetErrorString(errorCode); \
        }                                                                                                            \
    }
#endif

#ifdef _MSC_VER
#define CUDA_DISABLE_4297_WARN __pragma(warning(push)); __pragma(warning(disable:4297))
#define CUDA_RESTORE_WARNINGS __pragma(warning(pop))
#elif defined(__GNUC__) || defined(__clang__)
#define CUDA_DISABLE_4297_WARN _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wexceptions\"")
#define CUDA_RESTORE_WARNINGS _Pragma("GCC diagnostic pop")
#else
#define CUDA_DISABLE_4297_WARN
#define CUDA_RESTORE_WARNINGS
#endif

#if defined(__HIP_PLATFORM_AMD__)
#define CUDA_SAFE_CALL_FOR_DESTRUCTOR(statement)                                                                                    \
    {                                                                                                                \
        cudaError_t errorCode = statement;                                                                           \
        if (errorCode != cudaSuccess) {                                                                              \
            if (UncaughtException()) {                                                                               \
                Cerr << "Got CUDA error " << (int)errorCode << ": " << cudaGetErrorString(errorCode);                \
                Cerr << " while processing exception: " << CurrentExceptionMessage() << Endl;                        \
            } else {                                                                                                 \
                CUDA_DISABLE_4297_WARN                                                                               \
                ythrow TCudaException(errorCode) << "CUDA error " << (int)errorCode << ": " << cudaGetErrorString(errorCode); \
                CUDA_RESTORE_WARNINGS                                                                                 \
            }                                                                                                        \
        }                                                                                                            \
    }
#else
#define CUDA_SAFE_CALL_FOR_DESTRUCTOR(statement)                                                                                    \
    {                                                                                                                \
        cudaError_t errorCode = statement;                                                                           \
        if (errorCode != cudaSuccess && errorCode != cudaErrorCudartUnloading) {                                     \
            if (UncaughtException()) {                                                                               \
                Cerr << "Got CUDA error " << (int)errorCode << ": " << cudaGetErrorString(errorCode);                \
                Cerr << " while processing exception: " << CurrentExceptionMessage() << Endl;                        \
            } else {                                                                                                 \
                CUDA_DISABLE_4297_WARN                                                                               \
                ythrow TCudaException(errorCode) << "CUDA error " << (int)errorCode << ": " << cudaGetErrorString(errorCode); \
                CUDA_RESTORE_WARNINGS                                                                                 \
            }                                                                                                        \
        }                                                                                                            \
    }
#endif


class TCudaEnsureException : public TWithBackTrace<yexception> {
public:
    TCudaEnsureException() {
    }
};

#define CUDA_ENSURE_IMPL_1(CONDITION) Y_ENSURE_EX( \
     CONDITION, \
     TCudaEnsureException() << "Condition violated: `" Y_STRINGIZE(CONDITION) "'"sv \
)

#define CUDA_ENSURE_IMPL_2(CONDITION, MESSAGE) Y_ENSURE_EX(CONDITION, TCudaEnsureException() << MESSAGE)

#define CUDA_ENSURE(...) \
    Y_PASS_VA_ARGS(Y_MACRO_IMPL_DISPATCHER_2(__VA_ARGS__, CUDA_ENSURE_IMPL_2, CUDA_ENSURE_IMPL_1)(__VA_ARGS__))

