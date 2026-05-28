#include <catboost/libs/gpu_config/interface/get_gpu_device_count.h>

#include <catboost/libs/logging/logging.h>

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace NCB {

    int GetGpuDeviceCount() {
        int deviceCount = 0;

#if defined(__HIP_PLATFORM_AMD__)
        hipError_t status;
        if (hipSuccess != (status = hipGetDeviceCount(&deviceCount))) {
            CATBOOST_WARNING_LOG << "Error " << int(status) << " (" << hipGetErrorString(status) << ") ignored while obtaining device count" << Endl;
        }
#else
        cudaError_t status;
        if (cudaSuccess != (status = cudaGetDeviceCount(&deviceCount))) {
            CATBOOST_WARNING_LOG << "Error " << int(status) << " (" << cudaGetErrorString(status) << ") ignored while obtaining device count" << Endl;
        }
#endif
        return deviceCount;
    }

}

