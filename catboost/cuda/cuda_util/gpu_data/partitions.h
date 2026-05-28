#pragma once

#include <util/system/types.h>

struct TDataPartition {
    ui32 Offset;
    ui32 Size;

    __host__ __device__ TDataPartition(ui32 offset = 0,
                                        ui32 size = 0)
        : Offset(offset)
        , Size(size)
    {
    }
};

