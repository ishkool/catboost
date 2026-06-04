// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#pragma once

// Workaround for ROCm bug: hip_runtime_api.h uses UINT_MAX without including limits.h
// UINT_MAX, INT_MAX, CHAR_BIT are defined globally via CMake add_definitions()
// so we don't include <limits.h> here to avoid circular includes
// (since this file is named limits.h and is in the include path)

#include <thrust/detail/config.h>

#include <thrust/detail/type_traits.h>

#include <limits>

THRUST_NAMESPACE_BEGIN

template <typename T>
struct numeric_limits : std::numeric_limits<T>
{};

THRUST_NAMESPACE_END



