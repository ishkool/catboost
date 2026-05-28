# ROCm builds pass both HAVE_CUDA=yes (GPU support) and HAVE_ROCM=yes; only enable HIP.
# NVIDIA CUDA builds pass HAVE_CUDA=yes only.
if (HAVE_ROCM AND HAVE_CUDA)
  if(${CMAKE_VERSION} VERSION_LESS "3.17.0")
      message(FATAL_ERROR "Build with HIP requires at least cmake 3.17.0")
  endif()

  message(STATUS "=== Building with AMD ROCm/HIP support ===")
  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 17)


  include(global_flags)
  include(common)

  function(quote_if_contains_spaces OutVar Var)
    if (Var MATCHES ".*[ ].*")
      set(${OutVar} "\"${Var}\"" PARENT_SCOPE)
    else()
      set(${OutVar} ${Var} PARENT_SCOPE)
    endif()
  endfunction()

  function(get_hip_flags_from_cxx_flags OutHipFlags CxxFlags)
    # OutHipFlags is an output string
    # CxxFlags is a string

    set(skipList
      -gline-tables-only
      /Zc:inline # disable unreferenced functions (kernel registrators) remove
      -Wno-c++17-extensions
      #-flto
      #-faligned-allocation
      #-fsized-deallocation
      # While it might be reasonable to compile host part of .cu sources with these optimizations enabled,
      # nvcc passes these options down towards cicc which lacks x86_64 extensions support.
      #-msse2
      #-msse3
      #-mssse3
      #-msse4.1
      #-msse4.2
    )

    set(skipPrefixRegexp
      "(-fsanitize=|-fsanitize-coverage=|-fsanitize-blacklist=|--system-header-prefix|(/|-)std(:|=)c\\+\\+).*"
    )

    string(FIND "${CMAKE_CXX_COMPILER}" clang hostCompilerIsClangPos)
    string(COMPARE NOTEQUAL ${hostCompilerIsClangPos} -1 isHostCompilerClang)


    function(separate_arguments_with_special_symbols Output Src)
      string(REPLACE ";" "$<SEMICOLON>" LocalOutput "${Src}")
      separate_arguments(LocalOutput NATIVE_COMMAND ${LocalOutput})
      set(${Output} ${LocalOutput} PARENT_SCOPE)
    endfunction()

    separate_arguments_with_special_symbols(Separated_CxxFlags "${CxxFlags}")

    if (MSVC)
      set(flagPrefixSymbol "/")
    else()
      set(flagPrefixSymbol "-")
    endif()

    set(localHipCommonFlags "") # non host compiler options
    set(localHipCompilerOptions "")

    while (Separated_CxxFlags)
      list(POP_FRONT Separated_CxxFlags cxxFlag)
      if ((cxxFlag IN_LIST skipList) OR (cxxFlag MATCHES ${skipPrefixRegexp}))
        continue()
      endif()
      if ((cxxFlag STREQUAL -fopenmp=libomp) AND (NOT isHostCompilerClang))
        list(APPEND localHipCompilerOptions -fopenmp)
        continue()
      endif()
      if ((NOT isHostCompilerClang) AND (cxxFlag MATCHES "^\-\-target=.*"))
        continue()
      endif()
      if (cxxFlag MATCHES "^${flagPrefixSymbol}(D[^ ]+)=(.+)")
        set(key ${CMAKE_MATCH_1})
        quote_if_contains_spaces(safeValue "${CMAKE_MATCH_2}")
        list(APPEND localHipCommonFlags "-${key}=${safeValue}")
        continue()
      endif()
      if (cxxFlag MATCHES "^${flagPrefixSymbol}([DI])(.*)")
        set(key ${CMAKE_MATCH_1})
        if (CMAKE_MATCH_2)
          set(value ${CMAKE_MATCH_2})
          set(sep "")
        else()
          list(POP_FRONT Separated_CxxFlags value)
          set(sep " ")
        endif()
        quote_if_contains_spaces(safeValue "${value}")
        list(APPEND localHipCommonFlags "-${key}${sep}${safeValue}")
        continue()
      endif()
      list(APPEND localHipCompilerOptions ${cxxFlag})
    endwhile()

    if (isHostCompilerClang)
      # hipcc/clang reports unused things from .h files
      list(APPEND localHipCommonFlags -Wno-unused-function -Wno-unused-parameter)
      if (CMAKE_CXX_COMPILER_TARGET)
        list(APPEND localHipCompilerOptions "--target=${CMAKE_CXX_COMPILER_TARGET}")
      endif()
    endif()

    if (CMAKE_SYSROOT)
      list(APPEND localHipCompilerOptions "--sysroot=${CMAKE_SYSROOT}")
    endif()

    list(JOIN localHipCommonFlags " " joinedLocalHipCommonFlags)
    string(REPLACE "$<SEMICOLON>" ";" joinedLocalHipCommonFlags "${joinedLocalHipCommonFlags}")
    list(JOIN localHipCompilerOptions " " joinedLocalHipCompilerOptions)
    set(${OutHipFlags} "${joinedLocalHipCommonFlags} ${joinedLocalHipCompilerOptions}" PARENT_SCOPE)
  endfunction()

  get_hip_flags_from_cxx_flags(CMAKE_HIP_FLAGS "${CMAKE_CXX_FLAGS}")

  # HIP-specific flags
  string(APPEND CMAKE_HIP_FLAGS
    " -D__HIP_PLATFORM_AMD__=1"
    " -D__HIP__=1"
    " -DHIP_ENABLE_PRINTF=1"
    " \"-D__forceinline__=inline __attribute__((always_inline))\""
  )
 
  if(DEFINED ENV{ENABLE_CODE_COVERAGE}) 
    string(APPEND CMAKE_HIP_FLAGS
      " -fprofile-instr-generate -fcoverage-mapping"
    )
  endif()

  # Add clang's parent include directory for cuda_wrappers
  # amd_hip_runtime.h includes <include/cuda_wrappers/algorithm> which expects
  # the parent directory of clang's include to be in the system path
  if(DEFINED ENV{ROCM_PATH})
    set(ROCM_CLANG_PATH "$ENV{ROCM_PATH}/lib/llvm/lib/clang")
  else()
    set(ROCM_CLANG_PATH "/opt/rocm/lib/llvm/lib/clang")
  endif()
  file(GLOB CLANG_VERSIONS "${ROCM_CLANG_PATH}/*")
  if(CLANG_VERSIONS)
    list(SORT CLANG_VERSIONS)
    list(REVERSE CLANG_VERSIONS)
    list(GET CLANG_VERSIONS 0 CLANG_VER_DIR)
    message(STATUS "Using clang include directory: ${CLANG_VER_DIR}")
    string(APPEND CMAKE_HIP_FLAGS " -isystem ${CLANG_VER_DIR}")
  endif()
  
  # Workaround for ROCm bugs:
  # 1. hip_runtime_api.h uses UINT_MAX without including limits.h
  # 2. __clang_cuda_complex_builtins.h uses max/min without defining them
  # Solution: Define the include guard to prevent loading the problematic complex builtins header
  string(APPEND CMAKE_HIP_FLAGS
    " -DUINT_MAX=4294967295U"
    " -DINT_MAX=2147483647"
    " -DCHAR_BIT=8"
    " -D__CLANG_CUDA_COMPLEX_BUILTINS"
  )
  
  # Note: max/min workaround for __clang_cuda_complex_builtins.h is now in
  # contrib/libs/amd/thrust/limits.h which is included early in the HIP wrapper chain

  set(HIP_STD_VER 17)
  if(MSVC)
    set(HIP_STD "/std:c++${HIP_STD_VER}")
  else()
    set(HIP_STD "-std=c++${HIP_STD_VER}")
  endif()
  string(APPEND CMAKE_HIP_FLAGS " ${HIP_STD}")

  string(APPEND CMAKE_HIP_FLAGS " -DTHRUST_IGNORE_CUB_VERSION_CHECK")

  # use AMD versions from contrib
  # Include the parent directory so #include <thrust/...> and #include <hipcub/...> work
  # This prevents our thrust/limits.h from shadowing system <limits.h>
  set(CUDA_EXTRA_INCLUDE_DIRECTORIES
    ${PROJECT_SOURCE_DIR}/contrib/libs/amd
  )

  find_package(HIP REQUIRED)

  # Set arch for MI355 (gfx950), MI300A/X (gfx942), MI200 (gfx90a) and MI100 (gfx908)
  if(NOT DEFINED AMDGPU_TARGETS)
    set(AMDGPU_TARGETS "gfx908;gfx90a;gfx942;gfx950" CACHE STRING "AMD GPU targets")
  endif()
  
  # Convert to --offload-arch flags
  string(REPLACE ";" " --offload-arch=" OFFLOAD_ARCH_FLAGS "${AMDGPU_TARGETS}")
  string(APPEND CMAKE_HIP_FLAGS " --offload-arch=${OFFLOAD_ARCH_FLAGS}")

  # Set stdlib to use libstdc++
  string(APPEND CMAKE_HIP_FLAGS " -stdlib=libstdc++")

  message(VERBOSE "CMAKE_HIP_FLAGS = \"${CMAKE_HIP_FLAGS}\"")
  message(STATUS "CMAKE_HIP_FLAGS = \"${CMAKE_HIP_FLAGS}\"")

  function(target_cuda_flags Tgt)
    # For ROCm/HIP builds, ignore NVIDIA-specific -gencode flags
    # HIP uses --offload-arch which is already set in CMAKE_HIP_FLAGS
    # This function is a no-op for HIP to prevent CUDA architecture flags from being passed
  endfunction()

  function(target_cuda_cflags Tgt)
    # For ROCm/HIP builds, ignore CUDA-specific compiler options
    # HIP compiler options are already handled in CMAKE_HIP_FLAGS
    # This function is a no-op for HIP
  endfunction()

  function(target_cuda_sources Tgt Scope)
    # add include directories on per-CMakeLists file level because some non-CUDA source files may want to include calls to CUDA libs
    include_directories(${CUDA_EXTRA_INCLUDE_DIRECTORIES})

    set_source_files_properties(${ARGN} PROPERTIES
	    LANGUAGE HIP
	    COMPILE_OPTIONS "$<JOIN:$<TARGET_GENEX_EVAL:${Tgt},$<TARGET_PROPERTY:${Tgt},CUDA_FLAGS>>,;>"
    )
    target_sources(${Tgt} ${Scope} ${ARGN})
  endfunction()

elseif (HAVE_CUDA)
  # Upstream NVIDIA CUDA path (nvcc); kept in cuda_nvidia.cmake for clarity.
  include(${CMAKE_CURRENT_LIST_DIR}/cuda_nvidia.cmake)

endif()





