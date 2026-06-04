.. meta::
   :description: Build CatBoost with ROCm support from source for AMD GPUs
   :keywords: amd, rocm, finance, financial, fintech, algorithm, gpu, install, setup, env, docker, package, contribute, develop, build, pip, make

.. _catboost-source-build:

**********************************
Building ROCm CatBoost from source
**********************************

Building CatBoost from the source material provided by AMD is generally reserved for developers who want to
contribute to the codebase. Use the following instructions to build from source. 

Prerequisites
=============

Before proceeding, ensure you have installed a supported ROCm version,
operating system, and Python environment that are compatible with the
ROCm-Finance libraries. Verify that your system includes a supported AMD
Instinct GPU. For more information, see `ROCm-Finance installation prerequisites
<https://rocm.docs.amd.com/projects/rocm-finance-internal/en/main/install/prerequisites.html>`__.

For a consistent and streamlined setup experience, it's recommended to use
a ROCm development environment Docker container. See `Install ROCm-Finance
<https://rocm.docs.amd.com/projects/rocm-finance-internal/en/main/install/install.html>`__
for instructions.

.. _catboost-build:

Build from source
==================

#. Install ROCm first. 

   To get up and running quickly, a prebuilt ROCm-enabled container is recommended. The easiest way is to use the official ROCm Docker images from Docker Hub. For more information, see see `Running ROCm Docker containers <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.2/how-to/docker.html>`__.

   .. code-block:: bash
      
      docker run \
        --cap-add=SYS_PTRACE \
        --ipc=host \
        --privileged=true \
        --shm-size=128GB \
        --network=host \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        -it \
        -v $HOME:$HOME \
        --name rocm7 \
        rocm/dev-ubuntu-24.04:7.0.2-complete

#. Add required packages for development dependencies.

   .. code-block:: bash
      
      apt-get update
      apt-get install -y --no-install-recommends \
            ca-certificates \
            curl \
            gpg \
            wget
      # Add Kitware APT repo for newer CMake on Ubuntu 24.04 (Noble)
      wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
            | gpg --dearmor > /usr/share/keyrings/kitware-archive-keyring.gpg; \
         echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' \
            > /etc/apt/sources.list.d/kitware.list; \

#. Install the development environment setup.

   .. code-block:: bash
      
      apt-get update
      apt-get install -y \
            cmake \
            libgtest-dev \
            libgmock-dev \
            ninja-build \
            libssl-dev \
            libc++-dev \
            libc++abi-dev \
            libstdc++-12-dev \
            ragel \
            yasm \
            liblapack-dev \
            libblas-dev \
            lld \
            python3.12-venv
   
#. Clone the GitHub repository.

   .. code-block:: bash
      
      git clone https://github.com/ROCm/catboost.git

#. Create a virtual environment for development.

   .. code-block:: bash
      
      python3 -m venv .catboost-dev
      source .catboost-dev/bin/activate

#. Set the GPU and compiler targets.

   .. code-block:: bash
      
      # GPU and ROCm/Compiler environment
      export ROCM_PATH="/opt/rocm"
      export HIP_PATH="$ROCM_PATH"
      export CMAKE_PREFIX_PATH="$ROCM_PATH"
      export PATH="$ROCM_PATH/bin:$PATH"
      export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"
      export CC="$ROCM_PATH/llvm/bin/clang"
      export CXX="$ROCM_PATH/llvm/bin/clang++"
      export CMAKE_C_COMPILER="$ROCM_PATH/llvm/bin/clang"
      export CMAKE_CXX_COMPILER="$ROCM_PATH/llvm/bin/clang++"

#. Build the wheel file.

   .. code-block:: bash
      
      pip install conan numpy "cython>=3.0.10" traitlets numba ipython ipywidgets "setuptools>=65,<81" build wheel
      python setup.py bdist_wheel --with-cuda=${ROCM_PATH} --no-widget --parallel=$(nproc)

#. Build the test tools.

   .. code-block:: bash
      
      ninja model_comparator
      ninja resort_dsv_by_numeric_group_id
      ninja model_perftest
      ninja limited_precision_json_diff
      ninja limited_precision_dsv_diff

#. Build the CLI binary.

   .. code-block:: bash
      
      cmake -B build_hipgpu -G Ninja \
         -DCMAKE_BUILD_TYPE=Release \
         -DCATBOOST_COMPONENTS=app \
         -DHAVE_CUDA=yes \
         -DHAVE_ROCM=yes \
         -DCMAKE_C_COMPILER="${ROCM_PATH}/llvm/bin/clang" \
         -DCMAKE_CXX_COMPILER="${ROCM_PATH}/llvm/bin/clang++" \
         -DCMAKE_HIP_COMPILER="${ROCM_PATH}/llvm/bin/clang++" \
         -DCMAKE_PREFIX_PATH="${ROCM_PATH}"
      ln -sf "$(which yasm)" build_hipgpu/bin/yasm
      ln -sf "$(which ragel)" build_hipgpu/bin/ragel
      cmake --build build_hipgpu --target catboost 


.. _catboost-test:

Running unit tests
==================

Unit tests for CatBoost are available in both C++ and Python. See the sections below for the appropriate tests. 

Python unit tests
-----------------

To run the Python unit tests, follow the steps below.

#. Install the package under a virtual environment. Use conda, uv, or the Python venv to create one.
#. Install standard Python dependencies for testing.

   .. code-block:: bash
      
      pip install --upgrade --force-reinstall testpath numpy pandas scikit-learn scipy setuptools matplotlib cloudpickle pytest psutil joblib pyarrow dask traitlets numba ipython ipywidgets

#. Set environment variables for tests.

   .. code-block:: bash
      
      export YA_DIFF_TOOL=/usr/bin/diff
      export CMAKE_SOURCE_DIR=$PWD
      export CMAKE_BINARY_DIR=$PWD/build_hipgpu
      export TEST_OUTPUT_DIR="/tmp/catboost_test_output"
      export HAVE_CUDA=yes
      export HAVE_ROCM=yes

#. Run the Python API tests.

   .. code-block:: bash
      
      pytest catboost/python-package/ut/medium/test.py -v

C++ unit tests
--------------

To run the C++ unit tests, follow the steps below.

#. Ensure that you have a successful build of the CatBoost CLI as mentioned in the :ref:`catboost-build` steps above.
#. Set up the environment for CLI tests.

   .. code-block:: bash
      
      export CMAKE_BINARY_DIR=$PWD/build_hipgpu
      export CMAKE_SOURCE_DIR=$PWD
      export TEST_OUTPUT_DIR="/tmp/catboost_test_output"
      export YA_DIFF_TOOL=/usr/bin/diff
      export HAVE_CUDA=yes
      export HAVE_ROCM=yes

#. Run CLI tests.

   .. code-block:: bash
      
      cd catboost/pytest
      pytest -v 
