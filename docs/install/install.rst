.. meta::
   :description: Install ROCm CatBoost with ROCm support for AMD GPUs
   :keywords: amd, rocm, catboost, finance, financial, fintech, algorithm, gpu, install, setup, env, docker, pip, package, quick, start, lib

.. _catboost-install:

******************************
Installing CatBoost for ROCm
******************************

These instructions are for users looking to install and run CatBoost on their projects. Developers looking to contribute their own changes to the code, or who want a custom configuration to support specific versions of Linux and Python which are not natively supported, should refer to :ref:`catboost-source-build`.

.. _catboost-install-prerequisites:

Prerequisites
=============

Before proceeding, ensure that you have installed a supported ROCm version,
operating system, and Python version that are compatible with the ROCm-Finance
libraries. Verify that your system includes an AMD GPU fully supported by ROCm-Finance.
For more information, see `ROCm-Finance installation prerequisites
<https://rocm.docs.amd.com/projects/rocm-finance-internal/en/main/install/prerequisites.html>`__.

For a consistent and streamlined setup experience, it's recommended to use
a ROCm development environment Docker container. See
`Install ROCm-Finance <https://rocm.docs.amd.com/projects/rocm-finance-internal/en/main/install/install.html>`__ for instructions.

.. _catboost-pip:

Install using pip
=================

To get up and running quickly, a prebuilt ROCm-enabled container is recommended. The easiest way is to use the official ROCm
Docker images from Docker Hub. For more information, see `Running ROCm Docker containers <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.2/how-to/docker.html>`__.

For complete instructions on installing ROCm refer to `ROCm installation for Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.2/index.html>`__.

Install for ROCm 7.0.2
----------------------

1. Install the Docker container: 

   .. code-block:: shell
      
      docker pull rocm/dev-ubuntu-24.04:7.0.2-complete

2. Launch the Docker container: 

   .. code-block:: shell
      
      docker run -it \
         --cap-add=SYS_PTRACE \
         --ipc=host \
         --privileged=true \
         --shm-size=128GB \
         --network=host \
         --device=/dev/kfd \
         --device=/dev/dri \
         --group-add video \
         -v $HOME:$HOME \
         --name rocm7 \
         rocm/dev-ubuntu-24.04:7.0.2-complete

3. Install the libraries from the AMD-hosted PyPI repository:

   .. code-block:: shell
      
      pip install amd_catboost --extra-index-url=https://pypi.amd.com/rocm-7.0.2/simple/


.. _catboost-verify:

Verify your installation
========================

Verify the installation using the ``pip show`` command:

.. code-block:: shell
   
   pip show -v amd_catboost
   
   Name: amd_catboost
   Version: 1.2.8
   Summary: CatBoost Python Package
   Home-page: https://catboost.ai
   Author: CatBoost Developers
   Author-email:
   License: Apache License, Version 2.0
   Location: /.venv/lib/python3.12/site-packages
   Requires: graphviz, matplotlib, numpy, pandas, plotly, scipy, six
   Required-by:
   Metadata-Version: 2.4
   Installer: pip
   Classifiers:
   Development Status :: 5 - Production/Stable
   Topic :: Scientific/Engineering :: Artificial Intelligence
   Topic :: Software Development :: Libraries :: Python Modules
 

Run a sample script to verify and get started. 

.. code-block:: python
   
   import numpy
   from catboost import CatBoostRegressor
   dataset = numpy.array([[1,4,5,6],[4,5,6,7],[30,40,50,60],[20,15,85,60]])
   train_labels = [1.2,3.4,9.5,24.5]
   model = CatBoostRegressor(learning_rate=1, depth=6, loss_function='RMSE')
   fit_model = model.fit(dataset, train_labels)
   print(fit_model.get_params())
