.. meta::
   :description: Use CatBoost with ROCm support on AMD GPUs
   :keywords: amd, rocm, finance, financial, fintech, algorithm, gpu

*******************************
CatBoost for ROCm documentation
*******************************

CatBoost for ROCm provides GPU‑accelerated gradient boosting on AMD hardware,
enabling scalable, high‑performance machine learning for financial risk
modeling and data‑intensive workloads. This implementation utilizes optimized
kernels, enhanced memory management, and multi‑GPU scaling to accelerate
performance relative to CPU‑only baselines.

CatBoost is part of the `ROCm-Finance toolkit
<https://rocm.docs.amd.com/projects/rocm-finance-internal/en/main/>`__.

.. note:: 
   CatBoost is available for early access and should not be used for production workloads. 

The ROCm-Finance CatBoost source code is hosted on GitHub at
`<https://github.com/ROCm/catboost/>`__.

ROCm-Finance CatBoost documentation is organized into the following categories:

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Install

      * :doc:`/install/install`
      * :doc:`/install/build-from-source`

   .. grid-item-card:: Reference

      * `CatBoost documentation (upstream) <https://catboost.ai/docs/en/>`__
      * `Python quick start (upstream) <https://catboost.ai/docs/en/concepts/python-quickstart>`__
      * `R package quick start (upstream) <https://catboost.ai/docs/en/concepts/r-quickstart>`__
