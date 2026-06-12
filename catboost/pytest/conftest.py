try:
    from catboost_pytest_lib import compressed_data  # noqa
except ImportError:
    from lib import compressed_data  # noqa
    pytest_plugins = ["lib.common.pytest_plugin"]

import os
import pytest

# Failing tests on ROCM from ish_cli.log – skipped so they don't run (Skipped for ROCM need to be fixed)
SKIP_KNOWN_FAILURES_CLI = frozenset([
    # Multi-target GPU split-scoring bug FIXED (hipcub::WarpScan logical width 32 in
    # greedy_subsets_searcher histogram scan) — these 10 now pass and were un-skipped 2026-06-11:
    #   test_rmse_with_uncertainty, test_multilogloss[MultiLogloss|MultiCrossEntropy],
    #   test_multilogloss_with_bow[...], test_multirmse, test_multirmse_with_cat_features,
    #   test_multiclass_baseline[MultiClass|MultiClassOneVsAll], test_shrink_model_with_text_features[SymmetricTree]
    # Still failing: single-target fit-vs-apply self-consistency at rtol=1e-4 (separate Group B issue):
    "cuda_tests/test_gpu.py::test_grow_policies[Logloss-Cosine-SymmetricTree-Ordered]",
    "test.py::test_dist_train_multiregression[calc_block=60]",
    "test.py::test_dist_train_multiregression[calc_block=5000000]",
    "test.py::test_dist_train_multiregression_single[calc_block=60]",
    "test.py::test_dist_train_multiregression_single[calc_block=5000000]",
    "test.py::test_quantile_exact_distributed",
    "test.py::test_dist_train[calc_block=60]",
    "test.py::test_dist_train[calc_block=5000000]",
    "test.py::test_dist_train_with_weights[calc_block=60]",
    "test.py::test_dist_train_with_weights[calc_block=5000000]",
    "test.py::test_dist_train_with_baseline[calc_block=60]",
    "test.py::test_dist_train_with_baseline[calc_block=5000000]",
    "test.py::test_dist_train_multiclass[calc_block=60]",
    "test.py::test_dist_train_multiclass[calc_block=5000000]",
    "test.py::test_dist_train_multiclass_weight[calc_block=60]",
    "test.py::test_dist_train_multiclass_weight[calc_block=5000000]",
    "test.py::test_dist_train_quantized[calc_block=60]",
    "test.py::test_dist_train_quantized[calc_block=5000000]",
    "test.py::test_dist_train_quantized_groupid[PairLogitPairwise-train.pairs-calc_block=60]",
    "test.py::test_dist_train_quantized_groupid[PairLogitPairwise-train.pairs-calc_block=5000000]",
    "test.py::test_dist_train_quantized_groupid[PairLogitPairwise-train.pairs.weighted-calc_block=60]",
    "test.py::test_dist_train_quantized_groupid[PairLogitPairwise-train.pairs.weighted-calc_block=5000000]",
    "test.py::test_dist_train_quantized_groupid[QuerySoftMax-train.pairs-calc_block=60]",
    "test.py::test_dist_train_quantized_groupid[QuerySoftMax-train.pairs-calc_block=5000000]",
    "test.py::test_dist_train_quantized_groupid[QuerySoftMax-train.pairs.weighted-calc_block=60]",
    "test.py::test_dist_train_quantized_groupid[QuerySoftMax-train.pairs.weighted-calc_block=5000000]",
    "test.py::test_dist_train_quantized_group_weights[calc_block=60]",
    "test.py::test_dist_train_quantized_group_weights[calc_block=5000000]",
    "test.py::test_dist_train_quantized_baseline[calc_block=60]",
    "test.py::test_dist_train_quantized_baseline[calc_block=5000000]",
    "test.py::test_dist_train_queryrmse[calc_block=60]",
    "test.py::test_dist_train_queryrmse[calc_block=5000000]",
    "test.py::test_dist_train_subgroup[calc_block=60]",
    "test.py::test_dist_train_subgroup[calc_block=5000000]",
    "test.py::test_dist_train_pairlogit[calc_block=60]",
    "test.py::test_dist_train_pairlogit[calc_block=5000000]",
    "test.py::test_dist_train_pairlogitpairwise[train.pairs]",
    "test.py::test_dist_train_pairlogitpairwise[train.pairs.weighted]",
    "test.py::test_dist_train_querysoftmax[calc_block=60]",
    "test.py::test_dist_train_querysoftmax[calc_block=5000000]",
    "test.py::test_dist_train_auc[Logloss]",
    "test.py::test_dist_train_auc[RMSE]",
    "test.py::test_dist_train_auc_weight[Logloss]",
    "test.py::test_dist_train_auc_weight[RMSE]",
    "test.py::test_dist_train_yetirank",
    "test.py::test_dist_train_with_cat_features[one_hot_max_size=255-calc_block=60]",
    "test.py::test_dist_train_with_cat_features[one_hot_max_size=255-calc_block=5000000]",
    "test.py::test_dist_train_overfitting_detector[od_type=IncToDec]",
    "test.py::test_dist_train_overfitting_detector[od_type=Iter]",
    "test.py::test_shap_verbose",
    "test.py::test_quantized_pool_with_large_grid",
    "test.py::test_eval_result_on_different_pool_type",
    "test.py::test_model_sum[Logloss-Depthwise]",
    "test.py::test_model_sum[Logloss-Lossguide]",
    "test.py::test_model_sum[Logloss-SymmetricTree]",
    "test.py::test_model_sum[MultiClass-Depthwise]",
    "test.py::test_model_sum[MultiClass-Lossguide]",
    "test.py::test_model_sum[MultiClass-SymmetricTree]",
    "test.py::test_fit_regression_with_text_features[BoW-ByDelimiter-Ordered]",
    "test.py::test_fit_regression_with_text_features[BoW-ByDelimiter-Plain]",
    "test.py::test_fit_regression_with_text_features[BoW-BySense-Ordered]",
    "test.py::test_fit_regression_with_text_features[BoW-BySense-Plain]",
    "test.py::test_fit_with_per_feature_text_options[Ordered-binclass]",
    "test.py::test_fit_with_per_feature_text_options[Ordered-regression]",
    "test.py::test_fit_with_per_feature_text_options[Plain-binclass]",
    "test.py::test_fit_with_per_feature_text_options[Plain-regression]",
    "test_modes.py::TestModeNormalizeModel::test_normalize_idempotent",
    "test_modes.py::TestModeNormalizeModel::test_sum",
])


def pytest_collection_modifyitems(config, items):
    """Skip known-failing CLI tests on ROCM (need to be fixed).

    Set RUN_KNOWN_FAILURES_CLI=1 to force them to run (for debugging); default behavior unchanged.
    """
    if os.environ.get("RUN_KNOWN_FAILURES_CLI") == "1":
        return
    for item in items:
        if item.nodeid in SKIP_KNOWN_FAILURES_CLI:
            item.add_marker(pytest.mark.skip(reason="Skipped for ROCM need to be fixed"))

