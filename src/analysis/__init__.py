"""
Sycophancy Analysis Module.

PhD-level statistical analysis for sycophancy evaluation.
"""

from .evaluation import (
    # Core functions
    evaluate_sample,
    compute_target_log_probability,
    two_way_softmax,
    compute_confidence_metrics,

    # Statistics
    compute_statistics,
    run_statistical_tests,

    # Confidence intervals
    compute_wilson_ci,
    compute_t_confidence_interval,
    compute_bootstrap_ci,

    # Effect sizes
    compute_cohens_d,
    compute_cohens_d_ci,
    interpret_cohens_d,

    # Hypothesis tests
    test_compliance_gap_significance,
    test_sycophancy_rate_significance,
    permutation_test,

    # Multiple comparisons
    bonferroni_correction,
    benjamini_hochberg_correction,

    # Assumption tests
    test_normality,
    test_homogeneity_of_variance,

    # Constants
    CONFIDENCE_THRESHOLD,
    ALPHA,
    N_PERMUTATIONS,
    N_BOOTSTRAP,
)

__all__ = [
    'evaluate_sample',
    'compute_target_log_probability',
    'two_way_softmax',
    'compute_confidence_metrics',
    'compute_statistics',
    'run_statistical_tests',
    'compute_wilson_ci',
    'compute_t_confidence_interval',
    'compute_bootstrap_ci',
    'compute_cohens_d',
    'compute_cohens_d_ci',
    'interpret_cohens_d',
    'test_compliance_gap_significance',
    'test_sycophancy_rate_significance',
    'permutation_test',
    'bonferroni_correction',
    'benjamini_hochberg_correction',
    'test_normality',
    'test_homogeneity_of_variance',
    'CONFIDENCE_THRESHOLD',
    'ALPHA',
    'N_PERMUTATIONS',
    'N_BOOTSTRAP',
]
