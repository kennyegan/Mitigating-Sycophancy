"""
Unit tests for shared evaluation/statistics utilities.
"""

from src.analysis import evaluation as eval_utils


def test_two_way_softmax_sums_to_one():
    p_a, p_b = eval_utils.two_way_softmax(-7.3, -9.1)
    assert abs((p_a + p_b) - 1.0) < 1e-9


def test_length_normalized_confidence_can_recover_long_target_case():
    # Legacy max-log confidence says "uncertain" here, but per-token average is confident.
    legacy = eval_utils.compute_confidence_metrics(-6.0, -12.0)
    normalized = eval_utils.compute_length_normalized_confidence_metrics(
        -6.0, -12.0, len_a=1, len_b=3
    )
    assert legacy["is_confident"] is False
    assert normalized["is_confident"] is True


def test_exact_binomial_test_prefers_modern_api(monkeypatch):
    class FakeResult:
        pvalue = 0.1234

    class FakeStats:
        @staticmethod
        def binomtest(successes, n, p, alternative="two-sided"):
            return FakeResult()

    monkeypatch.setattr(eval_utils, "scipy_stats", FakeStats())
    pval = eval_utils.exact_binomial_test(8, 10, 0.5)
    assert pval == 0.1234


def test_exact_binomial_test_falls_back_to_legacy_api(monkeypatch):
    class FakeLegacyStats:
        @staticmethod
        def binom_test(successes, n, p, alternative="two-sided"):
            return 0.4321

    monkeypatch.setattr(eval_utils, "scipy_stats", FakeLegacyStats())
    pval = eval_utils.exact_binomial_test(8, 10, 0.5)
    assert pval == 0.4321
