"""Inference layer: window selection (Eq. 23), hierarchy fit (Eq. 24), hold-out scoring."""

from .binning import BinningSpec, build_binning, freedman_diaconis_bins  # noqa: F401
from .fit import (  # noqa: F401
    FitError,
    compound_gaussian_logpdf,
    eq24_error,
    fit_beta,
    fit_hierarchy,
    rolling_variance,
    select_window,
)
from .score import (  # noqa: F401
    GaussianModel,
    HModel,
    StudentTModel,
    block_bootstrap_difference,
    estimate_block_length,
    fit_gaussian,
    fit_student_t_n1,
    score_logpdf,
)

__all__ = [
    "BinningSpec", "build_binning", "freedman_diaconis_bins",
    "rolling_variance", "compound_gaussian_logpdf",
    "FitError", "eq24_error", "fit_beta", "fit_hierarchy", "select_window",
    "GaussianModel", "StudentTModel", "HModel", "fit_gaussian", "fit_student_t_n1",
    "score_logpdf", "block_bootstrap_difference", "estimate_block_length",
]
