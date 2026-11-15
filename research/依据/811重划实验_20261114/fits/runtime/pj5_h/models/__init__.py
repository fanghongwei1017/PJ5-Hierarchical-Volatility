"""Public model API used by the inference and experiment layers.

Only these functions should be imported by code outside ``pj5_h.models``.
"""

from __future__ import annotations

from .cascade import (  # noqa: F401
    background_logpdf,
    background_pdf,
    invgamma_logpdf,
    invgamma_pdf,
    invgamma_rvs,
    return_cdf,
    return_kurtosis_theory,
    return_logpdf,
    return_logpdf_n1_closed,
    return_pdf,
    return_pdf_n1_closed,
    static_cascade_sample,
)
from .meijer import background_pdf as background_pdf_meijer  # noqa: F401
from .meijer import return_pdf as return_pdf_meijer  # noqa: F401

__all__ = [
    "background_pdf",
    "background_logpdf",
    "background_pdf_meijer",
    "return_pdf",
    "return_logpdf",
    "return_pdf_meijer",
    "return_pdf_n1_closed",
    "return_logpdf_n1_closed",
    "return_cdf",
    "return_kurtosis_theory",
    "invgamma_pdf",
    "invgamma_logpdf",
    "invgamma_rvs",
    "static_cascade_sample",
]
