"""Reconstruct the two stages of the static mixture from the Ping An training set."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import invgamma, norm

FROZEN = Path(__file__).resolve().parents[1] / "依据/811重划实验_20261114"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=True)
    fit = json.loads((FROZEN / "fits/601318.SH_5min_raw.base.json").read_text("utf-8"))
    module = importlib.util.spec_from_file_location("static_engine", FROZEN / "fits/runtime/fit_engine.py")
    engine = importlib.util.module_from_spec(module)
    module.loader.exec_module(engine)

    frame = pd.read_csv(FROZEN / "data/historical/returns_601318.SH_5min.csv.gz",
                        usecols=["block", "trading_date", "session_id", "slot_index", "raw_log_return"],
                        float_precision="round_trip")
    training = frame.loc[frame.block.eq("train")].sort_values(["trading_date", "session_id", "slot_index"]).copy()
    training["x"] = training.raw_log_return / fit["scale"]
    windows = []
    for _, group in training.groupby(["trading_date", "session_id"]):
        if not np.all(np.diff(group.slot_index) == 1):
            raise ValueError("Example requires contiguous slots within each Ping An training session")
        windows.extend(np.var(np.lib.stride_tricks.sliding_window_view(group.x, fit["M_star"]),
                              axis=1, ddof=1))
    variances = np.asarray(windows)
    if len(variances) != fit["n_windows_train"]:
        raise ValueError("Window count differs from the static training record")
    edges = np.asarray(fit["bins"]["v_edges_log"])
    counts, _ = np.histogram(np.log(variances[variances > 0]), bins=edges)
    centers = np.exp((edges[:-1] + edges[1:]) / 2)
    empirical = counts / (counts.sum() * np.diff(edges) * centers)
    pd.DataFrame(dict(v=centers, density=empirical, count=counts)).to_csv(
        out / "variance_empirical.csv", index=False, float_format="%.12g")

    curves = {"v": np.exp(np.linspace(edges[0], edges[-1], 250))}
    selected = {row["N"]: row for row in fit["selection_table"]}
    for depth in (1, 8):
        beta = selected[depth]["beta"]
        result = engine.grid(depth, beta)
        curves[f"n{depth}"] = np.interp(np.log(curves["v"]), result[0], result[1]) / curves["v"]
        if depth == 1 and not np.allclose(curves["n1"], invgamma.pdf(curves["v"], beta + 1, scale=beta),
                                          atol=2e-6, rtol=2e-5):
            raise ValueError("Single-level curve disagrees with the analytical density")
    pd.DataFrame(curves).to_csv(out / "variance_theory.csv", index=False, float_format="%.12g")
    x = np.linspace(-4.5, 4.5, 301)
    slices = {"x": x}
    quantiles = np.quantile(variances, (0.1, 0.5, 0.9))
    for percentile, variance in zip((10, 50, 90), quantiles):
        slices[f"q{percentile}"] = norm.pdf(x, scale=np.sqrt(variance))
    slices["mixture"] = np.exp(engine.logpdf(x, result))
    pd.DataFrame(slices).to_csv(out / "variance_slices.csv", index=False, float_format="%.12g")
    print(dict(windows=len(variances), in_histogram=int(counts.sum()), quantiles=quantiles.tolist()))


if __name__ == "__main__":
    main()
