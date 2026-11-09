"""Paired fixed-checkpoint analysis on matched EchoNet-Dynamic test videos."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


BASELINES = {
    "no_cvc": "echomamba_no_cvc_s2",
    "full_cvc": "echomamba_cvc_full_s2",
    "cvc_x6": "echomamba_cvc_x6_s2",
}

METRICS = {
    "case_dice": ("segmentation_phase_centered/segmentation_metrics_per_case.csv", True),
    "case_hd95": ("segmentation_phase_centered/segmentation_metrics_per_case.csv", False),
    "tDice": ("temporal/temporal_metrics_per_case.csv", True),
    "IMV": ("temporal/temporal_metrics_per_case.csv", False),
    "CDMean": ("temporal/temporal_metrics_per_case.csv", False),
    "CDV": ("temporal/temporal_metrics_per_case.csv", False),
}


def bootstrap_ci(values, rng, iterations=10000):
    values = np.asarray(values, dtype=np.float64)
    estimates = np.empty(iterations, dtype=np.float64)
    # Chunking avoids constructing an iterations-by-N matrix at once.
    for start in range(0, iterations, 500):
        size = min(500, iterations - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        estimates[start:start + size] = values[indices].mean(axis=1)
    return np.percentile(estimates, [2.5, 97.5])


def rank_biserial(improvements):
    improvements = np.asarray(improvements, dtype=np.float64)
    improvements = improvements[improvements != 0]
    if not len(improvements):
        return 0.0
    ranks = rankdata(np.abs(improvements))
    positive = ranks[improvements > 0].sum()
    negative = ranks[improvements < 0].sum()
    return float((positive - negative) / (positive + negative))


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def load_metric(root, relative_path, metric):
    frame = pd.read_csv(root / relative_path)
    if "filename" not in frame or metric not in frame:
        raise RuntimeError(f"Required columns missing in {root / relative_path}")
    return frame[["filename", metric]].rename(columns={metric: "value"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root", default="experiments/evaluations/test")
    parser.add_argument("--output", default="experiments/statistics/v2_paired_test.csv")
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    root = Path(args.test_root)
    proposed_root = root / "dual_phase_uniform_stage2_probe_seed42"
    rows = []
    rng = np.random.default_rng(args.seed)

    for baseline_label, baseline_folder in BASELINES.items():
        baseline_root = root / "reference" / baseline_folder
        for metric, (relative_path, higher_is_better) in METRICS.items():
            proposed = load_metric(proposed_root, relative_path, metric)
            baseline = load_metric(baseline_root, relative_path, metric)
            paired = proposed.merge(baseline, on="filename", suffixes=("_new", "_old"))
            paired = paired.replace([np.inf, -np.inf], np.nan).dropna()
            raw_difference = paired["value_new"].to_numpy() - paired["value_old"].to_numpy()
            improvement = raw_difference if higher_is_better else -raw_difference
            ci_low, ci_high = bootstrap_ci(improvement, rng, args.bootstrap_iterations)
            try:
                test = wilcoxon(improvement, alternative="two-sided", zero_method="wilcox",
                                method="auto")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            except ValueError:
                statistic, p_value = 0.0, 1.0
            rows.append({
                "comparison": f"v2_vs_{baseline_label}",
                "metric": metric,
                "direction": "higher" if higher_is_better else "lower",
                "n_pairs": len(paired),
                "v2_mean": paired["value_new"].mean(),
                "baseline_mean": paired["value_old"].mean(),
                "raw_difference_v2_minus_baseline": raw_difference.mean(),
                "oriented_improvement": improvement.mean(),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "wilcoxon_statistic": statistic,
                "p_raw": p_value,
                "rank_biserial": rank_biserial(improvement),
            })

    results = pd.DataFrame(rows)
    results["p_holm_18"] = holm_adjust(results["p_raw"].to_numpy())
    results["significant_holm_0p05"] = results["p_holm_18"] < 0.05
    results["ci_supports_improvement"] = results["bootstrap_ci_low"] > 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    display = results[["comparison", "metric", "n_pairs", "oriented_improvement",
                       "bootstrap_ci_low", "bootstrap_ci_high", "p_holm_18",
                       "rank_biserial", "ci_supports_improvement"]]
    print(display.to_string(index=False))
    print(f"\nSaved: {output}")
    print("Positive oriented_improvement always means V2 is better.")


if __name__ == "__main__":
    main()
