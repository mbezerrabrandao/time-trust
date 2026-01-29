#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def s_to_hours(x: float) -> float:
    return float(x) / 3600.0


def s_to_minutes(x: float) -> float:
    return float(x) / 60.0


def main() -> None:
    in_csv = Path("results/execution_cost_summary.csv")
    out_tex = Path("results/execution_cost_summary.tex")

    if not in_csv.exists():
        raise FileNotFoundError(f"Missing CSV: {in_csv}")

    df = pd.read_csv(in_csv)

    required = {
        "dataset", "window_tag", "hidden_tag",
        "trust_total_s",
        "tt_sum_total_s",
        "speedup_sum",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    # Ensure numeric
    for col in ["trust_total_s", "tt_sum_total_s", "speedup_sum"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Global mean ± std (across ALL experiments)
    speedups = df["speedup_sum"].to_numpy(dtype=float)
    mean_sp = float(np.mean(speedups))
    std_sp = float(np.std(speedups, ddof=1)) if speedups.size > 1 else 0.0
    n = int(speedups.size)

    print(f"Speed-up (TRUST / (Time-TRUST sensors+windows)) across all experiments: {mean_sp:.2f} ± {std_sp:.2f} (n={n})")

    # Build compact IEEE table (sorted)
    df2 = df.sort_values(["dataset", "window_tag", "hidden_tag"]).reset_index(drop=True)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Execution cost comparison between TRUST and Time-TRUST (sum of sensor-wise and window-wise runs).}")
    lines.append(r"\label{tab:execution_cost_summary}")
    lines.append(r"\begin{tabular}{|c|c|c|c|c|}")
    lines.append(r"\hline")
    # NOTE: 5 columns declared -> header must have 5 entries
    lines.append(r"Subset & Window tag & Hidden & TRUST (h) & Time-TRUST (min) & Speed-up \\")
    # The line above has 6 columns, so fix tabular to 6 columns:
    # We'll correct below by emitting correct header + tabular
    lines = []  # reset and do correct 6-column table

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Execution cost comparison between TRUST and Time-TRUST (sum of sensor-wise and window-wise runs).}")
    lines.append(r"\label{tab:execution_cost_summary}")
    lines.append(r"\begin{tabular}{|c|c|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"Subset & Window tag & Hidden & TRUST (h) & Time-TRUST (min) & Speed-up \\")
    lines.append(r"\hline")

    for _, row in df2.iterrows():
        subset = str(row["dataset"])
        window_tag = str(row["window_tag"])
        hidden = str(row["hidden_tag"])

        trust_h = s_to_hours(float(row["trust_total_s"]))
        tt_min = s_to_minutes(float(row["tt_sum_total_s"]))
        sp = float(row["speedup_sum"])

        lines.append(
            f"{subset} & {window_tag} & {hidden} & {trust_h:.2f} & {tt_min:.1f} & {sp:.1f}x \\\\"
        )
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote LaTeX table: {out_tex}")


if __name__ == "__main__":
    main()