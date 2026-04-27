import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SEASONS = {
    "Annual": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "MAM": [3, 4, 5],
    "JJAS": [6, 7, 8, 9],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize early-vs-late rainfall changes with bootstrap confidence intervals."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to Step 3 GLM input CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/step4_earlylate"),
        help="Directory for Step 4E outputs.",
    )
    parser.add_argument(
        "--tau-mm",
        type=float,
        default=0.1,
        help="Wet-day threshold in mm/day.",
    )
    parser.add_argument(
        "--early-start",
        type=int,
        required=True,
        help="Start year for the early period.",
    )
    parser.add_argument(
        "--early-end",
        type=int,
        required=True,
        help="End year for the early period.",
    )
    parser.add_argument(
        "--late-start",
        type=int,
        required=True,
        help="Start year for the late period.",
    )
    parser.add_argument(
        "--late-end",
        type=int,
        required=True,
        help="End year for the late period.",
    )
    parser.add_argument(
        "--bootstrap-size",
        type=int,
        default=500,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260423,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--region-label",
        type=str,
        default="Study Region",
        help="Region label shown in the figure and text summary.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--season",
        action="append",
        default=[],
        help='Optional custom season in the form NAME:month1,month2,... Example: "OND:10,11,12". '
             "You can provide this argument multiple times. If omitted, default seasons are used.",
    )
    return parser.parse_args()


def parse_seasons(custom_season_args):
    if not custom_season_args:
        return DEFAULT_SEASONS.copy()

    seasons = {}
    for item in custom_season_args:
        if ":" not in item:
            raise ValueError(f"Invalid --season format: {item}")
        name, months_text = item.split(":", 1)
        months = [int(x.strip()) for x in months_text.split(",") if x.strip()]
        if not months:
            raise ValueError(f"No months found in --season argument: {item}")
        seasons[name.strip()] = months
    return seasons


def ensure_required_columns(df, required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def label_period(year_value, early_start, early_end, late_start, late_end):
    if early_start <= year_value <= early_end:
        return "Early"
    if late_start <= year_value <= late_end:
        return "Late"
    return "Other"


def pct(x):
    return 100.0 * x


def summarize_daily_periods(data, months, season_name):
    d = data[data["month"].isin(months)].copy()

    rows = []
    for period, group in d.groupby("period"):
        wet_frac = float(group["wet"].mean())
        mean_all = float(group["rain_mmday"].mean())
        mean_wet = float(group.loc[group["wet"] == 1, "rain_mmday"].mean()) if (group["wet"] == 1).any() else np.nan

        rows.append(
            {
                "season": season_name,
                "period": period,
                "n_days": int(len(group)),
                "wet_day_fraction": wet_frac,
                "wet_day_percent": pct(wet_frac),
                "mean_rain_all_days_mmday": mean_all,
                "mean_rain_wet_days_mmday": mean_wet,
            }
        )
    return pd.DataFrame(rows)


def seasonal_year_aggregate(data, season_name, months):
    d = data[data["month"].isin(months)].copy()
    d["season_year"] = d["year"]

    g = d.groupby(["period", "season_year"], as_index=False).agg(
        seasonal_total_mm=("rain_mmday", "sum"),
        wet_day_count=("wet", "sum"),
        wet_day_fraction=("wet", "mean"),
        mean_wet_day_intensity_mmday=("rain_wet_mmday", "mean"),
        n_days=("wet", "size"),
    )
    g["season"] = season_name
    return g


def summarize_seasonal_periods(data, season_name, months):
    ys = seasonal_year_aggregate(data, season_name, months)

    out = ys.groupby(["season", "period"], as_index=False).agg(
        n_years=("season_year", "size"),
        mean_seasonal_total_mm=("seasonal_total_mm", "mean"),
        sd_seasonal_total_mm=("seasonal_total_mm", "std"),
        mean_wet_day_count=("wet_day_count", "mean"),
        mean_wet_day_fraction=("wet_day_fraction", "mean"),
        mean_wet_day_percent=("wet_day_fraction", lambda x: 100.0 * np.mean(x)),
        mean_wet_day_intensity_mmday=("mean_wet_day_intensity_mmday", "mean"),
    )
    return out


def bootstrap_year_blocks_daily(data, months, seed, n_boot, metric_fn):
    rng = np.random.default_rng(seed)
    d = data[data["month"].isin(months)].copy()

    d_early = d[d["period"] == "Early"].copy()
    d_late = d[d["period"] == "Late"].copy()

    years_early = np.sort(d_early["year"].unique())
    years_late = np.sort(d_late["year"].unique())

    blocks_early = {yy: d_early[d_early["year"] == yy] for yy in years_early}
    blocks_late = {yy: d_late[d_late["year"] == yy] for yy in years_late}

    draws = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        samp_early = rng.choice(years_early, size=len(years_early), replace=True)
        samp_late = rng.choice(years_late, size=len(years_late), replace=True)

        be = pd.concat([blocks_early[yy] for yy in samp_early], ignore_index=True)
        bl = pd.concat([blocks_late[yy] for yy in samp_late], ignore_index=True)

        draws[b] = metric_fn(bl) - metric_fn(be)

    return draws


def bootstrap_year_blocks_seasonal(data, months, seed, n_boot, metric_col):
    ys = seasonal_year_aggregate(data, season_name="tmp", months=months)
    ys_early = ys[ys["period"] == "Early"].copy()
    ys_late = ys[ys["period"] == "Late"].copy()

    rng = np.random.default_rng(seed)
    vals_early = ys_early[metric_col].dropna().values
    vals_late = ys_late[metric_col].dropna().values

    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        be = rng.choice(vals_early, size=len(vals_early), replace=True)
        bl = rng.choice(vals_late, size=len(vals_late), replace=True)
        draws[b] = np.mean(bl) - np.mean(be)

    return draws


def summarize_draws(draws):
    return {
        "boot_mean_diff": float(np.mean(draws)),
        "ci2.5": float(np.percentile(draws, 2.5)),
        "ci50": float(np.percentile(draws, 50.0)),
        "ci97.5": float(np.percentile(draws, 97.5)),
        "p_boot_2s": float(2 * min((draws <= 0).mean(), (draws >= 0).mean())),
    }


def safe_percent_change(late_val, early_val):
    if np.isfinite(early_val) and early_val != 0:
        return 100.0 * (late_val - early_val) / early_val
    return np.nan


def get_metric(boot_tbl, season, metric):
    row = boot_tbl[(boot_tbl["season"] == season) & (boot_tbl["metric"] == metric)]
    if row.empty:
        return None
    return row.iloc[0]


def main():
    args = parse_args()

    input_csv = args.input
    output_dir = args.output_dir
    tau_mm = args.tau_mm
    early_start = args.early_start
    early_end = args.early_end
    late_start = args.late_start
    late_end = args.late_end
    n_boot = args.bootstrap_size
    seed = args.seed
    region_label = args.region_label
    dpi = args.dpi
    seasons = parse_seasons(args.season)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    required = ["date", "year", "month", "rain_mmday", "wet"]
    ensure_required_columns(df, required)

    wet_recalc = (df["rain_mmday"] > tau_mm).astype(int)
    if not np.array_equal(df["wet"].fillna(-999).astype(int).values, wet_recalc.values):
        print("[WARN] 'wet' column does not exactly match tau threshold. Recomputing wet from rain_mmday.")
        df["wet"] = wet_recalc

    if "rain_wet_mmday" not in df.columns:
        df["rain_wet_mmday"] = df["rain_mmday"].where(df["wet"] == 1, np.nan)

    df = df[(df["year"] >= early_start) & (df["year"] <= late_end)].copy()
    df["period"] = df["year"].map(lambda y: label_period(y, early_start, early_end, late_start, late_end))
    df = df[df["period"].isin(["Early", "Late"])].copy()

    input_stem = input_csv.stem

    # Daily summaries
    daily_tables = []
    for season_name, months in seasons.items():
        daily_tables.append(summarize_daily_periods(df, months, season_name))
    daily_summary = pd.concat(daily_tables, ignore_index=True)
    daily_summary.to_csv(output_dir / f"{input_stem}_early_late_daily_summary.csv", index=False)

    # Seasonal summaries
    seasonal_tables = []
    for season_name, months in seasons.items():
        seasonal_tables.append(summarize_seasonal_periods(df, season_name, months))
    seasonal_summary = pd.concat(seasonal_tables, ignore_index=True)
    seasonal_summary.to_csv(output_dir / f"{input_stem}_early_late_seasonal_summary.csv", index=False)

    # Bootstrap summary table
    boot_rows = []

    for season_name, months in seasons.items():
        d_sub = daily_summary[daily_summary["season"] == season_name].copy()
        if {"Early", "Late"} <= set(d_sub["period"]):
            e_row = d_sub[d_sub["period"] == "Early"].iloc[0]
            l_row = d_sub[d_sub["period"] == "Late"].iloc[0]

            def metric_wetfrac(g):
                return float(g["wet"].mean())

            draws = bootstrap_year_blocks_daily(df, months, seed + abs(hash(season_name)) % 1000, n_boot, metric_wetfrac)
            out = summarize_draws(draws)
            boot_rows.append(
                {
                    "season": season_name,
                    "metric": "wet_day_fraction",
                    "early": float(e_row["wet_day_fraction"]),
                    "late": float(l_row["wet_day_fraction"]),
                    "late_minus_early": float(l_row["wet_day_fraction"] - e_row["wet_day_fraction"]),
                    "percent_change_relative_to_early": safe_percent_change(
                        float(l_row["wet_day_fraction"]),
                        float(e_row["wet_day_fraction"]),
                    ),
                    **out,
                }
            )

            def metric_allrain(g):
                return float(g["rain_mmday"].mean())

            draws = bootstrap_year_blocks_daily(df, months, seed + abs(hash(season_name + "_all")) % 1000, n_boot, metric_allrain)
            out = summarize_draws(draws)
            boot_rows.append(
                {
                    "season": season_name,
                    "metric": "mean_rain_all_days_mmday",
                    "early": float(e_row["mean_rain_all_days_mmday"]),
                    "late": float(l_row["mean_rain_all_days_mmday"]),
                    "late_minus_early": float(l_row["mean_rain_all_days_mmday"] - e_row["mean_rain_all_days_mmday"]),
                    "percent_change_relative_to_early": safe_percent_change(
                        float(l_row["mean_rain_all_days_mmday"]),
                        float(e_row["mean_rain_all_days_mmday"]),
                    ),
                    **out,
                }
            )

        s_sub = seasonal_summary[seasonal_summary["season"] == season_name].copy()
        if {"Early", "Late"} <= set(s_sub["period"]):
            e_row = s_sub[s_sub["period"] == "Early"].iloc[0]
            l_row = s_sub[s_sub["period"] == "Late"].iloc[0]

            metric_map = {
                "mean_seasonal_total_mm": "seasonal_total_mm",
                "mean_wet_day_count": "wet_day_count",
                "mean_wet_day_intensity_mmday": "mean_wet_day_intensity_mmday",
            }

            for metric_name, source_col in metric_map.items():
                draws = bootstrap_year_blocks_seasonal(
                    df,
                    months,
                    seed + abs(hash(season_name + metric_name)) % 1000,
                    n_boot,
                    source_col,
                )

                out = summarize_draws(draws)
                early_val = float(e_row[metric_name])
                late_val = float(l_row[metric_name])

                boot_rows.append(
                    {
                        "season": season_name,
                        "metric": metric_name,
                        "early": early_val,
                        "late": late_val,
                        "late_minus_early": late_val - early_val,
                        "percent_change_relative_to_early": safe_percent_change(late_val, early_val),
                        **out,
                    }
                )

    boot_tbl = pd.DataFrame(boot_rows)
    boot_tbl.to_csv(output_dir / f"{input_stem}_bootstrap_early_late_summary.csv", index=False)

    # Plot: choose sensible defaults if available
    metrics_to_plot = [
        ("Annual", "wet_day_fraction", "Wet-day fraction (-)"),
        ("Annual", "mean_seasonal_total_mm", "Annual total rainfall (mm)"),
        ("JJAS", "mean_seasonal_total_mm", "JJAS total rainfall (mm)"),
        ("MAM", "mean_seasonal_total_mm", "MAM total rainfall (mm)"),
    ]

    fig = plt.figure(figsize=(10.5, 7.5))
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.35)

    for i, (season_name, metric_name, ylabel) in enumerate(metrics_to_plot):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        row = boot_tbl[(boot_tbl["season"] == season_name) & (boot_tbl["metric"] == metric_name)]

        if row.empty:
            ax.axis("off")
            continue

        r = row.iloc[0]
        vals = [r["early"], r["late"]]
        labels = [f"Early\n{early_start}-{early_end}", f"Late\n{late_start}-{late_end}"]

        ax.bar([0, 1], vals, color=["0.75", "0.35"], edgecolor="black", width=0.65)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{season_name}: {metric_name}")

        ymax = max(vals) if len(vals) > 0 else 0
        ax.text(
            0.5,
            ymax * 1.03 if ymax > 0 else 0.05,
            f"Δ = {r['late_minus_early']:.3f}\n95% CI [{r['ci2.5']:.3f}, {r['ci97.5']:.3f}]",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(
        f"Early-vs-late contrasts in wet-day occurrence and rainfall\n{region_label}",
        y=0.98,
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output_dir / f"{input_stem}_fig_early_late_contrast.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Manuscript-ready text
    ann_wet = get_metric(boot_tbl, "Annual", "wet_day_fraction")
    ann_tot = get_metric(boot_tbl, "Annual", "mean_seasonal_total_mm")
    jjas_tot = get_metric(boot_tbl, "JJAS", "mean_seasonal_total_mm")
    mam_tot = get_metric(boot_tbl, "MAM", "mean_seasonal_total_mm")

    lines = []
    lines.append("Early-vs-late period summary")
    lines.append("")
    lines.append(f"Input file: {input_csv}")
    lines.append(f"Region: {region_label}")
    lines.append(f"Early period: {early_start}-{early_end}")
    lines.append(f"Late period: {late_start}-{late_end}")
    lines.append("")

    if ann_wet is not None:
        lines.append(
            "Wet-day occurrence changed from "
            f"{100 * ann_wet['early']:.2f}% in the early period to "
            f"{100 * ann_wet['late']:.2f}% in the late period "
            f"(difference = {100 * ann_wet['late_minus_early']:.2f} percentage points; "
            f"95% bootstrap CI: {100 * ann_wet['ci2.5']:.2f} to {100 * ann_wet['ci97.5']:.2f} percentage points)."
        )

    if ann_tot is not None:
        lines.append(
            "Mean annual rainfall changed from "
            f"{ann_tot['early']:.1f} mm to {ann_tot['late']:.1f} mm "
            f"(difference = {ann_tot['late_minus_early']:.1f} mm; "
            f"95% bootstrap CI: {ann_tot['ci2.5']:.1f} to {ann_tot['ci97.5']:.1f} mm)."
        )

    if mam_tot is not None:
        lines.append(
            "For MAM, mean seasonal total rainfall changed from "
            f"{mam_tot['early']:.1f} mm to {mam_tot['late']:.1f} mm "
            f"(difference = {mam_tot['late_minus_early']:.1f} mm; "
            f"95% bootstrap CI: {mam_tot['ci2.5']:.1f} to {mam_tot['ci97.5']:.1f} mm)."
        )

    if jjas_tot is not None:
        lines.append(
            "For JJAS, mean seasonal total rainfall changed from "
            f"{jjas_tot['early']:.1f} mm to {jjas_tot['late']:.1f} mm "
            f"(difference = {jjas_tot['late_minus_early']:.1f} mm; "
            f"95% bootstrap CI: {jjas_tot['ci2.5']:.1f} to {jjas_tot['ci97.5']:.1f} mm)."
        )

    with (output_dir / f"{input_stem}_early_late_summary_for_manuscript.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("[OK] Saved:")
    for p in sorted(output_dir.glob(f"{input_stem}_*")):
        print(" -", p)


if __name__ == "__main__":
    main()