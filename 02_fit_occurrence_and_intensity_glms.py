import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

try:
    from statsmodels.stats.sandwich_covariance import cov_cluster
except Exception:
    cov_cluster = None


def str2bool_text(value: str) -> bool:
    value = value.strip().lower()
    if value in {"yes", "true", "1", "y"}:
        return True
    if value in {"no", "false", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected yes/no, true/false, 1/0.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit two-stage rainfall GLMs and save model comparisons, summaries, and diagnostics."
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
        default=Path("output/step4"),
        help="Directory for Step 4 outputs.",
    )
    parser.add_argument(
        "--break-year",
        type=int,
        default=2000,
        help="Breakpoint year for optional piecewise trend models.",
    )
    parser.add_argument(
        "--tau-mm",
        type=float,
        default=0.1,
        help="Wet-day threshold in mm/day. Used for Stage 2 filtering.",
    )
    parser.add_argument(
        "--use-mjo",
        type=str2bool_text,
        default=True,
        help="Whether to include MJO terms if available (yes/no).",
    )
    parser.add_argument(
        "--use-cluster-robust-se",
        type=str2bool_text,
        default=True,
        help="Whether to compute cluster-robust covariance if the cluster column exists (yes/no).",
    )
    parser.add_argument(
        "--cluster-col",
        type=str,
        default="year",
        help="Column name for cluster-robust covariance.",
    )
    return parser.parse_args()


def prepare_design(
    data: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    cluster_col: str,
    use_cluster_robust_se: bool,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    keep_cols = ["date"]
    extra_cols = [cluster_col] if use_cluster_robust_se else []
    use_cols = keep_cols + [y_col] + x_cols + extra_cols

    d = data.loc[:, use_cols].copy()
    d = d.dropna(axis=0, how="any").copy()

    y = d[y_col].astype(float)
    X = d[x_cols].astype(float)
    X = sm.add_constant(X, has_constant="add")
    return y, X, d


def fit_glm(
    data: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    family,
    cluster_col: str,
    use_cluster_robust_se: bool,
) -> dict:
    y, X, d = prepare_design(data, y_col, x_cols, cluster_col, use_cluster_robust_se)
    model = sm.GLM(y, X, family=family)
    res = model.fit()

    groups = None
    if use_cluster_robust_se:
        groups = d[cluster_col].values

    return {
        "res": res,
        "groups": groups,
        "n": int(res.nobs),
        "y": y,
        "X": X,
        "d": d,
    }


def write_cluster_robust_block(file_obj, res_wrapped, groups, alpha: float = 0.05) -> None:
    if cov_cluster is None:
        file_obj.write("Cluster-robust SE requested, but cov_cluster is unavailable.\n")
        return

    base_res = res_wrapped._results if hasattr(res_wrapped, "_results") else res_wrapped
    cov = cov_cluster(base_res, groups)
    se = np.sqrt(np.diag(cov))

    params = np.asarray(base_res.params)
    names = list(base_res.model.exog_names)

    zvals = params / se
    pvals = 2 * (1 - stats.norm.cdf(np.abs(zvals)))

    zcrit = stats.norm.ppf(1 - alpha / 2)
    ci_lo = params - zcrit * se
    ci_hi = params + zcrit * se

    file_obj.write("Cluster-robust covariance summary\n")
    file_obj.write("===============================================================================\n")
    file_obj.write("                 coef    std err          z      P>|z|      [0.025      0.975]\n")
    file_obj.write("-------------------------------------------------------------------------------\n")
    for nm, b, s, z, p, lo, hi in zip(names, params, se, zvals, pvals, ci_lo, ci_hi):
        file_obj.write(
            f"{nm:>16s} {b:>10.4f} {s:>10.4f} {z:>10.3f} {p:>10.3g} {lo:>11.4f} {hi:>11.4f}\n"
        )
    file_obj.write("===============================================================================\n")


def occurrence_metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    brier = np.mean((p - y_true) ** 2)
    rmse_brier = float(np.sqrt(brier))
    logloss = -float(np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    return {"Brier": float(brier), "RMSE_Brier": rmse_brier, "LogLoss": logloss}


def amounts_metrics(y_true: np.ndarray, mu: np.ndarray) -> dict:
    mu = np.clip(mu, 1e-12, None)
    rmse = float(np.sqrt(np.mean((y_true - mu) ** 2)))
    mae = float(np.mean(np.abs(y_true - mu)))
    bias = float(np.mean(mu - y_true))
    return {"RMSE_mm": rmse, "MAE_mm": mae, "Bias_mm": bias}


def calibration_plot(df_occ: pd.DataFrame, p_col: str, y_col: str, out_png: Path) -> None:
    bins = np.linspace(0, 1, 11)
    dfc = df_occ[[p_col, y_col]].dropna().copy()
    dfc["bin"] = pd.cut(dfc[p_col], bins=bins, include_lowest=True)

    grp = dfc.groupby("bin", observed=True).agg(
        p_mean=(p_col, "mean"),
        y_mean=(y_col, "mean"),
        n=(y_col, "size"),
    ).reset_index()

    fig = plt.figure(figsize=(6.5, 5.0))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.scatter(grp["p_mean"], grp["y_mean"], s=np.clip(grp["n"].values / 50, 10, 200))

    for _, row in grp.iterrows():
        plt.text(row["p_mean"], row["y_mean"], str(int(row["n"])), fontsize=8)

    plt.xlabel("Mean predicted wet probability (bin)")
    plt.ylabel("Observed wet frequency (bin)")
    plt.title("Occurrence model calibration")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def residual_by_year_plot(dates: pd.Series, resid: np.ndarray, out_png: Path, title: str) -> None:
    df_r = pd.DataFrame({"date": dates, "resid": resid})
    df_r["year"] = pd.to_datetime(df_r["date"]).dt.year

    g = df_r.groupby("year").agg(
        mean=("resid", "mean"),
        std=("resid", "std"),
        n=("resid", "size"),
    ).reset_index()

    g["se"] = g["std"] / np.sqrt(np.maximum(g["n"], 1))
    g["lo"] = g["mean"] - 1.96 * g["se"]
    g["hi"] = g["mean"] + 1.96 * g["se"]

    fig = plt.figure(figsize=(9, 4))
    plt.plot(g["year"], g["mean"])
    plt.fill_between(g["year"].values, g["lo"].values, g["hi"].values, alpha=0.25)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Year")
    plt.ylabel("Mean residual")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def anscombe_qq_plot(y: np.ndarray, mu: np.ndarray, out_png: Path) -> None:
    r_a = np.power(np.clip(y / np.clip(mu, 1e-12, None), 1e-12, None), 1.0 / 3.0)
    r_a = (r_a - np.mean(r_a)) / np.std(r_a)

    osm, osr = stats.probplot(r_a, dist="norm", fit=False)

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(osm, osr, s=10)

    lr = stats.linregress(osm, osr)
    slope = lr.slope if hasattr(lr, "slope") else lr[0]
    intercept = lr.intercept if hasattr(lr, "intercept") else lr[1]

    x = np.array([min(osm), max(osm)])
    plt.plot(x, intercept + slope * x)
    plt.xlabel("Normal quantiles")
    plt.ylabel("Anscombe-style residual quantiles")
    plt.title("Gamma amounts: Anscombe-style normal probability plot")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def build_mjo_columns(df: pd.DataFrame, use_mjo: bool) -> list[str]:
    if not use_mjo:
        return []

    mjo_cols = []
    for cand in ["mjo_x", "mjo_y", "rmm1", "rmm2"]:
        if cand in df.columns:
            mjo_cols.append(cand)

    if "mjo_x" in mjo_cols and "mjo_y" in mjo_cols:
        return ["mjo_x", "mjo_y"]

    return mjo_cols


def ensure_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")


def main() -> None:
    args = parse_args()

    input_csv = args.input
    output_dir = args.output_dir
    break_year = args.break_year
    tau_mm = args.tau_mm
    use_mjo = args.use_mjo
    use_cluster_robust_se = args.use_cluster_robust_se
    cluster_col = args.cluster_col

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    required_base = [
        "date", "year", "month", "doy", "rain_mmday", "wet", "rain_wet_mmday",
        "sin_doy1", "cos_doy1", "sin_doy2", "cos_doy2",
    ]
    ensure_required_columns(df, required_base)

    if use_cluster_robust_se and cluster_col not in df.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found in input CSV.")

    if "year_c" not in df.columns:
        df["year_c"] = df["year"] - df["year"].median()

    df["post_break"] = (df["year"] >= break_year).astype(int)
    df["year_postbreak"] = df["post_break"] * (df["year"] - break_year)

    for drv in ["nino34", "dmi"]:
        if drv in df.columns:
            if f"{drv}_x_cos1" not in df.columns:
                df[f"{drv}_x_cos1"] = df[drv] * df["cos_doy1"]
            if f"{drv}_x_sin1" not in df.columns:
                df[f"{drv}_x_sin1"] = df[drv] * df["sin_doy1"]

    wet_lags = sorted([col for col in df.columns if col.startswith("wet_lag")])
    rain_lags = sorted([col for col in df.columns if col.startswith("log1p_rain_lag")])

    if not wet_lags:
        raise ValueError("No wet_lag* columns found in input CSV.")
    if not rain_lags:
        raise ValueError("No log1p_rain_lag* columns found in input CSV.")

    season_cols = ["sin_doy1", "cos_doy1", "sin_doy2", "cos_doy2"]
    occ_persist_cols = sorted(wet_lags) + (["prev5_wet_count"] if "prev5_wet_count" in df.columns else [])
    amt_persist_cols = sorted(rain_lags) + (["prev5_rain_sum"] if "prev5_rain_sum" in df.columns else [])

    driver_cols = [col for col in ["nino34", "dmi"] if col in df.columns]
    driver_int_cols = [
        col for col in ["nino34_x_cos1", "nino34_x_sin1", "dmi_x_cos1", "dmi_x_sin1"]
        if col in df.columns
    ]
    mjo_cols = build_mjo_columns(df, use_mjo)

    occ_scenarios = {
        "O0_baseline": season_cols + occ_persist_cols,
        "O1_linear_trend": season_cols + occ_persist_cols + ["year_c"],
        "O2_piecewise_trend": season_cols + occ_persist_cols + ["year_c", "year_postbreak"],
        "O3_drivers": season_cols + occ_persist_cols + driver_cols + driver_int_cols + mjo_cols,
        "O4_drivers_plus_trend": season_cols + occ_persist_cols + driver_cols + driver_int_cols + mjo_cols + ["year_c"],
    }

    amt_scenarios = {
        "A0_baseline": season_cols + amt_persist_cols,
        "A1_linear_trend": season_cols + amt_persist_cols + ["year_c"],
        "A2_piecewise_trend": season_cols + amt_persist_cols + ["year_c", "year_postbreak"],
        "A3_drivers": season_cols + amt_persist_cols + driver_cols + driver_int_cols + mjo_cols,
        "A4_drivers_plus_trend": season_cols + amt_persist_cols + driver_cols + driver_int_cols + mjo_cols + ["year_c"],
    }

    input_stem = input_csv.stem

    # Stage 1
    occ_rows = []
    occ_fits = {}

    for name, xcols in occ_scenarios.items():
        fit = fit_glm(
            df,
            y_col="wet",
            x_cols=xcols,
            family=sm.families.Binomial(sm.families.links.Logit()),
            cluster_col=cluster_col,
            use_cluster_robust_se=use_cluster_robust_se,
        )
        res = fit["res"]
        p = res.fittedvalues.values
        y = fit["y"].values

        occ_rows.append(
            {
                "model": name,
                "nobs": fit["n"],
                "k_params": int(res.df_model + 1),
                "loglik": float(res.llf),
                "AIC": float(res.aic),
                **occurrence_metrics(y, p),
            }
        )
        occ_fits[name] = fit

    df_occ_tbl = pd.DataFrame(occ_rows).sort_values("AIC").reset_index(drop=True)
    df_occ_tbl.to_csv(output_dir / f"{input_stem}_models_occurrence_comparison.csv", index=False)

    best_occ_name = df_occ_tbl.iloc[0]["model"]
    best_occ = occ_fits[best_occ_name]
    best_occ_res = best_occ["res"]

    with (output_dir / f"{input_stem}_best_occurrence_model_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"INPUT FILE: {input_csv}\n")
        f.write(f"BEST OCCURRENCE MODEL: {best_occ_name}\n\n")
        f.write(best_occ_res.summary().as_text())
        if use_cluster_robust_se and best_occ["groups"] is not None:
            f.write("\n\n--- Cluster-robust SE ---\n")
            write_cluster_robust_block(f, best_occ_res, best_occ["groups"])

    df_occ_diag = best_occ["d"].copy()
    df_occ_diag["p_wet"] = best_occ_res.fittedvalues.values
    calibration_plot(
        df_occ_diag,
        "p_wet",
        "wet",
        output_dir / f"{input_stem}_fig_occ_calibration_bins.png",
    )
    residual_by_year_plot(
        best_occ["d"]["date"],
        best_occ_res.resid_pearson,
        output_dir / f"{input_stem}_fig_occ_residual_by_year.png",
        f"Occurrence: Pearson residual mean by year ({best_occ_name})",
    )

    # Stage 2
    df_wet = df[(df["wet"] == 1) & (df["rain_wet_mmday"] > tau_mm)].copy()

    amt_rows = []
    amt_fits = {}

    for name, xcols in amt_scenarios.items():
        fit = fit_glm(
            df_wet,
            y_col="rain_wet_mmday",
            x_cols=xcols,
            family=sm.families.Gamma(sm.families.links.Log()),
            cluster_col=cluster_col,
            use_cluster_robust_se=use_cluster_robust_se,
        )
        res = fit["res"]
        mu = res.fittedvalues.values
        y = fit["y"].values

        amt_rows.append(
            {
                "model": name,
                "nobs": fit["n"],
                "k_params": int(res.df_model + 1),
                "loglik": float(res.llf),
                "AIC": float(res.aic),
                **amounts_metrics(y, mu),
            }
        )
        amt_fits[name] = fit

    df_amt_tbl = pd.DataFrame(amt_rows).sort_values("AIC").reset_index(drop=True)
    df_amt_tbl.to_csv(output_dir / f"{input_stem}_models_amounts_comparison.csv", index=False)

    best_amt_name = df_amt_tbl.iloc[0]["model"]
    best_amt = amt_fits[best_amt_name]
    best_amt_res = best_amt["res"]

    with (output_dir / f"{input_stem}_best_amounts_model_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"INPUT FILE: {input_csv}\n")
        f.write(f"BEST AMOUNTS MODEL: {best_amt_name}\n\n")
        f.write(best_amt_res.summary().as_text())
        if use_cluster_robust_se and best_amt["groups"] is not None:
            f.write("\n\n--- Cluster-robust SE ---\n")
            write_cluster_robust_block(f, best_amt_res, best_amt["groups"])

    residual_by_year_plot(
        best_amt["d"]["date"],
        best_amt_res.resid_pearson,
        output_dir / f"{input_stem}_fig_amt_residual_by_year.png",
        f"Amounts: Pearson residual mean by year ({best_amt_name})",
    )
    anscombe_qq_plot(
        best_amt["y"].values,
        best_amt_res.fittedvalues.values,
        output_dir / f"{input_stem}_fig_amt_anscombe_qq.png",
    )

    print("\n=== Saved outputs to ===")
    print(output_dir)
    print("\n=== Best models by AIC ===")
    print("Occurrence:", best_occ_name)
    print("Amounts   :", best_amt_name)
    print("\nDone.")


if __name__ == "__main__":
    main()