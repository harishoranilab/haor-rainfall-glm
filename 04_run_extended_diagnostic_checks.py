import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


def str2bool_text(value):
    value = value.strip().lower()
    if value in {"yes", "true", "1", "y"}:
        return True
    if value in {"no", "false", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected yes/no, true/false, 1/0.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run extended diagnostics for final two-stage rainfall GLMs."
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
        default=Path("output/step4_diagnostics"),
        help="Directory for Step 4D outputs.",
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
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=11,
        help="Base matplotlib font size.",
    )
    parser.add_argument(
        "--tail-qs",
        type=float,
        nargs="+",
        default=[0.90, 0.95, 0.99],
        help="Upper-tail quantiles to check for the Gamma model.",
    )
    return parser.parse_args()


def ensure_required_columns(df, required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")


def build_mjo_columns(df, use_mjo):
    if not use_mjo:
        return []

    mjo_cols = []
    for cand in ["mjo_x", "mjo_y", "rmm1", "rmm2"]:
        if cand in df.columns:
            mjo_cols.append(cand)

    if "mjo_x" in mjo_cols and "mjo_y" in mjo_cols:
        return ["mjo_x", "mjo_y"]

    return mjo_cols


def prep_design(data, y_col, x_cols):
    d = data.dropna(subset=[y_col] + x_cols).copy()
    y = d[y_col].astype(float).values
    X = sm.add_constant(d[x_cols].astype(float), has_constant="add")
    return d, y, X


def fit_glm(data, y_col, x_cols, family):
    d, y, X = prep_design(data, y_col, x_cols)
    res = sm.GLM(y, X, family=family).fit()
    return d, y, X, res


def savefig(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def residual_by_year_plot(dates, resid, out_png, title, ylabel, dpi):
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

    fig = plt.figure(figsize=(8.8, 4.2))
    ax = plt.gca()
    ax.plot(g["year"], g["mean"], linewidth=1.8, color="black")
    ax.fill_between(g["year"].values, g["lo"].values, g["hi"].values, color="0.85", alpha=1.0)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    savefig(fig, out_png, dpi)


def hist_plot(x, out_png, title, xlabel, dpi):
    fig = plt.figure(figsize=(6.4, 4.4))
    ax = plt.gca()
    ax.hist(x, bins=35, color="0.75", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    savefig(fig, out_png, dpi)


def scatter_resid_vs_fitted(fitted, resid, out_png, title, xlabel, dpi):
    fig = plt.figure(figsize=(6.5, 5.0))
    ax = plt.gca()
    ax.scatter(fitted, resid, s=8, alpha=0.5, color="black")
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Residual")
    ax.set_title(title)
    savefig(fig, out_png, dpi)


def binned_residual_plot(fitted, resid, out_png, title, dpi, nbins=20):
    d = pd.DataFrame({"fitted": fitted, "resid": resid}).dropna()
    d["bin"] = pd.qcut(d["fitted"], q=min(nbins, d["fitted"].nunique()), duplicates="drop")

    g = d.groupby("bin", observed=True).agg(
        fitted_mean=("fitted", "mean"),
        resid_mean=("resid", "mean"),
        resid_sd=("resid", "std"),
        n=("resid", "size"),
    ).reset_index()

    g["se"] = g["resid_sd"] / np.sqrt(np.maximum(g["n"], 1))
    g["lo"] = g["resid_mean"] - 1.96 * g["se"]
    g["hi"] = g["resid_mean"] + 1.96 * g["se"]

    fig = plt.figure(figsize=(6.5, 5.0))
    ax = plt.gca()
    ax.plot(g["fitted_mean"], g["resid_mean"], marker="o", color="black")
    ax.fill_between(g["fitted_mean"].values, g["lo"].values, g["hi"].values, color="0.85")
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
    ax.set_xlabel("Mean fitted value (bin)")
    ax.set_ylabel("Mean residual")
    ax.set_title(title)
    savefig(fig, out_png, dpi)


def calibration_plot(df_occ, p_col, y_col, out_png, dpi):
    bins = np.linspace(0, 1, 11)
    dfc = df_occ[[p_col, y_col]].dropna().copy()
    dfc["bin"] = pd.cut(dfc[p_col], bins=bins, include_lowest=True)

    grp = dfc.groupby("bin", observed=True).agg(
        p_mean=(p_col, "mean"),
        y_mean=(y_col, "mean"),
        n=(y_col, "size"),
    ).reset_index()

    fig = plt.figure(figsize=(6.4, 5.0))
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    ax.scatter(grp["p_mean"], grp["y_mean"], s=np.clip(grp["n"].values / 50, 12, 180), color="black")

    for _, row in grp.iterrows():
        ax.text(row["p_mean"], row["y_mean"], str(int(row["n"])), fontsize=8, ha="left", va="bottom")

    ax.set_xlabel("Mean predicted wet probability (bin)")
    ax.set_ylabel("Observed wet frequency (bin)")
    ax.set_title("Occurrence model calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    savefig(fig, out_png, dpi)


def anscombe_qq_plot(y, mu, out_png, dpi, title="Gamma amounts: Anscombe-style normal probability plot"):
    r_a = np.power(np.clip(y / np.clip(mu, 1e-12, None), 1e-12, None), 1.0 / 3.0)
    r_a = (r_a - np.mean(r_a)) / np.std(r_a)

    osm, osr = stats.probplot(r_a, dist="norm", fit=False)
    lr = stats.linregress(osm, osr)

    fig = plt.figure(figsize=(5.8, 5.8))
    ax = plt.gca()
    ax.scatter(osm, osr, s=10, color="black")
    x = np.array([min(osm), max(osm)])
    ax.plot(x, lr.intercept + lr.slope * x, color="black", linewidth=1.2)
    ax.set_xlabel("Normal quantiles")
    ax.set_ylabel("Anscombe-style residual quantiles")
    ax.set_title(title)
    savefig(fig, out_png, dpi)


def gamma_deviance_residuals(y, mu):
    y = np.clip(y, 1e-12, None)
    mu = np.clip(mu, 1e-12, None)
    dev = 2.0 * (((y - mu) / mu) - np.log(y / mu))
    dev = np.clip(dev, 0.0, None)
    return np.sign(y - mu) * np.sqrt(dev)


def randomized_quantile_residuals_gamma(y, mu, scale_param):
    y = np.clip(y, 1e-12, None)
    mu = np.clip(mu, 1e-12, None)
    shape = 1.0 / scale_param
    scale_i = mu / shape
    pit = stats.gamma.cdf(y, a=shape, scale=scale_i)
    pit = np.clip(pit, 1e-10, 1 - 1e-10)
    z = stats.norm.ppf(pit)
    return z, pit


def quantile_qq_plot(z, out_png, title, dpi):
    osm, osr = stats.probplot(z, dist="norm", fit=False)
    lr = stats.linregress(osm, osr)

    fig = plt.figure(figsize=(5.8, 5.8))
    ax = plt.gca()
    ax.scatter(osm, osr, s=10, color="black")
    x = np.array([min(osm), max(osm)])
    ax.plot(x, lr.intercept + lr.slope * x, color="black", linewidth=1.2)
    ax.set_xlabel("Normal quantiles")
    ax.set_ylabel("Quantile residuals")
    ax.set_title(title)
    savefig(fig, out_png, dpi)


def pit_histogram(pit, out_png, dpi):
    fig = plt.figure(figsize=(6.0, 4.2))
    ax = plt.gca()
    ax.hist(pit, bins=20, range=(0, 1), color="0.75", edgecolor="black")
    ax.axhline(len(pit) / 20.0, linestyle="--", color="black", linewidth=1.0)
    ax.set_xlabel("PIT")
    ax.set_ylabel("Count")
    ax.set_title("Gamma PIT histogram")
    savefig(fig, out_png, dpi)


def gamma_upper_tail_checks(y, mu, scale_param, tail_qs, out_csv, out_png, dpi):
    shape = 1.0 / scale_param
    rows = []

    for q in tail_qs:
        scale_i = mu / shape
        q_i = stats.gamma.ppf(q, a=shape, scale=scale_i)
        obs_rate = float(np.mean(y > q_i))
        exp_rate = float(1.0 - q)
        rows.append(
            {
                "quantile_q": q,
                "expected_exceedance_rate": exp_rate,
                "observed_exceedance_rate": obs_rate,
                "difference_obs_minus_exp": obs_rate - exp_rate,
            }
        )

    tbl = pd.DataFrame(rows)
    tbl.to_csv(out_csv, index=False)

    fig = plt.figure(figsize=(5.8, 4.5))
    ax = plt.gca()
    ax.plot(
        tbl["expected_exceedance_rate"],
        tbl["observed_exceedance_rate"],
        marker="o",
        color="black",
        linewidth=1.5,
    )

    upper_lim = max(tbl["expected_exceedance_rate"].max(), tbl["observed_exceedance_rate"].max()) * 1.15
    lim = [0, upper_lim]
    ax.plot(lim, lim, linestyle="--", color="black", linewidth=1.0)

    for _, row in tbl.iterrows():
        ax.text(
            row["expected_exceedance_rate"],
            row["observed_exceedance_rate"],
            f"q={row['quantile_q']:.2f}",
            fontsize=9,
            ha="left",
            va="bottom",
        )

    ax.set_xlabel("Expected exceedance rate")
    ax.set_ylabel("Observed exceedance rate")
    ax.set_title("Gamma upper-tail exceedance check")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    savefig(fig, out_png, dpi)


def teleconnection_columns(df0, include_interactions=True, include_year=True):
    cols = []
    for c in ["nino34", "dmi", "mjo_x", "mjo_y", "rmm1", "rmm2"]:
        if c in df0.columns:
            cols.append(c)

    if include_year and "year_c" in df0.columns:
        cols.append("year_c")

    if include_interactions:
        for c in ["nino34_x_cos1", "nino34_x_sin1", "dmi_x_cos1", "dmi_x_sin1"]:
            if c in df0.columns:
                cols.append(c)

    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def correlation_and_vif(df0, cols, out_csv_corr, out_png_corr, out_csv_vif, title, dpi):
    d = df0[cols].dropna().copy()
    corr = d.corr(numeric_only=True)
    corr.to_csv(out_csv_corr)

    fig = plt.figure(figsize=(0.8 * len(cols) + 2.5, 0.8 * len(cols) + 2.0))
    ax = plt.gca()
    im = ax.imshow(corr.values, cmap="Greys", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    ax.set_title(title)

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")
    savefig(fig, out_png_corr, dpi)

    X = sm.add_constant(d.astype(float), has_constant="add")
    vif_rows = []
    for i, name in enumerate(X.columns):
        if name == "const":
            continue
        vif_rows.append(
            {
                "term": name,
                "VIF": float(variance_inflation_factor(X.values, i)),
            }
        )

    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False).reset_index(drop=True)
    vif_df.to_csv(out_csv_vif, index=False)
    return corr, vif_df


def main():
    args = parse_args()

    input_csv = args.input
    output_dir = args.output_dir
    tau_mm = args.tau_mm
    use_mjo = args.use_mjo
    dpi = args.dpi
    font_size = args.font_size
    tail_qs = args.tail_qs

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
        }
    )

    df = pd.read_csv(input_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    required = [
        "date",
        "year",
        "wet",
        "rain_mmday",
        "rain_wet_mmday",
        "sin_doy1",
        "cos_doy1",
        "sin_doy2",
        "cos_doy2",
    ]
    ensure_required_columns(df, required)

    if "year_c" not in df.columns:
        df["year_c"] = df["year"] - df["year"].median()

    for drv in ["nino34", "dmi"]:
        if drv in df.columns:
            if f"{drv}_x_cos1" not in df.columns:
                df[f"{drv}_x_cos1"] = df[drv] * df["cos_doy1"]
            if f"{drv}_x_sin1" not in df.columns:
                df[f"{drv}_x_sin1"] = df[drv] * df["sin_doy1"]

    mjo_cols = build_mjo_columns(df, use_mjo)

    wet_lags = sorted([c for c in df.columns if c.startswith("wet_lag")])
    rain_lags = sorted([c for c in df.columns if c.startswith("log1p_rain_lag")])

    if not wet_lags:
        raise ValueError("No wet_lag* columns found in input CSV.")
    if not rain_lags:
        raise ValueError("No log1p_rain_lag* columns found in input CSV.")

    occ_extra = ["prev5_wet_count"] if "prev5_wet_count" in df.columns else []
    amt_extra = ["prev5_rain_sum"] if "prev5_rain_sum" in df.columns else []

    x_occ = (
        ["sin_doy1", "cos_doy1", "sin_doy2", "cos_doy2"]
        + wet_lags
        + occ_extra
        + [c for c in ["nino34", "dmi"] if c in df.columns]
        + [c for c in ["nino34_x_cos1", "nino34_x_sin1", "dmi_x_cos1", "dmi_x_sin1"] if c in df.columns]
        + mjo_cols
        + ["year_c"]
    )

    x_amt = (
        ["sin_doy1", "cos_doy1", "sin_doy2", "cos_doy2"]
        + rain_lags
        + amt_extra
        + [c for c in ["nino34", "dmi"] if c in df.columns]
        + [c for c in ["nino34_x_cos1", "nino34_x_sin1", "dmi_x_cos1", "dmi_x_sin1"] if c in df.columns]
        + mjo_cols
    )

    input_stem = input_csv.stem

    d_occ, y_occ, X_occ, res_occ = fit_glm(
        df,
        "wet",
        x_occ,
        sm.families.Binomial(sm.families.links.Logit()),
    )

    df_wet = df[(df["wet"] == 1) & (df["rain_wet_mmday"] > tau_mm)].copy()
    d_amt, y_amt, X_amt, res_amt = fit_glm(
        df_wet,
        "rain_wet_mmday",
        x_amt,
        sm.families.Gamma(sm.families.links.Log()),
    )

    # Occurrence diagnostics
    fitted_occ = np.asarray(res_occ.fittedvalues)
    pearson_occ = np.asarray(res_occ.resid_pearson)

    scatter_resid_vs_fitted(
        fitted_occ,
        pearson_occ,
        output_dir / f"{input_stem}_occ_resid_vs_fitted.png",
        "Occurrence: Pearson residual vs fitted probability",
        "Fitted wet-day probability",
        dpi,
    )

    hist_plot(
        pearson_occ,
        output_dir / f"{input_stem}_occ_resid_hist.png",
        "Occurrence: Pearson residual distribution",
        "Pearson residual",
        dpi,
    )

    binned_residual_plot(
        fitted_occ,
        pearson_occ,
        output_dir / f"{input_stem}_occ_binned_residuals.png",
        "Occurrence: binned Pearson residuals vs fitted probability",
        dpi,
    )

    residual_by_year_plot(
        d_occ["date"],
        pearson_occ,
        output_dir / f"{input_stem}_occ_resid_by_year.png",
        "Occurrence: Pearson residual mean by year",
        "Mean residual",
        dpi,
    )

    df_occ_diag = d_occ.copy()
    df_occ_diag["p_wet"] = fitted_occ
    calibration_plot(
        df_occ_diag,
        "p_wet",
        "wet",
        output_dir / f"{input_stem}_occ_calibration_bins.png",
        dpi,
    )

    # Amount diagnostics
    fitted_amt = np.asarray(res_amt.fittedvalues)
    pearson_amt = np.asarray(res_amt.resid_pearson)
    deviance_amt = gamma_deviance_residuals(y_amt, fitted_amt)

    scatter_resid_vs_fitted(
        fitted_amt,
        pearson_amt,
        output_dir / f"{input_stem}_amt_pearson_vs_fitted.png",
        "Amounts: Pearson residual vs fitted mean",
        "Fitted conditional mean rainfall (mm)",
        dpi,
    )

    scatter_resid_vs_fitted(
        fitted_amt,
        deviance_amt,
        output_dir / f"{input_stem}_amt_deviance_vs_fitted.png",
        "Amounts: deviance residual vs fitted mean",
        "Fitted conditional mean rainfall (mm)",
        dpi,
    )

    hist_plot(
        pearson_amt,
        output_dir / f"{input_stem}_amt_resid_hist.png",
        "Amounts: Pearson residual distribution",
        "Pearson residual",
        dpi,
    )

    residual_by_year_plot(
        d_amt["date"],
        pearson_amt,
        output_dir / f"{input_stem}_amt_resid_by_year.png",
        "Amounts: Pearson residual mean by year",
        "Mean residual",
        dpi,
    )

    anscombe_qq_plot(
        y_amt,
        fitted_amt,
        output_dir / f"{input_stem}_amt_anscombe_qq.png",
        dpi,
    )

    # Gamma upper-tail checks
    scale_amt = float(res_amt.scale)
    rqres, pit = randomized_quantile_residuals_gamma(y_amt, fitted_amt, scale_amt)

    pit_histogram(
        pit,
        output_dir / f"{input_stem}_gamma_pit_hist.png",
        dpi,
    )

    quantile_qq_plot(
        rqres,
        output_dir / f"{input_stem}_gamma_quantile_residual_qq.png",
        "Gamma model: randomized quantile residual QQ plot",
        dpi,
    )

    gamma_upper_tail_checks(
        y_amt,
        fitted_amt,
        scale_amt,
        tail_qs,
        output_dir / f"{input_stem}_gamma_upper_tail_table.csv",
        output_dir / f"{input_stem}_gamma_upper_tail_exceedance.png",
        dpi,
    )

    # Teleconnection diagnostics
    tele_cols_occ = teleconnection_columns(d_occ, include_interactions=True, include_year=True)
    tele_cols_amt = teleconnection_columns(d_amt, include_interactions=True, include_year=False)

    if len(tele_cols_occ) >= 2:
        correlation_and_vif(
            d_occ,
            tele_cols_occ,
            output_dir / f"{input_stem}_teleconnection_correlations_occurrence.csv",
            output_dir / f"{input_stem}_teleconnection_corr_heatmap_occurrence.png",
            output_dir / f"{input_stem}_teleconnection_vif_occurrence.csv",
            "Teleconnection correlations (occurrence design)",
            dpi,
        )

    if len(tele_cols_amt) >= 2:
        correlation_and_vif(
            d_amt,
            tele_cols_amt,
            output_dir / f"{input_stem}_teleconnection_correlations_amounts.csv",
            output_dir / f"{input_stem}_teleconnection_corr_heatmap_amounts.png",
            output_dir / f"{input_stem}_teleconnection_vif_amounts.csv",
            "Teleconnection correlations (amounts design)",
            dpi,
        )

    with (output_dir / f"{input_stem}_diagnostics_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Extended diagnostics for final two-stage GLMs\n")
        f.write("=============================================\n\n")
        f.write(f"Input file: {input_csv}\n\n")

        f.write("Occurrence model\n")
        f.write(f"  N = {len(y_occ)}\n")
        f.write(f"  LogLik = {res_occ.llf:.3f}\n")
        f.write(f"  AIC = {res_occ.aic:.3f}\n\n")

        f.write("Amounts model\n")
        f.write(f"  N = {len(y_amt)}\n")
        f.write(f"  LogLik = {res_amt.llf:.3f}\n")
        f.write(f"  AIC = {res_amt.aic:.3f}\n")
        f.write(f"  Gamma scale = {scale_amt:.5f}\n")
        f.write(f"  Implied Gamma shape = {1.0 / scale_amt:.5f}\n\n")

        tail_tbl = pd.read_csv(output_dir / f"{input_stem}_gamma_upper_tail_table.csv")
        f.write("Tail exceedance checks (observed vs expected)\n")
        f.write(tail_tbl.to_string(index=False))
        f.write("\n\nSaved files:\n")
        for p in sorted(output_dir.glob(f"{input_stem}_*")):
            f.write(f" - {p.name}\n")

    print("[OK] Extended diagnostics complete.")
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()