import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def str2bool_text(value):
    value = value.strip().lower()
    if value in {"yes", "true", "1", "y"}:
        return True
    if value in {"no", "false", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected yes/no, true/false, 1/0.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run year-block bootstrap inference for two-stage rainfall GLMs."
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
        default=Path("output/step4_bootstrap"),
        help="Directory for Step 4C outputs.",
    )
    parser.add_argument(
        "--bootstrap-size",
        type=int,
        default=600,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20251204,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--year-col",
        type=str,
        default="year",
        help="Column name used for year-block resampling.",
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


def fit_glm_on_df(data, y_col, x_cols, family):
    d2 = data.dropna(subset=[y_col] + x_cols).copy()
    y = d2[y_col].astype(float).values
    X = sm.add_constant(d2[x_cols].astype(float), has_constant="add")
    result = sm.GLM(y, X, family=family).fit()
    return result


def bootstrap_year_blocks(data, y_col, x_cols, family, year_col, n_boot, seed):
    """
    Resample years with replacement. Within each sampled year, keep all rows.
    Returns:
        draws: array of shape [n_boot, n_params]
        names: parameter names in order
    """
    rng = np.random.default_rng(seed)

    d_cc = data.dropna(subset=[year_col, y_col] + x_cols).copy()
    years = np.sort(d_cc[year_col].dropna().unique())

    blocks = {yy: d_cc[d_cc[year_col] == yy] for yy in years}
    valid_years = np.array([yy for yy in years if len(blocks[yy]) > 0])

    if len(valid_years) < 10:
        raise ValueError("Too few valid years after dropping missing rows.")

    res0 = fit_glm_on_df(d_cc, y_col=y_col, x_cols=x_cols, family=family)
    names = res0.params.index.tolist()
    n_params = len(names)

    draws = np.empty((n_boot, n_params), dtype=float)

    for b in range(n_boot):
        sampled_years = rng.choice(valid_years, size=len(valid_years), replace=True)
        db = pd.concat([blocks[yy] for yy in sampled_years], ignore_index=True)

        ok = False
        for _ in range(4):
            try:
                resb = fit_glm_on_df(db, y_col=y_col, x_cols=x_cols, family=family)
                draws[b, :] = resb.params.reindex(names).values
                ok = True
                break
            except Exception:
                sampled_years = rng.choice(valid_years, size=len(valid_years), replace=True)
                db = pd.concat([blocks[yy] for yy in sampled_years], ignore_index=True)

        if not ok:
            raise RuntimeError("Bootstrap fit failed repeatedly. Consider reducing collinearity or bootstrap size.")

    return draws, names


def summarize_bootstrap(draws, names, point_est):
    """
    Compute percentile confidence intervals and simple two-sided bootstrap p-values.
    """
    df_sum = pd.DataFrame({"term": names})
    df_sum["coef_hat"] = [float(point_est.get(term, np.nan)) for term in names]
    df_sum["boot_mean"] = draws.mean(axis=0)
    df_sum["boot_sd"] = draws.std(axis=0, ddof=1)
    df_sum["ci2.5"] = np.percentile(draws, 2.5, axis=0)
    df_sum["ci50"] = np.percentile(draws, 50.0, axis=0)
    df_sum["ci97.5"] = np.percentile(draws, 97.5, axis=0)

    p_le0 = (draws <= 0).mean(axis=0)
    p_ge0 = (draws >= 0).mean(axis=0)
    df_sum["p_boot_2s"] = 2 * np.minimum(p_le0, p_ge0)

    df_sum["is_const"] = (df_sum["term"] == "const").astype(int)
    df_sum = df_sum.sort_values(["is_const", "term"], ascending=[False, True]).drop(columns=["is_const"])
    return df_sum.reset_index(drop=True)


def main():
    args = parse_args()

    input_csv = args.input
    output_dir = args.output_dir
    n_boot = args.bootstrap_size
    seed = args.seed
    year_col = args.year_col
    tau_mm = args.tau_mm
    use_mjo = args.use_mjo

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    required_cols = [
        "date",
        year_col,
        "wet",
        "rain_wet_mmday",
        "sin_doy1",
        "cos_doy1",
        "sin_doy2",
        "cos_doy2",
    ]
    ensure_required_columns(df, required_cols)

    if "year_c" not in df.columns:
        df["year_c"] = df[year_col] - df[year_col].median()

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

    occ_extra = ["prev5_wet_count"] if "prev5_wet_count" in df.columns else []
    amt_extra = ["prev5_rain_sum"] if "prev5_rain_sum" in df.columns else []

    mjo_cols = build_mjo_columns(df, use_mjo)

    x_occ = (
        ["sin_doy1", "cos_doy1", "sin_doy2", "cos_doy2"]
        + wet_lags
        + occ_extra
        + [col for col in ["nino34", "dmi"] if col in df.columns]
        + [col for col in ["nino34_x_cos1", "nino34_x_sin1", "dmi_x_cos1", "dmi_x_sin1"] if col in df.columns]
        + mjo_cols
        + ["year_c"]
    )

    x_amt = (
        ["sin_doy1", "cos_doy1", "sin_doy2", "cos_doy2"]
        + rain_lags
        + amt_extra
        + [col for col in ["nino34", "dmi"] if col in df.columns]
        + [col for col in ["nino34_x_cos1", "nino34_x_sin1", "dmi_x_cos1", "dmi_x_sin1"] if col in df.columns]
        + mjo_cols
    )

    input_stem = input_csv.stem

    print("[1/2] Fitting occurrence point estimate...")
    res_occ = fit_glm_on_df(
        df,
        y_col="wet",
        x_cols=x_occ,
        family=sm.families.Binomial(sm.families.links.Logit()),
    )
    print("  LL:", float(res_occ.llf), "AIC:", float(res_occ.aic), "N:", int(res_occ.nobs))

    print("[1/2] Bootstrapping occurrence by year blocks...")
    draws_occ, names_occ = bootstrap_year_blocks(
        df,
        y_col="wet",
        x_cols=x_occ,
        family=sm.families.Binomial(sm.families.links.Logit()),
        year_col=year_col,
        n_boot=n_boot,
        seed=seed,
    )
    occ_tbl = summarize_bootstrap(draws_occ, names_occ, res_occ.params)
    out_occ = output_dir / f"{input_stem}_bootstrap_occurrence_by_year.csv"
    occ_tbl.to_csv(out_occ, index=False)
    print("  Saved:", out_occ)

    df_wet = df[(df["wet"] == 1) & (df["rain_wet_mmday"] > tau_mm)].copy()

    print("[2/2] Fitting amount point estimate...")
    res_amt = fit_glm_on_df(
        df_wet,
        y_col="rain_wet_mmday",
        x_cols=x_amt,
        family=sm.families.Gamma(sm.families.links.Log()),
    )
    print("  LL:", float(res_amt.llf), "AIC:", float(res_amt.aic), "N:", int(res_amt.nobs))

    print("[2/2] Bootstrapping amount model by year blocks...")
    draws_amt, names_amt = bootstrap_year_blocks(
        df_wet,
        y_col="rain_wet_mmday",
        x_cols=x_amt,
        family=sm.families.Gamma(sm.families.links.Log()),
        year_col=year_col,
        n_boot=n_boot,
        seed=seed + 1,
    )
    amt_tbl = summarize_bootstrap(draws_amt, names_amt, res_amt.params)
    out_amt = output_dir / f"{input_stem}_bootstrap_amounts_by_year.csv"
    amt_tbl.to_csv(out_amt, index=False)
    print("  Saved:", out_amt)

    print("\n[OK] Done.")
    print("Outputs:")
    print(" -", out_occ)
    print(" -", out_amt)


if __name__ == "__main__":
    main()