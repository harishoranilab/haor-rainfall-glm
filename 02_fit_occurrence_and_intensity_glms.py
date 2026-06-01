import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    import matplotlib.pyplot as plt
except Exception:  
    plt = None


DEFAULT_YEAR_LENGTH = 365.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the exact two-stage rainfall GLM specified in equation.pdf."
    )
    parser.add_argument("--input", type=Path, required=True, help="Daily model input CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/strict_equation_glm"),
        help="Directory for model summaries and fitted outputs.",
    )
    parser.add_argument(
        "--tau-mm",
        type=float,
        default=0.1,
        help="Wet-day threshold in mm/day. Must match equation.pdf; default is 0.1.",
    )
    parser.add_argument(
        "--year-length",
        type=float,
        default=DEFAULT_YEAR_LENGTH,
        help="Denominator for seasonal harmonics. equation.pdf uses 365; default is 365.",
    )
    parser.add_argument(
        "--min-years",
        type=int,
        default=10,
        help="Minimum number of years required after complete-case filtering.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for optional diagnostic figures.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: Iterable[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {missing}")


def clean_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_mjo_xy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure mjo_x and mjo_y exist because equation.pdf uses MJOx and MJOy.

    Priority:
    1. Existing mjo_x and mjo_y.
    2. Convert mjo_phase and mjo_amp to Cartesian terms.
    3. Use rmm1 and rmm2 as mjo_x and mjo_y aliases.
    """
    if {"mjo_x", "mjo_y"}.issubset(df.columns):
        return df

    if {"mjo_phase", "mjo_amp"}.issubset(df.columns):
        phase = pd.to_numeric(df["mjo_phase"], errors="coerce")
        amp = pd.to_numeric(df["mjo_amp"], errors="coerce")
        theta = 2.0 * np.pi * (phase - 1.0) / 8.0
        df["mjo_x"] = amp * np.cos(theta)
        df["mjo_y"] = amp * np.sin(theta)
        return df

    if {"rmm1", "rmm2"}.issubset(df.columns):
        df["mjo_x"] = pd.to_numeric(df["rmm1"], errors="coerce")
        df["mjo_y"] = pd.to_numeric(df["rmm2"], errors="coerce")
        return df

    raise ValueError(
        "equation.pdf requires MJOx and MJOy. Provide either mjo_x/mjo_y, "
        "mjo_phase/mjo_amp, or rmm1/rmm2 in the input CSV."
    )


def recompute_equation_terms(df: pd.DataFrame, tau_mm: float, year_length: float) -> pd.DataFrame:
    """Recompute all terms exactly needed by equation.pdf."""
    require_columns(df, ["date", "rain_mmday"], "equation-term construction")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
    df = df.sort_values("date").reset_index(drop=True)
    df["rain_mmday"] = pd.to_numeric(df["rain_mmday"], errors="coerce")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["doy"] = df["date"].dt.dayofyear.astype(float)

   
    df["wet"] = (df["rain_mmday"] > tau_mm).astype(int)
    df["rain_wet_mmday"] = df["rain_mmday"].where(df["wet"] == 1, np.nan)

  
    omega1_d = 2.0 * np.pi * df["doy"] / year_length
    omega2_d = 4.0 * np.pi * df["doy"] / year_length
    df["sin_doy1"] = np.sin(omega1_d)
    df["cos_doy1"] = np.cos(omega1_d)
    df["sin_doy2"] = np.sin(omega2_d)
    df["cos_doy2"] = np.cos(omega2_d)

   
    for j in range(1, 6):
        df[f"wet_lag{j}"] = df["wet"].shift(j)
    df["prev5_wet_count"] = df["wet"].shift(1).rolling(5, min_periods=5).sum()


    if (df["rain_mmday"].dropna() < 0).any():
        raise ValueError("rain_mmday contains negative values; Gamma/log1p rainfall model requires non-negative rainfall.")
    df["log1p_rain"] = np.log1p(df["rain_mmday"])
    for j in range(1, 5):
        df[f"log1p_rain_lag{j}"] = df["log1p_rain"].shift(j)
    df["prev5_rain_sum"] = df["rain_mmday"].shift(1).rolling(5, min_periods=5).sum()

  
    df["year_c"] = df["year"] - df["year"].median()

  
    require_columns(df, ["nino34", "dmi"], "climate-driver vector")
    df = add_mjo_xy(df)
    df = clean_numeric(df, ["nino34", "dmi", "mjo_x", "mjo_y"])

  
    df["nino34_x_sin1"] = df["nino34"] * df["sin_doy1"]
    df["nino34_x_cos1"] = df["nino34"] * df["cos_doy1"]
    df["dmi_x_sin1"] = df["dmi"] * df["sin_doy1"]
    df["dmi_x_cos1"] = df["dmi"] * df["cos_doy1"]

    return df


def occurrence_columns() -> list[str]:
    """Columns in the Stage 1 linear predictor eta_occ(t), matching equation.pdf."""
    return (
        ["sin_doy1", "cos_doy1", "sin_doy2", "cos_doy2"]
        + [f"wet_lag{j}" for j in range(1, 6)]
        + ["prev5_wet_count"]
        + ["nino34", "dmi", "mjo_x", "mjo_y", "year_c"]
        + ["nino34_x_sin1", "nino34_x_cos1", "dmi_x_sin1", "dmi_x_cos1"]
    )


def amount_columns() -> list[str]:
    """Columns in the Stage 2 linear predictor eta_amt(t), matching equation.pdf."""
    return (
        ["sin_doy1", "cos_doy1", "sin_doy2", "cos_doy2"]
        + [f"log1p_rain_lag{j}" for j in range(1, 5)]
        + ["prev5_rain_sum"]
        + ["nino34", "dmi", "mjo_x", "mjo_y"]
        + ["nino34_x_cos1", "nino34_x_sin1", "dmi_x_cos1", "dmi_x_sin1"]
    )


def fit_glm(data: pd.DataFrame, y_col: str, x_cols: list[str], family) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, object]:
    require_columns(data, [y_col] + x_cols, f"GLM fit for {y_col}")
    d = data.dropna(subset=[y_col] + x_cols).copy()
    y = d[y_col].astype(float)
    X = sm.add_constant(d[x_cols].astype(float), has_constant="add")
    res = sm.GLM(y, X, family=family).fit()
    return d, y, X, res


def save_model_outputs(
    output_dir: Path,
    stem: str,
    stage_name: str,
    d: pd.DataFrame,
    y: pd.Series,
    X: pd.DataFrame,
    res,
) -> None:
    
    with (output_dir / f"{stem}_{stage_name}_summary.txt").open("w", encoding="utf-8") as f:
        f.write(res.summary().as_text())
        f.write("\n\nModel columns in order:\n")
        for c in X.columns:
            f.write(f" - {c}\n")

    
    conf = res.conf_int()
    coef_tbl = pd.DataFrame(
        {
            "term": res.params.index,
            "coef": res.params.values,
            "std_err": res.bse.values,
            "z": res.tvalues.values,
            "p_value": res.pvalues.values,
            "ci_lower": conf.iloc[:, 0].values,
            "ci_upper": conf.iloc[:, 1].values,
        }
    )
    coef_tbl.to_csv(output_dir / f"{stem}_{stage_name}_coefficients.csv", index=False)

    # Fitted table
    fitted_tbl = d[["date", "year", "month", "doy"]].copy()
    fitted_tbl["observed"] = y.values
    fitted_tbl["fitted"] = np.asarray(res.fittedvalues)
    fitted_tbl["pearson_resid"] = np.asarray(res.resid_pearson)
    fitted_tbl.to_csv(output_dir / f"{stem}_{stage_name}_fitted_values.csv", index=False)


def write_audit_file(
    output_dir: Path,
    stem: str,
    input_csv: Path,
    tau_mm: float,
    year_length: float,
    occ_cols: list[str],
    amt_cols: list[str],
    res_occ,
    res_amt,
    n_years_occ: int,
    n_years_amt: int,
) -> None:
    lines = []
    lines.append("Strict equation.pdf GLM audit")
    lines.append("================================")
    lines.append(f"Input CSV: {input_csv}")
    lines.append(f"Wet threshold tau_mm: {tau_mm}")
    lines.append(f"Seasonal harmonic denominator: {year_length}")
    lines.append("")
    lines.append("Stage 1 occurrence model implemented exactly as:")
    lines.append("  wet ~ Binomial(logit), response = W_t")
    lines.append("  predictors = seasonal harmonics + W_t lags 1-5 + prev5_wet_count + X_t")
    lines.append("  X_t includes nino34, dmi, mjo_x, mjo_y, year_c, and ENSO/DMI seasonal interactions")
    lines.append(f"  N = {int(res_occ.nobs)}, years = {n_years_occ}, AIC = {float(res_occ.aic):.6f}")
    lines.append("  Columns:")
    for c in ["const"] + occ_cols:
        lines.append(f"    - {c}")
    lines.append("")
    lines.append("Stage 2 intensity model implemented exactly as:")
    lines.append("  rain_wet_mmday | wet day ~ Gamma(log), response = Y_t = P_t | P_t > tau")
    lines.append("  predictors = seasonal harmonics + log1p rainfall lags 1-4 + prev5_rain_sum + Z_t")
    lines.append("  Z_t includes nino34, dmi, mjo_x, mjo_y, and ENSO/DMI seasonal interactions")
    lines.append("  IMPORTANT: year_c is NOT included in Stage 2 because equation.pdf Z_t does not include it.")
    lines.append(f"  N = {int(res_amt.nobs)}, years = {n_years_amt}, AIC = {float(res_amt.aic):.6f}")
    lines.append("  Columns:")
    for c in ["const"] + amt_cols:
        lines.append(f"    - {c}")
    lines.append("")
    lines.append("No model-selection scenarios, no piecewise trend, and no Stage-2 trend are fitted in this strict version.")

    with (output_dir / f"{stem}_STRICT_equation_audit.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_basic_diagnostics(output_dir: Path, stem: str, d_occ: pd.DataFrame, res_occ, d_amt: pd.DataFrame, res_amt, dpi: int) -> None:
    if plt is None:
        return

    
    occ = d_occ[["wet"]].copy()
    occ["p_wet"] = np.asarray(res_occ.fittedvalues)
    occ["bin"] = pd.qcut(occ["p_wet"], q=10, duplicates="drop")
    g = occ.groupby("bin", observed=True).agg(p_mean=("p_wet", "mean"), obs_freq=("wet", "mean"), n=("wet", "size")).reset_index()

    fig = plt.figure(figsize=(6.2, 5.2))
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    ax.scatter(g["p_mean"], g["obs_freq"], s=np.clip(g["n"].values / 50, 10, 150))
    ax.set_xlabel("Mean predicted wet-day probability")
    ax.set_ylabel("Observed wet-day frequency")
    ax.set_title("Stage 1 occurrence calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_occurrence_calibration.png", dpi=dpi)
    plt.close(fig)

    # Amount residual vs fitted
    fig = plt.figure(figsize=(6.2, 5.2))
    ax = plt.gca()
    ax.scatter(np.asarray(res_amt.fittedvalues), np.asarray(res_amt.resid_pearson), s=8, alpha=0.55)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Fitted conditional wet-day rainfall, mu_t (mm/day)")
    ax.set_ylabel("Pearson residual")
    ax.set_title("Stage 2 Gamma amount residuals")
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_amount_residual_vs_fitted.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_csv = args.input
    output_dir = args.output_dir
    tau_mm = args.tau_mm
    year_length = args.year_length
    min_years = args.min_years
    dpi = args.dpi

    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(input_csv, parse_dates=["date"])
    df = recompute_equation_terms(df_raw, tau_mm=tau_mm, year_length=year_length)

    occ_cols = occurrence_columns()
    amt_cols = amount_columns()

    
    df_model = df.dropna(subset=occ_cols + amt_cols + ["wet", "rain_wet_mmday"]).copy()

    n_years_total = df_model["year"].nunique()
    if n_years_total < min_years:
        raise ValueError(f"Too few complete years after filtering: {n_years_total}. Required at least {min_years}.")

    stem = input_csv.stem
    df_model.to_csv(output_dir / f"{stem}_equation_terms_used.csv", index=False)

    
    d_occ, y_occ, X_occ, res_occ = fit_glm(
        df_model,
        y_col="wet",
        x_cols=occ_cols,
        family=sm.families.Binomial(sm.families.links.Logit()),
    )

    
    df_wet = df_model[(df_model["wet"] == 1) & (df_model["rain_wet_mmday"] > tau_mm)].copy()
    d_amt, y_amt, X_amt, res_amt = fit_glm(
        df_wet,
        y_col="rain_wet_mmday",
        x_cols=amt_cols,
        family=sm.families.Gamma(sm.families.links.Log()),
    )

    save_model_outputs(output_dir, stem, "stage1_occurrence", d_occ, y_occ, X_occ, res_occ)
    save_model_outputs(output_dir, stem, "stage2_intensity", d_amt, y_amt, X_amt, res_amt)

    
    X_occ_all = sm.add_constant(df_model[occ_cols].astype(float), has_constant="add")
    X_amt_all = sm.add_constant(df_model[amt_cols].astype(float), has_constant="add")
    pred = df_model[["date", "year", "month", "doy", "rain_mmday", "wet", "rain_wet_mmday"]].copy()
    pred["p_wet"] = np.asarray(res_occ.predict(X_occ_all))
    pred["mu_wet_day_mmday"] = np.asarray(res_amt.predict(X_amt_all))
    pred["expected_rain_mmday_two_stage"] = pred["p_wet"] * pred["mu_wet_day_mmday"]
    pred.to_csv(output_dir / f"{stem}_two_stage_predictions.csv", index=False)

    write_audit_file(
        output_dir=output_dir,
        stem=stem,
        input_csv=input_csv,
        tau_mm=tau_mm,
        year_length=year_length,
        occ_cols=occ_cols,
        amt_cols=amt_cols,
        res_occ=res_occ,
        res_amt=res_amt,
        n_years_occ=d_occ["year"].nunique(),
        n_years_amt=d_amt["year"].nunique(),
    )

    plot_basic_diagnostics(output_dir, stem, d_occ, res_occ, d_amt, res_amt, dpi)

    print("[OK] Strict equation.pdf two-stage GLM fitted.")
    print("Input:", input_csv)
    print("Output directory:", output_dir)
    print("Stage 1 occurrence: N =", int(res_occ.nobs), "AIC =", float(res_occ.aic))
    print("Stage 2 intensity : N =", int(res_amt.nobs), "AIC =", float(res_amt.aic))
    print("Saved key files:")
    print(" -", output_dir / f"{stem}_STRICT_equation_audit.txt")
    print(" -", output_dir / f"{stem}_stage1_occurrence_summary.txt")
    print(" -", output_dir / f"{stem}_stage2_intensity_summary.txt")
    print(" -", output_dir / f"{stem}_two_stage_predictions.csv")


if __name__ == "__main__":
    main()
