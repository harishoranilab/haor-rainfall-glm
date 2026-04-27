import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


VAR_CANDIDATES = ["precip", "precipitation", "rain", "prcp", "tp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a daily rainfall GLM input table from CHIRPS and climate-driver data."
    )
    parser.add_argument(
        "--chirps-dir",
        type=Path,
        required=True,
        help="Directory containing CHIRPS daily NetCDF files.",
    )
    parser.add_argument(
        "--drivers-csv",
        type=Path,
        required=True,
        help="Path to daily climate-driver CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/step3"),
        help="Directory for Step 3 outputs.",
    )
    parser.add_argument(
        "--region-name",
        type=str,
        required=True,
        help="Name used in output filenames.",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        required=True,
        help="Bounding box: lon_min lon_max lat_min lat_max",
    )
    parser.add_argument(
        "--tau-mm",
        type=float,
        default=0.1,
        help="Wet-day threshold in mm/day.",
    )
    parser.add_argument(
        "--wet-lags",
        type=int,
        default=5,
        help="Number of wet-day lag predictors.",
    )
    parser.add_argument(
        "--rain-lags",
        type=int,
        default=4,
        help="Number of lagged rainfall predictors.",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=365,
        help="Time chunk size for xarray/dask.",
    )
    parser.add_argument(
        "--lat-chunk",
        type=int,
        default=120,
        help="Latitude chunk size for xarray/dask.",
    )
    parser.add_argument(
        "--lon-chunk",
        type=int,
        default=120,
        help="Longitude chunk size for xarray/dask.",
    )
    return parser.parse_args()


def infer_year_range_from_chirps(chirps_dir: Path) -> tuple[int, int, list[Path]]:
    pat = re.compile(r"chirps-v2\.0\.(\d{4})\.days_p25\.nc$")
    files = sorted(chirps_dir.glob("chirps-v2.0.*.days_p25.nc"))

    years = []
    matched_files = []

    for path in files:
        match = pat.search(path.name)
        if match:
            years.append(int(match.group(1)))
            matched_files.append(path)

    if not years:
        raise FileNotFoundError(
            f"No CHIRPS files matching 'chirps-v2.0.YYYY.days_p25.nc' were found in: {chirps_dir}"
        )

    return min(years), max(years), matched_files


def find_coord_name(ds: xr.Dataset, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in ds.coords:
            return candidate
    for candidate in candidates:
        if candidate in ds.variables:
            return candidate
    raise KeyError(f"Could not find any of these coordinates/variables: {candidates}")


def find_rain_var(ds: xr.Dataset) -> str:
    for var_name in VAR_CANDIDATES:
        if var_name in ds.data_vars:
            return var_name

    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]

    raise KeyError(f"Could not detect rainfall variable. Found: {list(ds.data_vars)}")


def normalize_longitudes(
    lon: xr.DataArray,
    target_min: float,
    target_max: float,
) -> tuple[xr.DataArray, float, float]:
    lon_vals = lon.values
    lon_min_ds = float(np.nanmin(lon_vals))
    lon_max_ds = float(np.nanmax(lon_vals))

    if lon_min_ds >= 0.0 and lon_max_ds > 180.0 and (target_min < 0.0 or target_max <= 180.0):
        lon_new = ((lon + 180) % 360) - 180
        return lon_new, target_min, target_max

    if lon_min_ds < 0.0 and (target_min >= 0.0 and target_max > 180.0):
        def convert(x: float) -> float:
            return ((x + 180) % 360) - 180
        return lon, convert(target_min), convert(target_max)

    return lon, target_min, target_max


def lat_slice(lat_vals: xr.DataArray, lat_min: float, lat_max: float):
    increasing = bool(lat_vals[1].item() > lat_vals[0].item()) if lat_vals.size >= 2 else True
    return slice(lat_min, lat_max) if increasing else slice(lat_max, lat_min)


def lon_slice(lon_vals: xr.DataArray, lon_min: float, lon_max: float):
    increasing = bool(lon_vals[1].item() > lon_vals[0].item()) if lon_vals.size >= 2 else True
    return slice(lon_min, lon_max) if increasing else slice(lon_max, lon_min)


def add_seasonality_terms(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    doy = df[date_col].dt.dayofyear.astype(float)
    df["sin_doy1"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy1"] = np.cos(2 * np.pi * doy / 365.25)
    df["sin_doy2"] = np.sin(4 * np.pi * doy / 365.25)
    df["cos_doy2"] = np.cos(4 * np.pi * doy / 365.25)
    return df


def add_mjo_cartesian(df: pd.DataFrame) -> pd.DataFrame:
    if "mjo_phase" in df.columns and "mjo_amp" in df.columns:
        phase = df["mjo_phase"].astype(float)
        amp = df["mjo_amp"].astype(float)
        theta = 2 * np.pi * (phase - 1.0) / 8.0

        df["mjo_x"] = amp * np.cos(theta)
        df["mjo_y"] = amp * np.sin(theta)
        df["mjo_active"] = (amp >= 1.0).astype(int)

    return df


def pct_missing(series: pd.Series) -> float:
    return float(100.0 * series.isna().mean())


def main() -> None:
    args = parse_args()

    chirps_dir = args.chirps_dir
    drivers_csv = args.drivers_csv
    output_dir = args.output_dir
    region_name = args.region_name
    bbox = tuple(args.bbox)
    tau_mm = args.tau_mm
    wet_lags = args.wet_lags
    rain_lags = args.rain_lags
    time_chunk = args.time_chunk
    lat_chunk = args.lat_chunk
    lon_chunk = args.lon_chunk

    output_dir.mkdir(parents=True, exist_ok=True)

    start_year, end_year, chirps_files = infer_year_range_from_chirps(chirps_dir)
    print(f"[OK] CHIRPS files detected: {len(chirps_files)} | years {start_year}-{end_year}")

    open_kwargs = dict(combine="by_coords", decode_times=True)

    try:
        ds = xr.open_mfdataset(
            [str(path) for path in chirps_files],
            chunks={"time": time_chunk, "latitude": lat_chunk, "longitude": lon_chunk},
            **open_kwargs,
        )
    except Exception:
        ds = xr.open_mfdataset([str(path) for path in chirps_files], **open_kwargs)

    lat_name = find_coord_name(ds, ["latitude", "lat", "y"])
    lon_name = find_coord_name(ds, ["longitude", "lon", "x"])
    time_name = find_coord_name(ds, ["time", "t", "date"])
    rain_var = find_rain_var(ds)

    print(f"[OK] Detected names -> lat: {lat_name}, lon: {lon_name}, time: {time_name}, var: {rain_var}")

    lon = ds[lon_name]
    lon, lon_min, lon_max = normalize_longitudes(lon, bbox[0], bbox[1])
    lat_min, lat_max = bbox[2], bbox[3]

    if lon is not ds[lon_name]:
        ds = ds.assign_coords({lon_name: lon})

    ds_sub = ds.sel(
        {
            lat_name: lat_slice(ds[lat_name], lat_min, lat_max),
            lon_name: lon_slice(ds[lon_name], lon_min, lon_max),
        }
    )

    da = ds_sub[rain_var]
    units = da.attrs.get("units", "unknown")
    print(f"[INFO] Rain units attribute: {units}")

    weights = np.cos(np.deg2rad(ds_sub[lat_name]))
    weights = xr.DataArray(weights, dims=[lat_name], coords={lat_name: ds_sub[lat_name]})

    w2d = weights.broadcast_like(da.isel({time_name: 0}))
    numerator = (da * w2d).sum(dim=[lat_name, lon_name], skipna=True)
    denominator = w2d.where(np.isfinite(da.isel({time_name: 0}))).sum(
        dim=[lat_name, lon_name],
        skipna=True,
    )

    rain_mean = (numerator / denominator).rename("rain_mmday")

    df_rain = rain_mean.to_series().reset_index()
    df_rain = df_rain.rename(columns={time_name: "date"})
    df_rain["date"] = pd.to_datetime(df_rain["date"]).dt.floor("D")
    df_rain = df_rain.sort_values("date").reset_index(drop=True)

    out_rain = output_dir / f"chirps_region_daily_{region_name}_{start_year}_{end_year}.csv"
    df_rain.to_csv(out_rain, index=False)
    print(f"[OK] Saved region rainfall daily series: {out_rain}")

    df_drv = pd.read_csv(drivers_csv, parse_dates=["date"])
    df_drv["date"] = pd.to_datetime(df_drv["date"]).dt.floor("D")

    df = df_rain.merge(df_drv, on="date", how="left")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["doy"] = df["date"].dt.dayofyear

    df["wet"] = (df["rain_mmday"] > tau_mm).astype(int)
    df["rain_wet_mmday"] = df["rain_mmday"].where(df["wet"] == 1, np.nan)

    df = add_seasonality_terms(df, "date")
    df = add_mjo_cartesian(df)

    for lag in range(1, wet_lags + 1):
        df[f"wet_lag{lag}"] = df["wet"].shift(lag)

    df["log1p_rain"] = np.log1p(df["rain_mmday"])
    for lag in range(1, rain_lags + 1):
        df[f"log1p_rain_lag{lag}"] = df["log1p_rain"].shift(lag)

    df["prev5_rain_sum"] = df["rain_mmday"].shift(1).rolling(5, min_periods=5).sum()
    df["prev5_wet_count"] = df["wet"].shift(1).rolling(5, min_periods=5).sum()
    df["is_JJAS"] = df["month"].isin([6, 7, 8, 9]).astype(int)

    min_lag = max(wet_lags, rain_lags, 5)
    df_out = df.iloc[min_lag:].copy()

    out_glm = output_dir / f"glm_input_daily_{region_name}_{start_year}_{end_year}.csv"
    df_out.to_csv(out_glm, index=False)
    print(f"[OK] Saved GLM input table: {out_glm}")

    print("\n[QC SUMMARY]")
    print("Date range:", df_out["date"].min().date(), "to", df_out["date"].max().date())
    print("N days:", len(df_out))
    print("Mean rain (mm/day):", float(df_out["rain_mmday"].mean()))
    print("Wet-day fraction:", float(df_out["wet"].mean()))

    for col in ["nino34", "dmi", "rmm1", "rmm2", "mjo_phase", "mjo_amp"]:
        if col in df_out.columns:
            print(f"Missing {col:10s}: {pct_missing(df_out[col]):6.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()