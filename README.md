Haor Rainfall GLM Workflow

This repository contains the core analysis workflow for a two-stage generalized linear modeling (GLM) study of daily rainfall occurrence and wet-day intensity.

The workflow is organized as a reproducible sequence of scripts for:

1. building a daily rainfall modeling table from gridded rainfall and climate-driver data
2. fitting two-stage rainfall GLMs
3. conducting year-block bootstrap inference
4. running extended diagnostic checks
5. summarizing early-versus-late rainfall changes


Repository contents

haor-rainfall-glm/
README.md
requirements.txt
01_build_daily_model_input_table.py
02_fit_occurrence_and_intensity_glms.py
03_conduct_year_block_bootstrap_analysis.py
04_run_extended_diagnostic_checks.py
05_summarize_early_late_changes.py


Required Python packages

Install dependencies with:

pip install -r requirements.txt

Typical packages used in this repository are:

- numpy
- pandas
- xarray
- netcdf4
- dask
- matplotlib
- scipy
- statsmodels


Input data

This repository does not include raw input data.

Users should prepare the required inputs separately, including:

- CHIRPS daily rainfall NetCDF files
- daily climate-driver CSV file

The scripts are designed so that users can provide their own paths and settings through command-line arguments.


Workflow overview

1. Build daily rainfall modeling table

Script:
01_build_daily_model_input_table.py

Purpose:
- read CHIRPS daily NetCDF files
- subset a user-defined region
- compute area-weighted daily mean rainfall
- merge rainfall with daily climate-driver data
- create wet/dry, lag, and seasonal predictors for two-stage GLMs

Example:
python 01_build_daily_model_input_table.py --chirps-dir data/chirpsp25 --drivers-csv data/climate_drivers/climate_drivers_daily_1981_2025.csv --output-dir output/step3 --region-name haor_NEIndia_Bangladesh --bbox 88.0 93.5 23.0 26.5 --tau-mm 0.1 --wet-lags 5 --rain-lags 4

Main outputs:
- chirps_region_daily_<region_name>_<start>_<end>.csv
- glm_input_daily_<region_name>_<start>_<end>.csv


2. Fit occurrence and intensity GLMs

Script:
02_fit_occurrence_and_intensity_glms.py

Purpose:
- fit a two-stage rainfall GLM framework
- Stage 1: occurrence model using Binomial GLM with logit link
- Stage 2: wet-day amount model using Gamma GLM with log link
- compare baseline, trend, and climate-driver scenarios
- save model comparison tables, summaries, and core diagnostics

Example:
python 02_fit_occurrence_and_intensity_glms.py --input output/step3/glm_input_daily_haor_NEIndia_Bangladesh_1981_2024.csv --output-dir output/step4 --break-year 2000 --use-mjo yes --use-cluster-robust-se yes --cluster-col year

Main outputs include:
- occurrence model comparison table
- amount model comparison table
- best-model summary text files
- calibration and residual diagnostic figures


3. Conduct year-block bootstrap analysis

Script:
03_conduct_year_block_bootstrap_analysis.py

Purpose:
- fit the selected final occurrence and amount models
- perform year-block bootstrap resampling
- estimate bootstrap confidence intervals
- compute bootstrap-based significance summaries

Example:
python 03_conduct_year_block_bootstrap_analysis.py --input output/step3/glm_input_daily_haor_NEIndia_Bangladesh_1981_2024.csv --output-dir output/step4_bootstrap --bootstrap-size 600 --seed 20251204 --year-col year --tau-mm 0.1 --use-mjo yes

Main outputs include:
- bootstrap summary table for the occurrence model
- bootstrap summary table for the amount model


4. Run extended diagnostic checks

Script:
04_run_extended_diagnostic_checks.py

Purpose:
- run residual diagnostics for both stages
- check calibration of the occurrence model
- assess Gamma fit for wet-day rainfall intensity
- produce PIT and quantile residual checks
- examine upper-tail performance of the Gamma model
- compute teleconnection correlations and VIF diagnostics

Example:
python 04_run_extended_diagnostic_checks.py --input output/step3/glm_input_daily_haor_NEIndia_Bangladesh_1981_2024.csv --output-dir output/step4_diagnostics --tau-mm 0.1 --use-mjo yes --dpi 300 --font-size 11

Main outputs include:
- residual plots
- calibration plots
- Gamma PIT and quantile residual figures
- upper-tail exceedance checks
- teleconnection correlation and VIF tables


5. Summarize early-versus-late changes

Script:
05_summarize_early_late_changes.py

Purpose:
- split the study period into early and late periods
- summarize wet-day occurrence and rainfall statistics
- estimate bootstrap confidence intervals for early-versus-late differences
- generate a reviewer-friendly contrast figure
- create a manuscript-ready text summary

Example:
python 05_summarize_early_late_changes.py --input output/step3/glm_input_daily_haor_NEIndia_Bangladesh_1981_2024.csv --output-dir output/step4_earlylate --tau-mm 0.1 --early-start 1981 --early-end 2002 --late-start 2003 --late-end 2024 --bootstrap-size 500 --seed 20260423 --region-label "Haor Basin, Bangladesh"

Main outputs include:
- early-versus-late daily summary table
- early-versus-late seasonal summary table
- bootstrap summary table
- contrast figure
- manuscript-ready text summary


Recommended run order

Run the scripts in this order:

1. 01_build_daily_model_input_table.py
2. 02_fit_occurrence_and_intensity_glms.py
3. 03_conduct_year_block_bootstrap_analysis.py
4. 04_run_extended_diagnostic_checks.py
5. 05_summarize_early_late_changes.py


Notes

- Raw data are not included in this repository.
- Users should provide their own local data paths through the command line.
- Output filenames are automatically built from the input file stem where applicable.
- The repository is intended to share the core reproducible analysis workflow of the study.


Code availability

The code used for data preprocessing, two-stage GLM fitting, bootstrap inference, extended diagnostics, and early-versus-late summary analysis is publicly available in this repository to support transparency and reproducibility.