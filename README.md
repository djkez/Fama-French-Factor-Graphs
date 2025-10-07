# Fama–French Factor Graphs

Fama–French Factor Graphs is a Python-based analytical tool for visualising factor model regressions using the Fama–French framework.

The program enables users to plot and compare exposures to the standard Fama–French 2×3 factors (Market, SMB, HML, RMW, CMA) alongside the momentum factor, producing publication-quality graphs for research or reporting.

The repository also includes a merging utility that combines the Fama–French 2×3 factor dataset with the Momentum data files (both downloadable from Kenneth French’s Data Library). This merged version allows for six-factor analysis within the same time window.

# Important! 
You will need to download Fama/French 5 Factors (2x3) and Momentum Factor (Mom) from Ken French's website.
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

Merge the Fama–French 2×3 factors with Momentum

# This repo includes a small utility, Merge_FF_Momentum.py, that combines the monthly Fama–French 2×3 factor file with the Momentum file from Kenneth French’s Data Library so you can run six-factor analysis on aligned dates.

Get the two input files (monthly)

Download ‘F-F_Research_Data_5_Factors_2x3’ (Monthly) — CSV version.

Download ‘F-F_Momentum_Factor’ (Monthly) — CSV version.

Do not tick ‘In percent’. Leave returns in decimal form.

Unzip the downloads so you have two .csv files.

Quick start (no arguments)

Double-click Merge_FF_Momentum.py (or run it with Python).
You’ll be prompted to pick:

the 5-Factors 2×3 CSV, then

the Momentum CSV,
and the script will write a merged file named:

Merged.FF.Factors.Plus.Momentum.csv

in the current folder (or ./data/ if that folder exists).

Command-line usage (PowerShell)
# From the repo root (adjust paths if needed)
python ".\Merge_FF_Momentum.py" `
  --ff_csv ".\data\F-F_Research_Data_5_Factors_2x3.csv" `
  --mom_csv ".\data\F-F_Momentum_Factor.csv" `
  --out ".\data\Merged.FF.Factors.Plus.Momentum.csv"

Arguments

--ff_csv Path to the F-F_Research_Data_5_Factors_2x3 CSV (monthly, decimals)

--mom_csv Path to the F-F_Momentum_Factor CSV (monthly, decimals)

--out Output path for the merged CSV (optional; defaults to Merged.FF.Factors.Plus.Momentum.csv)

What the script does

Parses and cleans French’s headers/footers.

Harmonises dates to monthly and takes the intersection of available months.

Normalises column names and keeps returns in decimal units.

Outputs columns similar to:

Date, Mkt-RF, SMB, HML, RMW, CMA, Mom, RF

(Date formatted as YYYY-MM.)

Using the merged file in the app

In Factor Graphs, load Merged.FF.Factors.Plus.Momentum.csv when you want momentum included.
**If charts look wrong, recheck that the inputs were downloaded without ‘In percent’.**

A standalone Windows .exe is provided under Releases so the tool can be run without a Python installation.

# Download the latest Windows executable from the **Releases** tab.

## Run without Python
1. Go to **Releases** (right side of the repo page).
2. Download the `.zip` file.
3. Unzip and double-click `Factor Graphs.exe`.

## Run with Python instead (developers)
```bash
pip install -r Libraries Required.txt
python "Factor Graphs.py"
