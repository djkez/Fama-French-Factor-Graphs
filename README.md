# Fama–French Factor Graphs

Fama–French Factor Graphs is a Python-based analytical tool for visualising factor model regressions using the Fama–French framework.

The program enables users to plot and compare exposures to the standard Fama–French 2×3 factors (Market, SMB, HML, RMW, CMA) alongside the momentum factor, producing quality graphs for research or reporting.

The repository also includes a merging utility that combines the Fama–French 2×3 factor dataset with the Momentum data files (both downloadable from Kenneth French’s Data Library). This merged version allows for six-factor analysis within the same time window.

# Important! 
You will need to download Fama/French 5 Factors (2x3) and Momentum Factor (Mom) from Ken French's website and merge the two files.
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

# Merging

This repo includes a small utility, Merge_FF_Momentum.py, that combines the monthly Fama–French 2×3 factor file with the Momentum file from Kenneth French’s Data Library so you can run six-factor analysis on aligned dates.

Get the two input files (monthly)

Download ‘F-F_Research_Data_5_Factors_2x3’ (Monthly) — CSV version.

Download ‘F-F_Momentum_Factor’ (Monthly) — CSV version.

# Command-line usage (PowerShell)
**1. Open PowerShell**

Press Start → type PowerShell → hit Enter.

You should see a black/blue PowerShell window.

**2. Go to the folder where the script is saved**

If you saved it into your Documents\Python Programs folder, type:

cd "C:\Users\Kiera\Documents\Python Programs"

(Replace the path with wherever merge_ff_momentum.py actually is for you.)

**3. Run the script with your two CSV files**

Example command:

python merge_ff_momentum.py "F-F_Research_Data_5_Factors_2x3.csv" "F-F_Momentum_Factor.csv" -o merged_ff_5f_plus_mom.csv

Explanation:

python merge_ff_momentum.py → runs the script

"F-F_Research_Data_5_Factors_2x3.csv" → your 5-factor file

"F-F_Momentum_Factor.csv" → the momentum factor file

-o merged_ff_5f_plus_mom.csv → output filename (you can rename it if you want)

**4. If you get an Excel pop-up when inspecting the new merged file**

That popup is just Excel being Excel 😅 — it’s warning you that when it opens a CSV, it might automatically show large numbers (like dates written as 202307) in scientific notation (e.g., 2.02307E+05).

For your merged factor file, you should click Don’t Convert.

**Here’s why:**

The script already cleaned your dates into a proper YYYY-MM-DD format, so there shouldn’t be any big integer values left.

If you click Convert, Excel will still try to 'help' by auto-formatting columns, which can sometimes mess things up.

If you click Don’t Convert, Excel will show the raw values exactly as written in the CSV (no unwanted conversions).

**And importantly:**

This popup only affects how Excel displays the CSV, not how Python or your GUI reads it.

So even if you accidentally click Convert, your GUI will still read the file fine — it doesn’t use Excel.

**What the script does**

Parses and cleans French’s headers/footers.

Harmonises dates to monthly and takes the intersection of available months.

Normalises column names and keeps returns in decimal units.

**Outputs columns similar to:**

Date, Mkt-RF, SMB, HML, RMW, CMA, Mom, RF

(Date formatted as YYYY-MM.)

# Using the merged file in the app

**Do not tick ‘In percent’. Leave returns in decimal form.**

In Factor Graphs, load Merged.FF.Factors.Plus.Momentum.csv when you want momentum included.
**If charts look wrong, recheck that the inputs were downloaded without ‘In percent’.**

A standalone Windows .exe is provided under Releases so the tool can be run without a Python installation.

# Download the latest Windows executable from the **Releases** tab.

## Run without Python
1. Go to **Releases** (right side of the repo page).
2. Download the `Factor.Graphs.exe` file.
3. Double-click `Factor Graphs.exe`.

## Run with Python instead (developers)
```bash
pip install -r Libraries Required.txt
python "Factor Graphs.py"
