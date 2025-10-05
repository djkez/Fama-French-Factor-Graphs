# Fama–French Factor Graphs

Fama–French Factor Graphs is a Python-based analytical tool for visualising factor model regressions using the Fama–French framework.

The program enables users to plot and compare exposures to the standard Fama–French 2×3 factors (Market, SMB, HML, RMW, CMA) alongside the momentum factor, producing publication-quality graphs for research or reporting.

The repository also includes a merging utility that combines the Fama–French 2×3 factor dataset with the Momentum data files (both downloadable from Kenneth French’s Data Library). This merged version allows for six-factor analysis within the same time window.

# Important! 
You will need to download Fama/French 5 Factors (2x3) and Momentum Factor (Mom) from Ken French's website.
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

The repository also includes a merging utility that combines the Fama–French 2×3 factor dataset with the Momentum data files (both downloadable from Kenneth French’s Data Library). This merged version allows for six-factor analysis within the same time window.

**When preparing the merged dataset, ensure you do not tick 'In percent' when downloading the CSVs from French’s website, otherwise the program will misread the inputs.** This is crucial!

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
