# Fama–French Factor Graphs

Fama–French Factor Graphs is a Python-based analytical tool for visualising factor model regressions using the Fama–French framework.

The program enables users to plot and compare exposures to the standard Fama–French 2×3 factors (Market, SMB, HML, RMW, CMA) alongside the momentum factor, producing publication-quality graphs for research or reporting.

The repository also includes a merging utility that combines the Fama–French 2×3 factor dataset with the Momentum data files (both downloadable from Kenneth French’s Data Library). This merged version allows for six-factor analysis within the same time window.

https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

Important: when preparing the merged dataset, ensure you do not tick 'In percent' when downloading the CSVs from French’s website, otherwise the program will misread the inputs.

A standalone Windows .exe is provided under Releases so the tool can be run without a Python installation.
