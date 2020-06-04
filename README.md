# CercaCovid_analysis1
Python and R commands used in the paper "A mobile app for a population-wide screening of COVID-19 cases in Lombardy"

INPUT:

Three input files are required to perfom the analyses:

1. Data from CercaCovid app:

data are available on request

2. Data about SARS-Cov-2 positive swabs:

you can find the data used for the paper in the file dpc-covid19-ita-regioni.csv.
Updated data are also avaiable at https://github.com/pcm-dpc/COVID-19/tree/master/dati-regioni

3. Data about Covid-19 mortality in Italy.

Data are reported in the ISTAT report (https://www.istat.it/it/files//2020/05/Rapporto_Istat_ISS.pdf)
were manually extracted and organized into the file Mortality_ISTAT.tab 

## Contents

This folder includes the following objects:

- `README.md`: this file, in markdown format.
- `Create_plots_and_images.py`: main Python script.
- `Rcommands.R`: R script for generating additional figures.
- `plot_utils.py`: Python module for plot utilities.
- `Shp.zip`: zipped folder containing shape files at two different levels related to Regione Lombardia.
- `dpc-covid19-ita-regioni.csv`: data about SARS-Cov-2 positive swabs over time in Italy (see above)
- `file Mortality_ISTAT.tab`: motality data in Italy extracted from the ISTAT report (see above)

## Requirements

Before running the scripts make sure that:

- The CSV input file `CERCACOVID_questionari_clean_2020-05-04.csv` is in the same directory as the script.
- All the required packages are installed (see following sections)

### Python dependencies

The results and figure of the paper have been generated with Python 3.7.4 and the following packages (versions used are in brackets):

- pandas (v. 1.0.1)
- numpy (v. 1.18.1)
- seaborn (v. 0.10.0)
- matplotlib (v. 3.2.0)
- plotly (v. 4.5.0)
- geopandas (v. 0.7.0)
- shapely (v. 1.7.0)

### R dependencies

ff (v. 2.2.14.2)
ggplot2 (v. 3.3.0)
ggpubr (v. 0.2.3)
ggrepel (v. 0.8.1)

## Instructions

To reproduce all results and figures of the paper, from a terminal:

1. Run the Python script `Create_plots_and_images.py`. The script will generate:
    - A CSV file `CERCACOVID_questionari_score.csv` containing the score associated to each user.
    - The images `Extended Data Figure 3.png`, `Extended Data Figure 5.png`, `Extended Data Figure 6a.png`, `Extended Data Figure 6b.png`.
    - The Excel files `Extended Data Figure 4_table.xlsx` and file `Numerical_values.xlsx`.
2. Run the R script `Rcommands.R`. The script will generate:
    - The images `Extended_Data_Figure1.pdf`,`Extended_Data_Figure2.pdf`,`Extended_Data_Figure7.pdf`,`Extended_Data_Figure8.pdf`,`Extended_Data_Figure9.pdf`,`Figure1.pdf`,`Figure2.pdf`,`Figure3.pdf`


