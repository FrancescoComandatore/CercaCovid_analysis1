# CercaCovid_analysis1
Python and R commands used in the paper "A mobile app for a population-wide screening of COVID-19 cases in Lombardy"

INPUT:

Three input files are required to perfom the analyses:

1. Data from CercaCovid app:

data are available on request ...

2. Data of SARS-Cov-2 positive swabs:

you can find the data used for the paper in the file dpc-covid19-ita-regioni.csv.
Updated data are also avaiable at https://github.com/pcm-dpc/COVID-19/tree/master/dati-regioni

3. Data about Covid-19 mortality in Italy.

Data are reported in the ISTAT report (https://www.istat.it/it/files//2020/05/Rapporto_Istat_ISS.pdf)
were manually extracted and organized into the file Mortality_ISTAT.tab 

RUNNING:

You have to run the Python script for first. It takes as input the CercaCovid data and produces a list of plots and a table 
with a Covid score for each questionnaire. The CercaCovid data and the Covid score table are the input of the
Rcommands.R script.



