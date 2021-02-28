This repo contains the code to reproduce the analysis in the chapter 4 of the thesis. Note that all files > 50 mb were removed (DDM traces csv files) but are accessible upon request.

DDM/ folder contains summary stats for the best fitting model as well as code to fit the DDMs using HDDM python package, assess convergence and compute the BPIC and DIC scores.

MixedModels/ folder contains the code used to preprocess the data, generate the .stan (LMEdata_model_init.py and GLMEdata_model_init.py) file and to fit the GLMMs on RT, PMT, MT and proportion correct (GLMEfit.py).

Notebooks numbered from 1 to 3 contains the code used to generate and plot the results.