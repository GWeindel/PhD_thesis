# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
import pystan
import arviz as az
import matplotlib.pyplot as plt
import stan_utility
import patsy

data = pd.read_csv('../EMG_slopes.csv')

LMEdata = data[np.abs(data.slope) < data.slope.std()*2].copy()
print(LMEdata.head())
LMEdata['SAT'] = LMEdata.apply(lambda row: 0 if row['SAT'] == "speed" else 1, axis=1)
LMEdata['Validity'] = LMEdata.apply(lambda row: -0.5 if row['valid'] == "invalid" else (
	0.5 if row['valid'] == 'valid' else 0), axis=1)
LMEdata['participant'] = LMEdata.participant.replace(LMEdata.participant.unique(), np.arange(len(LMEdata.participant.unique()))+1) 
LMEdata = LMEdata[["slope","participant",
                   "SAT","Validity"]]
print(LMEdata.head())

LME = stan_utility.compile_model('LME.stan', model_name="LME")

fixeff_form = "1+SAT+Validity+SAT:Validity"#Fixed effects formula
raneff_form = fixeff_form #Random effects formula
fixeff = np.asarray(patsy.dmatrix(fixeff_form, LMEdata)) #FE design matrix
raneff = np.asarray(patsy.dmatrix(raneff_form, LMEdata)) #RE design matrix
prior_intercept = np.asarray([8e4,5e4])#prior for intercept, mu and sigma
priors_mu = np.repeat(0, 3) #Priors on mu for FE
priors_sigma =  np.repeat(5e4, 3) # priors on sigma for FE
priors_raneff = [0, 5e4] #Priors on RE
prior_sd = [0, 5e4] #priors on residual sigma

RT_LME_data = dict(
    N = len(LMEdata),
    P = fixeff.shape[-1], #number of pop level effects
    J = len(LMEdata.participant.unique()),
    n_u = raneff.shape[-1],
    subj = LMEdata.participant,
    X = fixeff,
    Z_u = raneff,
    y = LMEdata.slope.values,
    p_intercept = prior_intercept, p_sd = prior_sd, p_fmu = priors_mu, p_fsigma = priors_sigma, p_r = priors_raneff,
    logT = 0
)

RT_fit = LME.sampling(data=RT_LME_data, iter=2000, chains=6, n_jobs=6,
                      warmup = 1000,  control=dict(adapt_delta=0.99))

RT_fit = az.from_pystan(posterior=RT_fit, posterior_predictive='y_hat', observed_data="y", log_likelihood='log_lik',
                                   coords={'b': fixeff_form.split('+')[1:]}, dims={'raw_beta': ['b']})

RT_fit.to_netcdf("FittedModels/slope_fit.nc")
