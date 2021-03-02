# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
import pystan
import arviz as az
import matplotlib.pyplot as plt
import stan_utility
import patsy

LMEdata = pd.read_csv('LMEdata.csv')
print(LMEdata.head())
LME = stan_utility.compile_model('LME.stan', model_name="LME")

fixeff_form = "1+SAT+Validity+Incorrect+SAT:Validity+SAT:Incorrect+Validity:Incorrect+SAT:Validity:Incorrect"#Fixed effects formula
raneff_form = fixeff_form #Random effects formula
fixeff = np.asarray(patsy.dmatrix(fixeff_form, LMEdata)) #FE design matrix
raneff = np.asarray(patsy.dmatrix(raneff_form, LMEdata)) #RE design matrix
prior_intercept = np.asarray([5.85,.3])#prior for intercept, mu and sigma
priors_mu = np.repeat(0, 7) #Priors on mu for FE
priors_sigma =  np.repeat(.4, 7) # priors on sigma for FE
priors_raneff = [0, .4] #Priors on RE
prior_sd = [0, .4] #priors on residual sigma

PMT_LME_data = dict(
    N = len(LMEdata),
    P = fixeff.shape[-1], #number of pop level effects
    J = len(LMEdata.participant.unique()),
    n_u = raneff.shape[-1],
    subj = LMEdata.participant,
    X = fixeff,
    Z_u = raneff,
    y = LMEdata.logprmt.values,
    p_intercept = prior_intercept, p_sd = prior_sd, p_fmu = priors_mu, p_fsigma = priors_sigma, p_r = priors_raneff,
    logT = 1
)

PMT_fit = LME.sampling(data=PMT_LME_data, iter=2000, chains=6, n_jobs=6,
                      warmup = 1000,  control=dict(adapt_delta=0.99))

PMT_fit = az.from_pystan(posterior=PMT_fit, posterior_predictive='y_hat', observed_data="y", log_likelihood='log_lik',
                                   coords={'b': fixeff_form.split('+')[1:]}, dims={'raw_beta': ['b']})

PMT_fit.to_netcdf("FittedModels/PMT_fit.nc")