# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
import pystan
import os, sys
import stan_utility
import patsy
import arviz as az

GLMEdata = pd.read_csv('GLMEdata.csv')

GLME = stan_utility.compile_model('GLME.stan', model_name="GLME")


fixeff_form = "1+SAT+FC+contrast+SAT:FC+SAT:contrast+FC:contrast+SAT:FC:contrast"#Fixed effects formula
raneff_form = fixeff_form #Random effects formula
fixeff = np.asarray(patsy.dmatrix(fixeff_form, GLMEdata)) #FE design matrix
raneff = np.asarray(patsy.dmatrix(raneff_form, GLMEdata)) #RE design matrix
prior_intercept = np.asarray([1,1])
priors_mu = np.repeat(0,7) #Priors on mu for FE
priors_sigma =  np.repeat(.5,7) # priors on sigma for FE
priors_raneff = [0,.5] #Priors on RE

Precision_GLME_data = dict(
    N = len(GLMEdata),
    P = fixeff.shape[-1], #number of pop level effects
    J = len(GLMEdata.participant.unique()),
    n_u = raneff.shape[-1],
    subj = GLMEdata.participant,
    X = fixeff,
    Z_u = raneff,
    y = np.asarray(GLMEdata.response.values),
    p_intercept = prior_intercept,
    p_fmu = priors_mu, p_fsigma = priors_sigma, p_r = priors_raneff
)

Precision_fit = GLME.sampling(data=Precision_GLME_data, iter=2000, chains=6, n_jobs=6,
                      warmup = 1000,  control=dict(adapt_delta=0.99))
stan_utility.check_treedepth(Precision_fit)
stan_utility.check_energy(Precision_fit)
stan_utility.check_div(Precision_fit)

Precision_fit = az.from_pystan(posterior=Precision_fit, posterior_predictive='y_hat', observed_data="y", log_likelihood='log_lik',
                                   coords={'b': fixeff_form.split('+')[1:]}, dims={'raw_beta': ['b']})

Precision_fit.to_netcdf("FittedModels/Precision_fit.nc")
