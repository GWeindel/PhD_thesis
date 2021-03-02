# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
import pystan
import os, sys
import stan_utility
import patsy

data = pd.read_csv('../data_modelling_ready.csv')

GLMEdata = data.copy() 
print(GLMEdata.head())
GLMEdata['SAT'] = GLMEdata.apply(lambda row: 0 if row['SAT'] == "speed" else 1, axis=1)
GLMEdata['Validity'] = GLMEdata.apply(lambda row: -0.5 if row['valid'] == "invalid" else (
	0.5 if row['valid'] == 'valid' else 0), axis=1)
GLMEdata['participant'] = GLMEdata.participant.replace(GLMEdata.participant.unique(), np.arange(len(GLMEdata.participant.unique()))+1) 
GLMEdata['response'] = [int(x) for x in GLMEdata.response]
GLMEdata = GLMEdata[["response","participant",
                   "SAT","Validity"]]
print(GLMEdata.head())

GLMEdata.to_csv('GLMEdata.csv')

GLME = """
data {
  int<lower=0> N;               //N observations total
  int<lower=1> P;               //N population-level effects  (intercept + beta)
  int<lower=0> J;               //N subj
  int<lower=1> n_u;             //N subj ranefs (ex. x * x | part = 4) if random effect on all effects, same argument as P can be passed
  int<lower=1,upper=J> subj[N]; //subj ID vector
  matrix[N,P] X;                //fixef design matrix
  matrix[N, n_u] Z_u;           //subj ranef design matrix, X if raneff = fixeff
  int<lower=0,upper=1> y[N];// DV
  vector[2] p_intercept;        // prior on intercept
  vector[P-1] p_fmu;              // priors on fixef mu
  vector[P-1] p_fsigma;           // priors on fixef sigma
  vector[2] p_r;                // priors on ranef
}

transformed data {
  matrix[N,P-1] X_beta; // removing intercept
  X_beta = block(X,1,2,N,P-1); // returns X from row 1 and col 2 to N rows and P-1 columns
}

parameters {
  real<lower=0> alpha;          // intercept
  vector[P-1] beta;             //population-level effects coefs (w\o intercept)
  cholesky_factor_corr[n_u] L_u;  //cholesky factor of subj ranef corr matrix
  vector<lower=0>[n_u] sigma_u; //subj ranef std
  vector[n_u] z_u[J];           //spherical subj ranef
}

transformed parameters {
  vector[n_u] u[J];             //subj ranefs
  {
    matrix[n_u,n_u] Sigma_u;    //subj ranef cov matrix
    Sigma_u = diag_pre_multiply(sigma_u,L_u);
    for(j in 1:J)
      u[j] = Sigma_u * z_u[j];
  }
}

model {
  vector[N] theta;
  //priors
  L_u ~ lkj_corr_cholesky(2.0);
  alpha ~ normal(p_intercept[1], p_intercept[2]);
  beta ~ normal(p_fmu,p_fsigma);
  for (j in 1:J)
    z_u[j] ~ normal(p_r[1],p_r[2]);

  //likelihood
  for (n in 1:N)
    theta[n] = alpha + X_beta[n] * beta + Z_u[n] * u[subj[n]];
  y ~ bernoulli_logit(theta);
}

generated quantities {
  vector[P-1] raw_beta; //raw effect size if log values
  real raw_intercept;
  vector[N] log_lik;
  vector[N] y_hat;          // predicted y
  matrix[n_u,n_u] Cor_u;
  Cor_u = tcrossprod(L_u);  //Correlations between random effects by subj
  raw_intercept = inv_logit(alpha);
  raw_beta = inv_logit(alpha + beta) - inv_logit(alpha);
  for (n in 1:N){
    log_lik[n] = bernoulli_logit_lpmf(y[n] | alpha + X_beta[n] * beta + Z_u[n] * u[subj[n]]);
    y_hat[n] = bernoulli_rng(inv_logit(alpha + X_beta[n] * beta + Z_u[n] * u[subj[n]]));
  }
}
"""

f = open("GLME.stan","w+")
f.write(GLME)
f.close()
GLME = stan_utility.compile_model('GLME.stan', model_name="GLME")
