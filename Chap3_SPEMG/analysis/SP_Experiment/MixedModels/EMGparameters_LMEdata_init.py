# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
import pystan
import stan_utility

data = pd.read_csv('../EMG_parameters.csv')

LMEdata = data.copy()
print(LMEdata.head())
LMEdata['SAT'] = LMEdata.apply(lambda row: 0 if row['SAT'] == "speed" else 1, axis=1)
LMEdata['Validity'] = LMEdata.apply(lambda row: -0.5 if row['valid'] == "invalid" else (
	0.5 if row['valid'] == 'valid' else 0), axis=1)
LMEdata['participant'] = LMEdata.participant.replace(LMEdata.participant.unique(), np.arange(len(LMEdata.participant.unique()))+1) 
LMEdata = LMEdata[["slope","baseline","participant",
                   "SAT","Validity"]]
print(LMEdata.head())

LMEdata.to_csv('EMGpars_LMEdata.csv')
