import pandas as pd
import numpy as np
import glob, os

data = pd.read_csv("../../../DecomposingRT/Raw_data/markers/MRK_SAT.csv")
trim = data[(data.trialType!="UT") & (data.trialType!="NR")].copy()
pretrim = len(trim.reset_index().index)
trim.drop(trim[((trim.rt  < trim.rt.median()) & (trim.MADrt >2))
          | ((trim.rt  > trim.rt.median()) & (trim.MADrt >7))].index, inplace=True)
trim.drop(trim[((trim.mt  < trim.mt.median()) & (trim.MADmt >2))
          | ((trim.mt  > trim.mt.median()) & (trim.MADmt >7))].index, inplace=True)
trim.drop(trim[((trim.pmt  < trim.pmt.median()) & (trim.MADpmt >2))
          | ((trim.pmt  > trim.pmt.median()) & (trim.MADpmt >7))].index, inplace=True)
trim = trim[~trim.pmt.isna()]#23 trials with na values for PMT, why ?
trim.reset_index(inplace=True)
trim = trim.rename(columns = {'condition':'SAT'})


print((len(data.index)-pretrim)/len(data.index))
print((len(data.index)-len(trim.index))/len(data.index))
print(((len(data.index)-pretrim)/len(data.index))-(len(data.index)-len(trim.index))/len(data.index))
del data 


Fastdm_data = trim.copy()
Fastdm_data['exp'] = Fastdm_data.apply(lambda row: 'one' if row['exp'] == 1 else 'two', axis=1)
Fastdm_data = Fastdm_data.rename(columns = {'rt':'RT', 'precision':'response', 'condition':'SAT'})
Fastdm_data = Fastdm_data[(Fastdm_data.trialType != 'NR') & (Fastdm_data.trialType != 'UT')].reset_index(drop=True)#Removing unmarked EMG trials
Fastdm_data.response = Fastdm_data.response.astype(int)
Fastdm_data['RT'] = Fastdm_data['RT'] / 1000

import glob,os
precision = 5
method = 'ks'#You can switch with ml, cs or ks

Free = {
    'v': ['SAT',"contraste"],
    't0':['SAT',"contraste"],
    'a':['SAT']}

Fixed = {'zr':0.5,
         'szr':0, 
         'sv':0,
         'p':0,
         'd':0}
import FDM_functions_GR as fdm
fdm.generate_files(Fastdm_data, free = Free, fixed = Fixed, method=method, precision=precision)

fdm.fit(nproc=30)

pars = fdm.get_parameters()

pars.to_csv('fit_results_Weindel2018.csv')

for f in glob.glob("*.ctl"):
    os.remove(f)
    
for f in glob.glob("*.lst"):
    os.remove(f)
    
for f in glob.glob("data_*.csv"):
    os.remove(f)
