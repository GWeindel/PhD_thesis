import pandas as pd
import numpy as np
import glob, os

data = pd.read_csv("../../Markers/MRK_ForceEMG_correct.csv")
trim = data[(data.trialType!="UT") & (data.trialType!="NR")].copy()
pretrim = len(trim.reset_index().index)
trim.drop(trim[((trim.rt  < trim.rt.median()) & (trim.MADrt >2))
          | ((trim.rt  > trim.rt.median()) & (trim.MADrt >7))].index, inplace=True)
trim.drop(trim[((trim.mt  < trim.mt.median()) & (trim.MADmt >2))
          | ((trim.mt  > trim.mt.median()) & (trim.MADmt >7))].index, inplace=True)
trim.drop(trim[((trim.prmt  < trim.prmt.median()) & (trim.MADprmt >2))
          | ((trim.prmt  > trim.prmt.median()) & (trim.MADprmt >7))].index, inplace=True)
trim = trim[~trim.prmt.isna()]
trim.reset_index(inplace=True)
trim["original_contrast"] = trim["contrast"]
#trim["contrast"] = trim.apply(lambda row: 1 if row["contrast"]== 1 or row["contrast"]== 2 else 
#                              (2 if row["contrast"]== 3 or row["contrast"]== 4 else 3) , axis=1)

print((len(data.index)-pretrim)/len(data.index))
print((len(data.index)-len(trim.index))/len(data.index))
print(((len(data.index)-pretrim)/len(data.index))-(len(data.index)-len(trim.index))/len(data.index))
del data 

Fastdm_data = trim.copy()
Fastdm_data['FC'] = Fastdm_data.apply(lambda row: 'one' if row['FC'] == 1 else 'two', axis=1)
Fastdm_data = Fastdm_data.rename(columns = {'rt':'RT'})
Fastdm_data = Fastdm_data[(Fastdm_data.trialType != 'NR') & (Fastdm_data.trialType != 'UT')].reset_index(drop=True)
Fastdm_data.contrast = Fastdm_data.contrast.astype(str)
Fastdm_data.response = Fastdm_data.response.astype(int)
Fastdm_data = Fastdm_data[np.isfinite(Fastdm_data.prmt)]
Fastdm_data['RT'] = Fastdm_data['RT'] / 1000

import glob,os
precision = 5
method = 'ks'#You can switch with ml, cs or ks

Free = {
    'v': ['SAT','FC',"contrast"],
    't0':['SAT','FC',"contrast"],
    'a':['SAT','FC']}

Fixed = {'zr':0.5,
         'szr':0, 
         'sv':0,
         'p':0,
         'd':0}
import FDM_functions_FC as fdm
fdm.generate_files(Fastdm_data, free = Free, fixed = Fixed, method=method, precision=precision)

fdm.fit(nproc=16)

pars = fdm.get_parameters()

pars.to_csv('fit_results_6con.csv')

for f in glob.glob("*.ctl"):
    os.remove(f)
    
for f in glob.glob("*.lst"):
    os.remove(f)
    
for f in glob.glob("data_*.csv"):
    os.remove(f)
