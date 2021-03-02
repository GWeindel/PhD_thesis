#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:00:51 2018

@author: gabriel
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

os.curdir = ('synRT6/')
list_dir = os.listdir(os.curdir)
list_ = []
for f in list_dir:
    raw_ = pd.read_table('synRT6/'+f, header=None, sep=' ',skipinitialspace=True
                         ,skiprows=0)
    raw_['participant'], raw_['FC'], raw_['SAT'],raw_['contrast'] = \
        f.split('_')[0], int(f.split('_')[1]),f.split('_')[2], \
        int(f.split('_')[3].split('.l')[0])
    list_.append(raw_)
synRT = pd.concat(list_, ignore_index=True)    
synRT.columns = ["response","rt","participant", "FC", "SAT", "contrast"]
synRT["rt"] = synRT["rt"]*1000

data = pd.read_csv('../../Markers/MRK_ForceEMG_correct.csv')
trim = data[(data.trialType!="UT") & (data.trialType!="NR")].copy()
trim.drop(trim[((trim.rt  < trim.rt.median()) & (trim.MADrt >2))
          | ((trim.rt  > trim.rt.median()) & (trim.MADrt >7))].index, inplace=True)
trim.drop(trim[((trim.mt  < trim.mt.median()) & (trim.MADmt >2))
          | ((trim.mt  > trim.mt.median()) & (trim.MADmt >7))].index, inplace=True)
trim.drop(trim[((trim.prmt  < trim.prmt.median()) & (trim.MADprmt >2))
          | ((trim.prmt  > trim.prmt.median()) & (trim.MADprmt >7))].index, inplace=True)
trim = trim[~trim.prmt.isna()]#23 trials with na values for PMT, why ?
trim.reset_index(inplace=True)
#trim["original_contrast"] = trim["contrast"]
#trim["contrast"] = trim.apply(lambda row: 1 if row["contrast"]== 1 or row["contrast"]== 2 else 
#                              (2 if row["contrast"]== 3 or row["contrast"]== 4 else 3) , axis=1)
trim = trim.rename(columns = {'rt':'RT', 'precision':'response'})



for FC, FC_dat in trim.groupby("FC"):
    FC_syn = synRT[synRT.FC == FC]
    for SAT, SAT_dat in FC_dat.groupby("SAT"):
        SAT_syn = FC_syn[FC_syn.SAT==SAT]
        Prec,synPrec, RTQuantiles, synQuantiles, subject, contrast, std_quantiles = [],[],[],[],[],[],[]
        meanPrec, meanRT, synmeanPrec, synmeanRT, meanstd = [],[],[],[],[]
        for con, con_dat in SAT_dat.groupby("contrast"):
            con_syn = SAT_syn[SAT_syn.contrast == con]
            for corr, corr_dat in con_dat.groupby("response"):
                meanPrec.append(float(len(corr_dat.response))/len(con_dat))
                corr_dat = corr_dat.copy()
                corr_dat["quantile"] = [(float(str(a).split(',')[0][1:]) + float(str(a).split(', ')[1][:-1]))/2 for a in pd.qcut(corr_dat.RT, 5)]
                mean_quantiles,std_quantiles = [],[]
                for quant, quant_dat in corr_dat.groupby("quantile"):
                    mean_quantiles.append(quant_dat.RT.mean())
                    std_quantiles.append(quant_dat.RT.std())
                meanRT.append(mean_quantiles)
                meanstd.append(std_quantiles)
                corr_syn = con_syn[con_syn.response == corr].copy()
                corr_syn["quantile"] = [(float(str(a).split(',')[0][1:]) + float(str(a).split(', ')[1][:-1]))/2 for a in pd.qcut(corr_syn.rt, 5)]
                synmeanPrec.append(float(len(corr_syn.response))/len(con_syn))
                mean_quantiles = []
                for quant, quant_dat in corr_syn.groupby("quantile"):
                    mean_quantiles.append(quant_dat.rt.mean())
                synmeanRT.append(mean_quantiles)
    
        QPdf = pd.DataFrame([meanRT,synmeanRT, meanPrec, synmeanPrec, contrast, meanstd]).T
        QPdf.columns=["RTQuantiles","synQuantiles","Precision","synPrecision","contrast","std_quantiles"]
        QPdf = QPdf.sort_values(by="Precision")
        color = ['#999999','#777777', '#555555','#333333','#111111']
        x = [x for x in QPdf["Precision"].values]
        y = [y for y in QPdf["RTQuantiles"].values]
        fig, ax = plt.subplots()
        for _x, _y in zip( x, y):
            n = 0
            for xp, yp in zip([_x] * len(_y), _y):
                n += 1
                ax.scatter([xp],[yp], marker=None, s = 0.0001)
                ax.text(xp-.01, yp-10, 'x', fontsize=12, color=color[n-1])#substracted values correct text offset
        plt.plot( [i for i in QPdf["synPrecision"].values],  [j for j in QPdf["synQuantiles"].values],'o-', color='gray', markerfacecolor="w", markeredgecolor="gray")
        plt.xlabel("Response proportion")
        plt.ylabel("RT quantiles (ms)")
        plt.xlim(0,1)
        plt.vlines(.5,np.min([np.min(synmeanRT),np.min(meanRT)])-50,np.max([np.max(synmeanRT),np.max(meanRT)])+100,linestyle=':')
        plt.ylim(np.min([np.min(synmeanRT),np.min(meanRT)])-50, np.max([np.max(synmeanRT),np.max(meanRT)])+100)
        plt.annotate('Correct', xy=(.9, np.max([np.max(synmeanRT),np.max(meanRT)])+50), xytext=(.505,np.max([np.max(synmeanRT),np.max(meanRT)])+50),
            arrowprops={'arrowstyle': '->'}, va='center')
        plt.annotate('Errors', xy=(.1, np.max([np.max(synmeanRT),np.max(meanRT)])+50), xytext=(.40,np.max([np.max(synmeanRT),np.max(meanRT)])+50),
            arrowprops={'arrowstyle': '->'}, va='center')
        plt.title('GoF in Force %s and in %s'%(FC,SAT))
        plt.savefig('QPplot6/%s%s.png'%(FC,SAT))
        plt.show() 
