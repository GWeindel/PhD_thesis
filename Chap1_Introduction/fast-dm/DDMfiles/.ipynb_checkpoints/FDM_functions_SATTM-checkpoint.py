#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 18:10:44 2017

@author: gweindel
"""

import os
import glob
import subprocess
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
plt.ioff()

def generate_files(dataset, free = {}, fixed = {}, method=None, precision=3):
    cols = ['RT','response','stimulus']
    for f in glob.glob("*.lst"):
        os.remove(f)
    for f in glob.glob("*.ctl"):
        os.remove(f)
    for f in glob.glob("data_*"):
        os.remove(f)
    for x in dataset.participant.unique():
#==============================================================================
#Writing config file
#==============================================================================
        cfg_file = open('experiment_%s.ctl'%x, 'w')
        cfg_file.write('method %s\n' %method)
        cfg_file.write('precision %i\n' %precision)
        for key in fixed:
            cfg_file.write('set %s %s\n' %(key, fixed[key]))
        for key in free:
            if len(free[key]) == 2:
                list_= []
                for factor in free[key]:
                    list_.append('%s' %factor)
                cfg_file.write('depends %s %s %s\n' %(key, list_[0], list_[1]))
            elif len(free[key]) == 3:# FIXME avoid declaring more than three factor
                list_= []
                for factor in free[key]:
                    list_.append('%s' %factor)
                cfg_file.write('depends %s %s %s %s\n' %(key, list_[0], list_[1], list_[2]))
            else :
                cfg_file.write('depends %s %s\n' %(key, free[key][0]))
        cfg_file.write('format TIME RESPONSE stimulus\n')
        cfg_file.write('load data_%s.csv\n'%x)
        cfg_file.write('log parameter_%s.lst\n'%x)
        cfg_file.close()
#==============================================================================
#Writing data file
#==============================================================================
        temp = dataset
        temp = temp.loc[0:,cols]
        temp.to_csv('data_%s.csv'%x, index=False, \
        header=False, columns=cols, sep='\t')

def fit(nproc=6):#nproc=4):
        # Remove current datafiles
        pool = mp.Pool(nproc)
        f = []
        for file in glob.glob("*.ctl"):
            f.append(file)
        pool.map(runFDM, f)

def runFDM(config_file):
    print('Proceeding %s' %config_file)
    p = subprocess.Popen(["wine fast-dm.exe %s" %config_file], stdout= \
    subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, shell=True)
    m = p.communicate()
    return True

def runCDF(y, sub,z,k):
    print('Creating CDF for %s' %sub)
    list_dir = os.listdir(os.getcwd())
    for f in list_dir:
        if str(sub+z+str(k)) in f:
            os.remove(f)
    a = subprocess.Popen(["wine plot-cdf.exe -a %f -z %f -v %f -t %f\
    -Z %f -V %f -T %f -d %f -o %s.lst" %(y[0],y[1],y[2],y[3],y[4],
    y[5],y[6],y[7],'cdf_'+ str(sub+z+str(k)))], stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,bufsize=1, shell=True)
    b = a.communicate()
    print("wine plot-cdf.exe -a %f -z %f -v %f -t %f\
    -Z %f -V %f -T %f -d %f -o %s.lst" %(y[0],y[1],y[2],y[3],y[4],
    y[5],y[6],y[7],'cdf_'+str(sub+z+str(k))))
    return True

def get_parameters():
    list_dir = os.listdir(os.getcwd())
    list_=[]
    for f in list_dir:
        i = f.split('_')[0]#get label
        if i == "parameter":#only file with parameter* taken
            par = pd.read_csv(f,header=0,delim_whitespace=True)
            par = par.T.reset_index()
            participant = par[0][0]
            fit = par[0].iloc[-3]
            par = par[1:-4]
            par = par.reset_index()
            par.loc[par['index'].str.contains('1'), 'stimulus'] = 1
            par.loc[par['index'].str.contains('2'), 'stimulus'] = 2
            par.loc[par['index'].str.contains('3'), 'stimulus'] = 3
            par.loc[par['index'].str.contains('4'), 'stimulus'] = 4
            par.loc[par['index'].str.contains('5'), 'stimulus'] = 5
            par.loc[par['index'].str.contains('6'), 'stimulus'] = 6
            parameter = []
            for i in par['index'].str.split('_', expand=False).tolist():
                parameter.append(i[0])
            par['parameter'] = pd.Series(parameter)
            del par['index']
            par = par.rename(columns={0:'value'})
            par['participant'] = participant
            par['fit'] = fit
            list_.append(par)
    pars = pd.concat(list_, ignore_index=True)
    return pars


def draw_indiv_CDF(sub, data, pars, free, fixed):
    #for xx in pars.participant.unique():
    SAT = pars.SAT.unique()
    contraste = pars.contraste.unique()
    fig, axarr = plt.subplots(2,1)
    axdict = {'Speed':axarr[0],'Accuracy':axarr[1]}
    part = pars[pars.participant==sub]
    tmp1 = part[part.parameters.isin(free)==False]
    fix = pd.DataFrame.from_dict(fixed, orient='index').reset_index()
    fix = fix.rename(columns={'index':'parameters', 0:'value'})
    fix['SAT'], fix['contraste'] = np.nan, np.nan
    fix['participant'] = sub
    for z in SAT:
        for k in contraste:
            tmp2 = part[part.SAT == z]
            tmp3 = part[part.contraste == k]
            par = pd.concat([tmp1,tmp2,tmp3])
            par['ranked_parameters'] = par['parameters'].map(custom_dict)
            par = par.sort_values(by='ranked_parameters')
            y = np.asarray(par.value)
            print(y, sub, z, k)
            runCDF(y, sub, z, k)
            eCDF = data[(data.contraste == k) & (data.participant==sub) & (data.SAT == z)]
            eCDF['CRT'] = eCDF.apply(lambda row: -row['RT'] if row['response'] == 0 else row['RT'], axis=1)
            eCDF['CRT'] = np.sort(eCDF['CRT'])
            eCDF = eCDF.reset_index()
            eCDF['order'] = [float(x+1)/432 for x in eCDF.index]
            pCDF = pd.read_csv('%s.lst' %str('cdf_'+sub+z+str(k)), sep=' ', header=None)
            axdict[z].plot(eCDF['CRT'],eCDF['order'], '.', color=color[str(k)])
            axdict[z].plot(pCDF[0], pCDF[1], '-', color=color[k], alpha=0.5)
            axdict[z].set_xlim(-1.5,1.5)
            axdict[z].set_ylim(0,1)
            axdict[z].set_title(z)
        fig.suptitle(sub + '\n' + '%s' %free)
        fig.show()


reversDict = {'Accuracy':'Speed', 'Speed':'Accuracy'}
def draw_CDF(dirname, data,pars,free, fixed):
    SAT = pars.SAT.dropna().unique()
    contraste = pars.contraste.dropna().unique()
    for xx in pars.participant.unique():
        fig, axarr = plt.subplots(2,1)
        axdict = {'Speed':axarr[0],'Accuracy':axarr[1]}
        part = pars[pars.participant==xx]
        fix = pd.DataFrame.from_dict(fixed, orient='index').reset_index()
        fix = fix.rename(columns={'index':'parameter', 0:'value'})
        fix['SAT'] = np.nan
        fix['contraste'] = np.nan
        fix['participant'] = xx
        common = part[part.parameter.isin(free)==False]
        for z in SAT:
            for k in contraste:
                tmpK = part[(part.contraste == k) & (part.SAT != reversDict[z])]
                k = int(k)
                tmpZ = part[(part.SAT==z) & part.contraste.isnull()]
                par = pd.concat([common,tmpZ,tmpK, fix])
                par['ranked_parameter'] = par['parameter'].map(custom_dict)
                par = par.sort_values(by='ranked_parameter')
                y = np.asarray(par.value)
                runCDF(y, xx, z, k)
                eCDF = data[(data.contraste == str(k)) & (data.participant==xx) & (data.SAT == z)]
                eCDF['CRT'] = eCDF.apply(lambda row: -row['RT'] if row['response'] == 0 else row['RT'], axis=1)
                eCDF['CRT'] = np.sort(eCDF['CRT'])
                eCDF = eCDF.reset_index(drop=True)
                eCDF['order'] = [float(x+1)/len(eCDF) for x in eCDF.index]
                pCDF = pd.read_csv('%s.lst' %str('cdf_'+xx+z+str(k)), sep=' ', header=None)
                axdict[z].plot(eCDF['CRT'],eCDF['order'], '.', color=color[str(k)])
                axdict[z].plot(pCDF[0], pCDF[1], '-', color=color[str(k)], alpha=0.5)
                axdict[z].set_xlim(eCDF.CRT.min(),eCDF.CRT.max())
                axdict[z].set_ylim(0,1)
                axdict[z].set_title(z)
        fig.savefig(os.getcwd()+'/%s/plots/%s'%(dirname, xx))
