#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:00:51 2018

@author: gabriel
"""

import subprocess
import numpy
import pandas as pd


def simulatingRT(a,v,t0,st0, sub, FC, SAT, SS):
    print('Simulating RT for %s' %sub+str(SS))
    p = subprocess.Popen(["wine construct-samples.exe -a %f -z 0.5 -v %f -t %f\
        -Z 0 -V 0 -T %f -d 0 -n 10000 -N 1 -p 5 -o %s.lst" %(float(a), float(v), 
        float(t0), float(st0), 'synRT/'+ sub + '_' + str(int(FC)) +\
        '_' + str(SAT) + '_' + str(int(SS)))], stdout= subprocess.PIPE, 
        stderr=subprocess.STDOUT, bufsize=1, shell=True)
    m = p.communicate()
    return True

fit = pd.read_csv('DDMfiles/fit_results_RT.csv')
params = []
for xx, subj_dat in fit.groupby(['participant']): 
    zr = subj_dat.loc[subj_dat['parameter']=='zr'].value.mean()
    st0 = subj_dat.loc[subj_dat['parameter']=='st0'].value.values[0]
    for SAT, SAT_dat in subj_dat.groupby('SAT'):
        a = SAT_dat.loc[SAT_dat['parameter']=='a'].value.values[0]
        t0 = SAT_dat.loc[SAT_dat['parameter']=='t0'].value.values[0]
        for contrastXstim, c_dat in subj_dat.groupby(['contrast','stimulus']):
            v = c_dat.loc[c_dat['parameter']=='v'].value.values[0]
            simulatingRT(a,v,t0,st0, str(cell.split('.')[0]), contrastXstim[0], SAT, contrastXstim[1])
