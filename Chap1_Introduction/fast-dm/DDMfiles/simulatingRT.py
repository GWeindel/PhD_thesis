#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:00:51 2018

@author: gabriel
"""

import subprocess
import pandas as pd


def simulatingRT(a,v,t0,st0, sub, FC, SAT, SS):
    print('Simulating RT for %s' %sub+str(SS))
    p = subprocess.Popen(["wine construct-samples.exe -a %f -z 0.5 -v %f -t %f\
        -Z 0 -V 0 -T %f -d 0 -n 10000 -N 1 -p 5 -o %s.lst" %(float(a), float(v), 
        float(t0), float(st0), 'synRT6/'+ sub + '_' + str(int(FC)) +\
        '_' + str(SAT) + '_' + str(int(SS)))], stdout= subprocess.PIPE, 
        stderr=subprocess.STDOUT, bufsize=1, shell=True)
    m = p.communicate()
    return True

fit_RT = pd.read_csv('fit_results_6con.csv')


for cell, cell_dat in fit_RT.groupby(["participant"]):
    st0 = cell_dat[cell_dat.parameter =="st0"].value.values[0]
    for FC, FC_dat in cell_dat.groupby(['FC']):
        for SAT, SAT_dat  in FC_dat.groupby(["SAT"]):
            a = SAT_dat[SAT_dat.parameter =="a"].value.values[0]
            for SS, con_dat in SAT_dat.groupby("contrast"):
                print(SS)
                t0 = con_dat[con_dat.parameter =="t0"].value.values[0]
                v = con_dat[con_dat.parameter =="v"].value.values[0]
                simulatingRT(a,v,t0,st0, str(cell.split('.')[0]), FC, SAT, SS)
