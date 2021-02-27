#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:57:29 2020

@author: gabriel
"""

import pandas as pd
import numpy as np

df = pd.read_csv('extracted_df.csv',index_col=0)
df_after_rep = pd.read_csv('extracted_df_after_rep.csv',index_col=0)

def transform_to_array(serie):
   return [list(x.strip('[]').split()) for x in serie]

df['force_code'] = transform_to_array(df_after_rep.other_code.copy())

del df_after_rep

resp_id = {100.:'left', 200.:'right'}
force_id = {'1':'low', '2':'high'}
trig_id = {'speed/left/1':111,'speed/left/2':112,'speed/left/3':113,
           'speed/right/1':121,'speed/right/2':122,'speed/right/3':123,
           'accuracy/left/1':211,'accuracy/left/2':212,'accuracy/left/3':213,
           'accuracy/right/1':221,'accuracy/right/2':222,'accuracy/right/3':223}



#indigest and slow code to recover force triggers in 'other_codes' after response
df.force_code = df.apply(lambda row :
            int(row['force_code'][np.where(np.isin(row['force_code'],list(force_id.keys())))[0][0]])
            if np.sum(np.isin(row['force_code'],list(force_id.keys()))) > 0 else np.nan, axis=1)
missing_force_trigger = df[~np.isfinite(df.force_code)].index
df.loc[missing_force_trigger, 'force_code'] = df.loc[missing_force_trigger-1,'force_code'].values


trig_id = {v: k for k, v in trig_id.items()}

#creating trial index
df['trial'] = np.nan
for sub, sub_dat in df.groupby('participant'):
    df.loc[df.participant == sub, 'trial'] = np.arange(len(sub_dat))

#Removing trials with no response : 
df = df[np.isfinite(df.resp_code)]

#Defining PRMT : 
emg_chan = {100.:'0',200.:'1'}
df.onset_code = transform_to_array(df.onset_code)
df.onset_chan = transform_to_array(df.onset_chan)
df.onset_time = transform_to_array(df.onset_time)

df['rt'] = np.nan
df['prmt'] = np.nan
df['mt'] = np.nan
faulty_trials = 0
for trial, row in df.iterrows():
    times = [float(x) for x in row.onset_time]
    chan_emg_response = emg_chan[row.resp_code]
    try :
        if row.emg_type != "no_emg": #gets last onset (one same chanel as response)
            last_emg = times[np.where(np.isin(row.onset_chan, chan_emg_response))[0][-1]]
            df.loc[trial, 'rt'] = row.resp_time - row.stim_time
            df.loc[trial, 'prmt'] = last_emg - row.stim_time
            df.loc[trial, 'mt'] = row.resp_time - last_emg
        else:#except means no EMG
            df.loc[trial, 'rt'] = row.resp_time - row.stim_time
            df.loc[trial, 'prmt'] = np.nan
            df.loc[trial, 'mt'] = np.nan
    except : 
        df.loc[trial, 'sequence_onset'] = "R"
        row.rt = row.resp_time - row.stim_time
        print([row.participant,row.trial])
        faulty_trials += 1
# Check negative PMT

# Recovering general infos
clean_df = df[['participant','trial']].copy()
clean_df['givenResp'] = [resp_id[x] for x in df['resp_code']]
clean_df['SAT'],clean_df['expdResp'],clean_df['contrast'] = zip(*[trig_id[x].split('/') for x in df['stim_code']])
clean_df['response'] = clean_df.apply(lambda row: 1 if row['expdResp'] == row['givenResp'] else 0, axis=1)
clean_df['FC'] = df['force_code']

#Defining RT
clean_df['rt'] = df['rt']*1000
clean_df['prmt'] = df['prmt']*1000
clean_df['mt'] = df['mt']*1000

#Redifinning EMG categories :
unique_emg = ["pureC", 'pureI']
no_emg = ["no_emg", 'unclassified']
clean_df['trialType'] = df.apply(lambda row : "SA" if len(row['sequence_onset']) == 2 else
                                 ("MA" if len(row['sequence_onset']) > 2 else 'UT'), axis=1)
clean_df['EMG_sequence'] = df.sequence_onset

clean_df.to_csv('clean_df.csv')

