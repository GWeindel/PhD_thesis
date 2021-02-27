# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:37:43 2020

@author: Administrateur
"""

# A résoudre pourquoi sur event il y a plus de SA que sur continu ?N?

import os
import debut as dbt
import pandas as pd
import eventsdf as evt

path_mrk = "processed/corrected/"
list_dir = os.listdir(path_mrk)

sf=1024 

###### set event names and codes correspondance, adapt to your data ###### 
stim_id = {'speed/left/1':111,'speed/left/2':112,'speed/left/3':113,
           'speed/right/1':121,'speed/right/2':122,'speed/right/3':123,
           'accuracy/left/1':211,'accuracy/left/2':212,'accuracy/left/3':213,
           'accuracy/right/1':221,'accuracy/right/2':222,'speed/accuracy/3':223,} # used for segmentation

resp_id = {'left': 100 , 'right' : 200}
emg_id = {'onset': 4 , 'offset': 5, 'peak': 6 }
force_id = {'onset': 4 , 'offset': 5, 'peak': 6 }

stim_resp = {111 : 100, 112 : 100 , 113 : 100,\
             211 : 100, 212 : 100 , 213 : 100,\
             121 : 200, 122 : 200 , 123 : 200,\
             221 : 200, 222 : 200 , 223 : 200} # this is the stim-resp mapping
resp_emg_chan = {100 : 0 , 200 : 1 } # mapping between response and emg (i.e., channel 0 coresponds to left , so associated with left response)
resp_force_chan = {100 : 2 , 200 : 3}

big_df = pd.DataFrame()
for f in list_dir:
    
    print('Reading file {}'.format(f))
    subj_name = f.split('_')[0]
    subj_events = dbt.load_continuous(os.path.join(path_mrk,f), sep=',', sf=sf, col_sample=0, col_code=2, col_chan=3)

#    # rename force events to avoid problems
#    for k in emg_id.keys():
#        subj_events.code[subj_events.find_events(code=emg_id[k],chan=list(resp_force_chan.values()))] = force_id[k]
        
    # create pandas DataFrame with one trial per row for push 1
    subj_df = evt.events_to_df(subj_events, list(stim_id.values()), list(stim_id.values()), list(resp_id.values()),\
                                     onset_codes=emg_id['onset'], offset_codes=emg_id['offset'],stop_after_response=True)
    
    subj_df = evt.decode_accuracy(subj_df, stim_resp, resp_emg_chan)
    subj_df = evt.classify_emg(subj_df)
    
# =============================================================================
#     Example pour la force
# =============================================================================
#     # decodes response and EMG event accuracy 
#    subjDf_emg = evt.decodeAccuracy(subjDf, stim_resp, resp_EMGchan)
#    # classify EMG trial
##        subjDf_emg = evt.classifyEMG(subjDf_emg, useOffset=True)
#    subjDf_emg = evt.classifyEMG(subjDf_emg, useOffset=False)
#
#    # decodes response and force event accuracy 
#    subjDf_force = evt.decodeAccuracy(subjDf, stim_resp, resp_forcechan)
#    # rename force columns to combine everything
#    subjDf_force = pd.DataFrame({'onset_force_correct': subjDf_force['onset_correct'],\
#                                 'offset_force_correct': subjDf_force['offset_correct'],\
#                                 'onset_force_incorrect': subjDf_force['onset_incorrect'],\
#                                 'offset_force_incorrect': subjDf_force['offset_incorrect'],\
#                                 })
    
    # add subject name col
    subj_df['participant'] = subj_name

    big_df = big_df.append(subj_df, ignore_index=True, sort=False)

big_df.to_csv('extracted_df.csv')    
          







