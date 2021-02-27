# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:30:02 2018

@author: Laure
"""

import os
import sys
import logging
import mne 
import numpy as np
import debut as dbt

#Problem last event !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


###### set event names and codes correspondance, adapt to your data ###### 
trig_id = {'speed/left/1':111,'speed/left/2':112,'speed/left/3':113,
           'speed/right/1':121,'speed/right/2':122,'speed/right/3':123,
           'accuracy/left/1':211,'accuracy/left/2':212,'accuracy/left/3':213,
           'accuracy/right/1':221,'accuracy/right/2':222,'accuracy/right/3':223,} # used for segmentation
resp_id = {'left': 100 , 'right' : 200}
force_id = {'low': 1 , 'high' : 2}
emg_id = {'onset': 4 , 'offset': 5, 'peak': 6}
resp_list = list(resp_id.values()) 

initial_expctd_events = sum([list(trig_id.values()), list(force_id.values()), list(resp_id.values())], []) 

# EMG channels index and names 
emg_channels_idx = {0: 'EMG_L', 1: 'EMG_R'}
force_channels_idx = {2: 'Erg1', 3: 'Erg2'}

########## set data and save paths: to adapt to your environment ########## 
path_bdf = ('../../raw_data/')
path_mrk = ('processed/automatic_detection')
        
########## set file name to read: change to your file name ########## 
name_subj = input("prompt")  # Python 2
data_fname = path_bdf + name_subj + '.bdf'

########## create log file
log_name = os.path.join(path_mrk, name_subj+'.log')
dbt.set_log_file(log_name, overwrite=True)
logging.info("Automatic onsets/offsets detection:")


raw = mne.io.read_raw_bdf(data_fname, preload=True)
raw = raw.pick(picks=[ 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Erg1', 'Erg2','Status'])
mne_events = mne.find_events(raw)

if name_subj == "S3": #removes incpmleted trail for S3
	mne_events = mne_events[:-5]

mne_events[:,2] = mne_events[:,2] - mne_events[:,1]

mne_events = np.array([list(x) for x in mne_events if x[2] in initial_expctd_events])   

mne.set_bipolar_reference(raw,anode=['EXG1','EXG3'], cathode=['EXG2','EXG4'],ch_name=['EMG_L','EMG_R'],\
copy=False)
raw.pick_channels(['EMG_L','EMG_R','Erg1','Erg2'])
raw.set_channel_types({'EMG_L':'emg',
                       'EMG_R':'emg',
                       'Erg1':'misc',
                       'Erg2':'misc'})

emg_channels_idx = {0: 'EMG_L', 1: 'EMG_R'}
force_channels_idx = {2: 'Erg1', 3: 'Erg2'}
channel_names = list(emg_channels_idx.values()) + list(force_channels_idx.values())


#Filtering
raw = dbt.use_mne.apply_filter(raw, ch_names=['EMG_L','EMG_R'], low_cutoff=10)
logging.info("\tHigh pass filtering of EMG traces at 10Hz")
raw = dbt.use_mne.apply_filter(raw, ch_names=['Erg1','Erg2'], high_cutoff=40)  
logging.info("\tLow pass filtering of Force traces at 40Hz")

data_raw = raw.get_data()

#data_raw[2,:] = data_raw[2,:] - np.mean(data_raw[2, 100:1000])
#data_raw[3,:] = data_raw[3,:] - np.mean(data_raw[3, 100:1000])  

events = dbt.Events(sample=mne_events[:,0], code=mne_events[:,2], chan=[-1]*mne_events.shape[0],\
                    sf=raw.info['sfreq'])


# interval used for segmentation, in seconds
tmin = -.4
tmax = 1.2
epoch_time = dbt.times(tmin,tmax,events.sf)
t0 = dbt.find_times(0,epoch_time)
tmax_sample = dbt.find_times(tmax,epoch_time)

epochs_events = events.segment(code_t0=list(trig_id.values()), tmin=tmin, tmax=tmax)

data_epochs = epochs_events.get_data(data_raw)

thEMG_raw = 5
thEMG_tk = 5
th_force = 8

logging.info("\t\t threshold for EMG left: " + str(thEMG_raw))
logging.info("\t\t threshold for EMG right: " + str(thEMG_raw))        

logging.info("\t\t threshold for EMG left Teager-Kaiser: " + str(thEMG_tk))
logging.info("\t\t threshold for EMG right Teager-Kaiser: " + str(thEMG_tk))

logging.info("\t\t threshold for force left: " + str(th_force))
logging.info("\t\t threshold for force right: " + str(th_force))  

mbsl_raw,stbsl_raw = dbt.global_var(data_epochs, epoch_time, cor_var=2.5, use_tkeo=False)
mbsl_tk, stbsl_tk = dbt.global_var(data_epochs, epoch_time, cor_var=2.5, use_tkeo=True)

for e in range(epochs_events.nb_trials()):
    # Onset and offset EMG detection
    for c in emg_channels_idx.keys():
        
        current = data_epochs[e,c,:]
        
        #Lcal mBl and stBl are recommended, to use global values, use mbsl_raw/mbsl_tk[c] and stbsl_raw/stbs_tk[c] computed above
        onsets,offsets = dbt.get_onsets(current, epoch_time, sf=events.sf,\
                                        th_raw= thEMG_raw, use_raw=True, time_limit_raw=.015, min_samples_raw=25,\
                                        varying_min_raw=1, mbsl_raw=None, stbsl_raw=None, \
                                        th_tkeo= thEMG_tk, use_tkeo=True, time_limit_tkeo=.015,  min_samples_tkeo=25,\
                                        varying_min_tkeo=0, mbsl_tkeo=None, stbsl_tkeo=None)
        
        # Remove burst starting and ending before time 0
        onsets = [onsets[b] for b in range(len(onsets)) if (offsets[b] > t0)]
        offsets = [offsets[b] for b in range(len(offsets)) if (offsets[b] > t0)]
        # If one onset remains before t0, put its latency to time 0
        onsets = [np.max((b,t0+1)) for b in onsets]
        
        # Remove bursts starting after the first response
        stim = epochs_events.list_evts_trials[e].find_events(code=list(trig_id.values()))
        resp = epochs_events.list_evts_trials[e].find_events(code=resp_list)
        if len(resp) > 0:
            #latency of the first response after the first stimulus
            resp_latency =  epochs_events.list_evts_trials[e].lat.sample[resp[resp > stim[0]][0]]
        else: 
            resp_latency = tmax_sample # if no response, resp latency is equal to tmax
        offsets = [offsets[b] for b in range(len(offsets)) if (onsets[b] < resp_latency+200)]
        onsets = [onsets[b] for b in range(len(onsets)) if (onsets[b] < resp_latency+200)]

        # Put in event structure
        onsets_events = dbt.Events(sample=onsets, time=epoch_time[onsets],\
                                   code=[emg_id['onset']]*len(onsets), chan=[c]*len(onsets), sf=epochs_events.sf) 
        offsets_events = dbt.Events(sample=offsets, time=epoch_time[offsets],\
                                    code=[emg_id['offset']]*len(offsets), chan=[c]*len(offsets), sf=epochs_events.sf) 
        
        # Add in epochs events
        epochs_events.list_evts_trials[e].add_events(onsets_events)
        epochs_events.list_evts_trials[e].add_events(offsets_events)
        
    # Onset and offset force detection
    for c in force_channels_idx.keys():

        current_force = data_epochs[e,c,:]
        force_intervals = dbt.detector_var(current_force  - min(current_force), epoch_time, th=th_force, time_limit=.025,\
                                           sf=epochs_events.sf, min_samples=50, varying_min=0,\
                                           use_derivative=True, moving_avg_size=10,\
                                           use_derivative_onset=True,moving_avg_size_onset=1)

        onsets_force = force_intervals[:,0]
        offsets_force = force_intervals[:,1]
           
        # Remove intervals starting and ending before time 0
        onsets_force = [onsets_force[b] for b in range(len(onsets_force)) if (offsets_force[b] > t0)]
        offsets_force = [offsets_force[b] for b in range(len(offsets_force)) if (offsets_force[b] > t0)]
        # If one onset remains before t0, put its latency to time 0
        onsets_force = [np.max((b,t0+1)) for b in onsets_force]
        
        # Remove intervals starting after the first response
        offsets_force = [offsets_force[b] for b in range(len(offsets_force)) if (onsets_force[b] < resp_latency)]
        onsets_force = [onsets_force[b] for b in range(len(onsets_force)) if (onsets_force[b] < resp_latency)]

        # Put in event structure
        onsets_force_events = dbt.Events(sample=onsets_force, time=epoch_time[onsets_force], code=[emg_id['onset']]*len(onsets_force), chan=[c]*len(onsets_force), sf=epochs_events.sf) 
        offsets_force_events = dbt.Events(sample=offsets_force, time=epoch_time[offsets_force], code=[emg_id['offset']]*len(offsets_force), chan=[c]*len(offsets_force), sf=epochs_events.sf) 

        # Get force peaks
        list_signals = dbt.get_signal_portions(current_force, onsets_force_events.lat.sample, offsets_force_events.lat.sample)
        _,peak_sample = dbt.get_signal_max(list_signals)
        peak_sample += onsets_force_events.lat.sample
    
        # Put in event structure
        if len(peak_sample) > 0:
            peaks_force_events = dbt.Events(sample=peak_sample, time=epoch_time[peak_sample], code=[emg_id['peak']]*len(peak_sample) , chan=[c]*len(peak_sample), sf=epochs_events.sf)
        
            # Add in epochs events
            epochs_events.list_evts_trials[e].add_events(onsets_force_events)
            epochs_events.list_evts_trials[e].add_events(offsets_force_events)
            epochs_events.list_evts_trials[e].add_events(peaks_force_events)


continuous_events_with_detection = epochs_events.as_continuous(drop_duplic=True)[0]
events.add_events(continuous_events_with_detection, drop_duplic=True)

fname_detected_mrk = os.path.join(path_mrk, name_subj+'_detectEMG.csv')
events.to_csv(fname_detected_mrk,\
              header='sample,time,code,chan', sep=',',\
              save_sample=True, save_time=True, save_code=True, save_chan=True)

logging.info("\tEvents saved in " + fname_detected_mrk) 
