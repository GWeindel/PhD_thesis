
import os
import logging
import sys
import numpy as np
import mne
import debut as dbt

path_bdf = ('../../raw_data/')
path_mrk = ('processed/automatic_detection')
path_corrected_mrk = ('processed/corrected')
        
        
trig_id = {'speed/left/1':111,'speed/left/2':112,'speed/left/3':113,
           'speed/right/1':121,'speed/right/2':122,'speed/right/3':123,
           'accuracy/left/1':211,'accuracy/left/2':212,'accuracy/left/3':213,
           'accuracy/right/1':221,'accuracy/right/2':222,'accuracy/right/3':223,} # used for segmentation
resp_id = {'left': 100 , 'right' : 200}
emg_id = {'onset': 4 , 'offset': 5, 'peak': 6}
resp_list = list(resp_id.values()) 

name_subj = input("prompt")  # Python 2
data_fname = path_bdf + name_subj + '.bdf'

log_name = os.path.join(path_mrk, name_subj + '.log')
dbt.set_log_file(log_name,overwrite=False)
logging.info("Correction of automatic onsets/offsets detection:")

raw = mne.io.read_raw_bdf(data_fname, preload=True)

mne_events = mne.find_events(raw)
if name_subj == "S3": #removes incpmleted trail for S3
	mne_events = mne_events[:-5]

mne_events[:,2] = mne_events[:,2] - mne_events[:,1]

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
logging.info("\tLow pass filtering of force traces at 40Hz")

data_raw = raw.get_data()

data_raw[2,:] = data_raw[2,:] - np.mean(data_raw[2, 100:1000])
data_raw[3,:] = data_raw[3,:] - np.mean(data_raw[3, 100:1000])  

fname_mrk = os.path.join(path_mrk, name_subj+'_detectEMG.csv') 
events = dbt.load_continuous(fname_mrk,\
                             col_sample=0, col_code=2, col_chan=3,\
                             sf=raw.info['sfreq'])
logging.info("\tLoad events from " + fname_mrk)

tmin = -.4
tmax = 2.5
epoch_time = dbt.times(tmin,tmax,events.sf)
t0 = dbt.find_times(0,epoch_time)
tmax_sample = dbt.find_times(tmax,epoch_time)

epochs_events = events.segment(code_t0=list(trig_id.values()), tmin=tmin, tmax=tmax)
data_epochs = epochs_events.get_data(data_raw)

for t in range(epochs_events.nb_trials()):
    
    # search second stim
    stim = epochs_events.list_evts_trials[t].find_events(code = list(trig_id.values()))
    
    # if more than one stim, remove everythin after second one
    if len(stim) > 1:
        events_next_trial = range(stim[1],epochs_events.list_evts_trials[t].nb_events())
        epochs_events.list_evts_trials[t].del_events(events_next_trial, print_del_evt=False)
    epochs_events.list_evts_trials[t]
    
viz = dbt.Viz(sys.argv)
    
    
viz.load_data(data_epochs, epoch_time, epochs_events,\
              code_movable_1=emg_id['onset'], code_movable_2=emg_id['offset'],\
              sync_chan=[[0,1],[0,1],[2,3],[2,3]],random_order=False)
              
viz.show()

corrected_epochs_events = viz.get_events()

unbalanced_cases = []
for t in range(data_epochs.shape[0]):
    for c in range(data_epochs.shape[1]):
        if len(corrected_epochs_events.list_evts_trials[t].find_events(code=emg_id['onset'], chan=c))\
           != len(corrected_epochs_events.list_evts_trials[t].find_events(code=emg_id['offset'], chan=c)):
            unbalanced_cases.append([t,c])

if len(unbalanced_cases) > 0:
    for t,c in unbalanced_cases:
        print(('Trial {} on channel {} does not have same number of onset and offset, please correct!').format(t,c))
    viz.show()
    
corrected_epochs_events = viz.get_events()

############ Double check
unbalanced_cases = []
for t in range(data_epochs.shape[0]):
    for c in range(data_epochs.shape[1]):
        if len(corrected_epochs_events.list_evts_trials[t].find_events(code=emg_id['onset'], chan=c))\
           != len(corrected_epochs_events.list_evts_trials[t].find_events(code=emg_id['offset'], chan=c)):
            unbalanced_cases.append([t,c])

if len(unbalanced_cases) > 0:
    for t,c in unbalanced_cases:
        print(('Trial {} on channel {} does not have same number of onset and offset, please correct!').format(t,c))
    viz.show()

corrected_epochs_events = viz.get_events()


for t in range(corrected_epochs_events.nb_trials()):
    
    # remove old peaks
    corrected_epochs_events.list_evts_trials[t].find_and_del_events(code=emg_id['peak'],print_del_evt=False)
    
    for c in force_channels_idx.keys():
    
        on_force = corrected_epochs_events.list_evts_trials[t].find_and_get_events(code=emg_id['onset'],\
                                                                                   chan=c, print_find_evt=False)
        off_force = corrected_epochs_events.list_evts_trials[t].find_and_get_events(code=emg_id['offset'],\
                                                                                    chan=c, print_find_evt=False)
        
        list_signals = dbt.get_signal_portions(data_epochs[t,c,:], on_force.lat.sample, off_force.lat.sample)
    
        m, peak_sample = dbt.get_signal_max(list_signals)
        peak_sample += on_force.lat.sample
        
        if len(peak_sample) > 0:
            peak_events = dbt.Events(sample=peak_sample,time=epoch_time[peak_sample],\
                                     code=[emg_id['peak']]*len(peak_sample), chan=[c]*len(peak_sample),\
                                     sf=corrected_epochs_events.sf)

            corrected_epochs_events.list_evts_trials[t].add_events(peak_events)


corrected_epochs_events.to_csv(os.path.join(path_corrected_mrk, name_subj +'_corrected_evts_segmented.csv'),\
                               header="sample,time,code,chan,trial_idx,trial_raw_tmin.sample,trial_raw_tmin.time,trial_raw_t0.sample,trial_raw_t0.time,trial_raw_tmax.sample,trial_raw_tmax.time",\
                               sep=',', save_sample=True, save_time=True, save_code=True, save_chan=True, save_trial_idx=True, save_tmin=True, save_t0=True, save_tmax=True)
                               
corrected_events_continuous,trial_idx = corrected_epochs_events.as_continuous()
corrected_events_continuous.add_events(events, drop_duplic=True)
corrected_events_continuous.to_csv(os.path.join(path_corrected_mrk, name_subj +'_corrected_evts.csv'),\
                                   header="sample,time,code,chan",\
                                   sep=',', save_sample=True, save_time=True, save_code=True, save_chan=True,\
                                   save_trial_idx=False)
