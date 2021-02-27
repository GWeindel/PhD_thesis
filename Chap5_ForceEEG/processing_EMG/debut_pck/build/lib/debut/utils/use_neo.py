# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:27:20 2018

@author: Laure Spieser and Boris Burle
Laboratoire de Neurosciences Cognitives
UMR 7291, CNRS, Aix-Marseille Université
3, Place Victor Hugo
13331 Marseille cedex 3
"""

import numpy as np
from .. import emgtools
from . import utilsfunc
#import neo

"""
Assumes one analog signal per electrode (i.e., channel), and electrode names
(i.e., channel names) is defined in analogsignal.name.
If more than one block/segment, assumes the same order of electrodes (i.e., channels)
in all blocks/segments.
"""

def get_ch_idx(blocks, ch_names='all'):
    """ 
    Gets channel indices.
    """
    # defines channels indices
    if ch_names == 'all': 
        ch_idx = list(range(blocks[0].segments[0].size['analogsignals']))
    else: 
        ch_names = utilsfunc.in_list(ch_names)
        list_names = [sig.name for sig in blocks[0].segments[0].analogsignals]
        ch_idx = [list_names.index(ch) for ch in ch_names if ch in list_names]
    return ch_idx

#def getTimesIdx(struct, tmin = 'first', tmax = 'last'):
#
#    if tmin == 'first' : tmin = 0
#    if tmax == 'last' : tmax = struct.times[-1]
#    times_idx = np.arange(struct.time_as_index(tmin),struct.time_as_index(tmax))
#    return times_idx


def get_data_array(blocks, ch_names='all', tmin=None, tmax=None):
    """ 
    Gets raw data.
    """
    ch_idx = get_ch_idx(blocks, ch_names)
    signals = np.array([])
    for bl in blocks:
        for seg in bl.segments:
            for ch in ch_idx:
                if len(signals) == 0:
                    signals = seg.analogsignals[ch].time_slice(t_start=tmin, t_stop=tmax).as_array()
                else:
                    signal = seg.analogsignals[ch].time_slice(t_start=tmin, t_stop=tmax).as_array()
                    if signals.shape[0] == signal.shape[0]:
                        signals = np.hstack((signals, signal))
                    else:
                        if ch_names == 'all':
                            ch_names = [sig.name for sig in blocks[0].segments[0].analogsignals]
                        raise ValueError('signal length do not match for channel {}'.format(utilsfunc.in_list(ch_names)[ch]))
    return signals.transpose()
    
def apply_filter(blocks, ch_names='all', N=3, low_cutoff=None, high_cutoff=None):
    """
    Apply high and low pass filters to input signal.
    
    Parameters:
    -----------
    raw : mne Raw data structure
        Raw mne data containing input signal to filter.
    ch_names : list  | str
        List of channel names on which filter will be applied. If 'all',
        filter is applied on all channels (Default 'all').
    N : int
        The order of the filter (Default 3).
    low_cutoff : float
        Cutoff frequency for high pass filter. If 'None', no high pass filter
        is applied.         
    high_cutoff : float
        Cutoff frequency for low pass filter. If 'None', no low pass filter
        is applied.
        
    Returns:
    --------
    raw_filter : raw mne data structure
        Raw mne data containing filtered signal.
    """
    from copy import deepcopy
    ch_idx = get_ch_idx(blocks, ch_names=ch_names)

    fblocks = deepcopy(blocks)
    for bl in fblocks:
        for seg in bl.segments:
            for ch in ch_idx:
                
                fsignal = seg.analogsignals[ch].as_array().transpose()
                if low_cutoff is not None:
                    fsignal = emgtools.hpFilter(fsignal, N=N, sf=seg.analogsignals[ch].sampling_rate, cutoff=low_cutoff)
                if high_cutoff is not None:
                    fsignal = emgtools.lpFilter(fsignal, N=N, sf=seg.analogsignals[ch].sampling_rate, cutoff=high_cutoff)
                seg.analogsignals[ch] = seg.analogsignals[ch].duplicate_with_new_array(fsignal.transpose())
    
    return fblocks

def bipolar_ref(blocks, anode, cathode, new_ch=None):
    """ 
    Sets bipolar montage.
    """
    anode_idx = get_ch_idx(blocks, anode)
    cathode_idx = get_ch_idx(blocks, cathode)
    if len(cathode_idx) == 1 :
        cathode_idx = cathode_idx * len(anode_idx)
    if new_ch is None :
        new_ch = anode_idx

    if len(anode_idx) != len(cathode_idx):
        raise ValueError('Number of anode and cathode must be equal, or number of cathode must be 1.')
    if len(new_ch) != len(anode_idx):
        raise ValueError('new_ch_names must include new name for each new channel.')
    
    for bl in blocks:
        for seg in bl.segments:
            for ch in range(len(anode_idx)):
                bipolar_signal = seg.analogsignals[anode_idx[ch]]._apply_operator(seg.analogsignals[cathode_idx[ch]],'__sub__')
                seg.analogsignals.append(seg.analogsignals[anode_idx[ch]].duplicate_with_new_array(bipolar_signal))
                seg.analogsignals[-1].name = new_ch[ch]
                seg.analogsignals[-1].annotations = {'bipolar_ref': {'anode': anode[ch], 'cathode': cathode[ch]}}

def select_channels(blocks, ch_names):
    """ 
    Selects designated channels.
    """
    ch_names = utilsfunc.in_list(ch_names)
    for bl in blocks:
        for seg in bl.segments:
            for ch in ch_names[::-1]:
                ch_idx = [idx for idx,sig in enumerate(seg.analogsignals) if sig.name == ch]
                if len(ch_idx) == 1:
                    seg.analogsignals.insert(0, seg.analogsignals.pop(ch_idx[0]))
                elif len(ch_idx) > 1:
                    raise ValueError('Can not select multiple channels, associate each signal to a single channel name.')
            seg.analogsignals = seg.analogsignals[:len(ch_names)]                    
            
def drop_channels(blocks, ch_names):
    """ 
    Deletes designated channels.
    """
    ch_names = utilsfunc.in_list(ch_names)
    for bl in blocks:
        for seg in bl.segments:
            seg.analogsignals = [sig for sig in seg.analogsignals if sig.name not in ch_names]
    
#def getVar(epochs, trials = 'all', ch_names = 'all', tmin = None, tmax = None, useTkeo = True, 
#           corVar = None):
#    """
#    Return mean and standard deviation of the input signal on specified
#    channels between tmin and tmax.
#
#    Parameters:
#    -----------
#    epochs : mne Epochs data structure
#        Input signal
#    ch_names : list  | str
#        List of channel names to use. If 'all', filter is
#        applied on all channels (Default 'all').
#    tmin : float | None
#        Start time for mean/variance computation. If None start at
#        first sample (default None).
#    tmax : float | None
#        End time for mean/variance computation. If None end at time 0
#        (default None).
#    useTkeo : bool
#        If True, compute mean and variance on the Teaker-Kayser
#        transformation of the input signal (default True).
#    corVar : float | None
#        If float, correct for outliers. The return mean and variance are
#        computed on all sample signals whose abs(z-score) < corVar
#        (default None).
#
#    Returns:
#    --------
#    mBl,sBl : list
#        mean and standard deviation on each channel
#
#    """
#        
#    if trials == 'all' : trials = np.arange(epochs._data.shape[0])
#    ch_idx = getChIdx(epochs, ch_names)
#    data_epochs = epochs._data[np.ix_(trials,ch_idx)]
#    
#    mBl,stBl = emg.getGlobalVariance(data_epochs, epochs.times, tmin = tmin, tmax = tmax, useTkeo = useTkeo, corVar = corVar)
#    
#    return mBl, stBl

