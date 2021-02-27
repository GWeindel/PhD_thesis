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


def get_ch_idx(raw, ch_names='all'):
    """ 
    Gets channel indices.
    """
    if ch_names == 'all': 
        ch_idx = list(range(len(raw.ch_names)))
    else:
        ch_idx = [raw.ch_names.index(ch) for ch in ch_names if ch in raw.ch_names]
    return ch_idx

def get_times_idx(raw, tmin=None, tmax=None):
    """ 
    Gets times indices.
    """
    if tmin == None:
        tmin = 0
    if tmax == None: 
        tmax = raw.times[-1]
    times_idx = np.arange(raw.time_as_index(tmin), raw.time_as_index(tmax))
    return times_idx

def get_data_array(raw, ch_names='all', tmin=None, tmax=None):
    """ 
    Gets raw data.
    """
    ch_idx = get_ch_idx(raw, ch_names)
    if tmin is None:
        samplemin = 0
    else:
        samplemin = raw.time_as_index(tmin)
    if tmax is not None:
        samplemax = raw.time_as_index(tmax)[0]
    else:
        samplemax = None
    return raw.get_data(ch_idx, start=samplemin, stop=samplemax)

def apply_filter(raw, ch_names='all', n=3, low_cutoff=None, high_cutoff=None):
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
    ch_idx = get_ch_idx(raw, ch_names)    
    array = raw._data[ch_idx, :]
        
    # Apply highpass and lowpass filters
    if low_cutoff is not None:
        array = emgtools.hpFilter(array, n=n, sf=raw.info['sfreq'], cutoff=low_cutoff)
    if high_cutoff is not None:
        array = emgtools.lpFilter(array, n=n, sf=raw.info['sfreq'], cutoff=high_cutoff)
    
    raw_filter = raw.copy()
    raw_filter._data[ch_idx, :] = array
    
    return raw_filter

def bipolar_ref(raw, anode, cathode, new_ch=None, copy=False, ch_info=None):
    """ 
    Sets bipolar montage.
    """
    from mne import set_bipolar_reference

    cathode = utilsfunc.in_list(cathode)
    if len(cathode) == 1 :
        cathode = cathode * len(utilsfunc.in_list(anode))
    if new_ch is None :
        new_ch = utilsfunc.in_list(anode)
        
    if copy is False:
        set_bipolar_reference(raw,anode=anode,\
                              cathode=cathode,ch_name=new_ch,\
                              copy=False, ch_info=ch_info)
    else:
        return set_bipolar_reference(raw,anode=anode,\
                                     cathode=cathode,ch_name=new_ch,\
                                     copy=True, ch_info=ch_info)

def select_channels(raw, ch_names):
    """ 
    Selects designated channels.
    """
    return raw.pick_channels(utilsfunc.in_list(ch_names))

def drop_channels(raw, ch_names):
    """ 
    Deletes designated channels.
    """
    return raw.drop_channels(utilsfunc.in_list(ch_names))

#def get_var(epochs, trials='all', ch_names='all', tmin=None, tmax=None, use_tkeo=True, 
#            cor_var=None):
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
#    use_tkeo : bool
#        If True, compute mean and variance on the Teaker-Kayser
#        transformation of the input signal (default True).
#    cor_var : float | None
#        If float, correct for outliers. The return mean and variance are
#        computed on all sample signals whose abs(z-score) < cor_var
#        (default None).
#
#    Returns:
#    --------
#    mbsl,stbsl : list
#        mean and standard deviation on each channel
#
#    """
#        
#    if trials == 'all' : trials = np.arange(epochs._data.shape[0])
#    ch_idx = get_ch_idx(epochs, ch_names)
#    data_epochs = epochs._data[np.ix_(trials,ch_idx)]
#    
#    mbsl,stbsl = emgtools.global_var(data_epochs, epochs.times, tmin=tmin, tmax=tmax, use_tkeo=use_tkeo, cor_var=cor_var)
#    
#    return mbsl, stbsl
    
