# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:27:20 2018

@author: Laure Spieser and Boris Burle
Laboratoire de Neurosciences Cognitives
UMR 7291, CNRS, Aix-Marseille Université
3, Place Victor Hugo
13331 Marseille cedex 3
"""

from .. import emgtools
from . import utilsfunc

def load_txt_file(fname, sep='\t', headerlines=1, ch_names_line=None):
    """Loads signal from text file.

    Load signal in text file, organised with channels in columns and time sample
    in rows.

    Parameters:
    -----------
    fname : str
        Name of the file to load.
    sep : str
        Separator between columns (default "\t").
    headerlines : int
        Number of headerlines to skip (default 1).
    ch_names_line : int
        Line in text file containing channel names. If None, channel names are
        ['channel0', 'channel1', ...] (default None).

    Returns:
    -----------
    array : 2D array
        Channels on axis 0, time samples in axis 1.
    ch_dict : dict
        Channels dictionnaries, with channel names as keys and channel index as 
        values.
    """
    import numpy as np
    
    f = open(fname, 'r')
    data_txt = f.read()
    f.close()

    data_list = data_txt.split('\n')
    data_list.remove('')
    
    nr_ts = len(data_list[headerlines:])
    nr_channels = len(data_list[headerlines:][0].split(sep))
    array = np.empty((nr_channels, nr_ts))

    for r in range(nr_ts):
        data_line = data_list[headerlines + r].split(sep)
        for c in range(nr_channels):
            array[c, r] = data_line[c]
        
    ch_dict = {}    
    if ch_names_line is None:
        list_chan = ['channel' + c for c in range(nr_channels)]
    else:
        list_chan = data_list[ch_names_line].split(sep)

    for c in range(nr_channels):
        ch_dict[list_chan[c]] = c 
        
    return array, ch_dict

def get_ch_idx(ch_dict, ch_names='all'):
    """ 
    Gets channel indices.
    """
    if ch_names == 'all': 
        ch_idx = list(ch_dict.values())
    else: 
        ch_names = utilsfunc.in_list(ch_names)
        ch_idx = [ch_dict[ch] for ch in ch_names if ch in ch_dict.keys()]
    return ch_idx

#def getTimesIdx(struct, tmin = 'first', tmax = 'last'):
#
#    if tmin == 'first' : tmin = 0
#    if tmax == 'last' : tmax = struct.times[-1]
#    times_idx = np.arange(struct.time_as_index(tmin),struct.time_as_index(tmax))
#    return times_idx


def apply_filter(data, ch_dict, sf, ch_names='all', N=3, low_cutoff=None, high_cutoff=None):
    """
    Apply high and low pass filters to input signal.
    
    Parameters:
    -----------
    data : 2D array
        Array containing input signal to filter, channels are on axis 0, 
        time samples on axis 1.
    ch_dict: dict
        Channel names / channel index correspondance dictionnary.
    sf : int | float
        Input signal sampling frequency.
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
    data_filter : 2D array 
        Data array containing filtered signal.
    
    """
    ch_idx = get_ch_idx(ch_dict, ch_names=ch_names)

    for ch in ch_idx:
        if low_cutoff is not None:
            data[ch] = emgtools.hpFilter(data[ch], N=N, sf=sf, cutoff=low_cutoff)
        if high_cutoff is not None:
            data[ch] = emgtools.lpFilter(data[ch], N=N, sf=sf, cutoff=high_cutoff)
    return data
    
def bipolar_ref(data, ch_dict, anode, cathode, new_ch=None):
    """
    Sets bipolar montage.
    """
    import numpy as np
    anode_idx = get_ch_idx(ch_dict, anode)
    cathode_idx = get_ch_idx(ch_dict, cathode)
    if len(cathode_idx) == 1 :
        cathode_idx = cathode_idx * len(anode_idx)
    if new_ch is None :
        new_ch = anode_idx

    if len(anode_idx) != len(cathode_idx):
        raise ValueError('Number of anode and cathode must be equal, or number of cathode must be 1.')
    if len(new_ch) != len(anode_idx):
        raise ValueError('new_ch names must include new name for each new channel.')
    
    for ch in range(len(anode_idx)):
        
        bipolar_signal = data[anode_idx[ch]] - data[cathode_idx[ch]]
        data = np.vstack((data, bipolar_signal))
        ch_dict[new_ch[ch]] = data.shape[0] - 1
        
    return drop_channels(data, ch_dict, anode + cathode)        

def select_channels(data, ch_dict, ch_names):
    """ 
    Selects designated channels.
    """
    import numpy as np
    ch_idx = get_ch_idx(ch_dict, ch_names)
    select_data = np.empty((len(ch_idx), data.shape[1]))
    select_ch_dict = {}
    for ch in range(len(ch_idx)):
        select_data[ch] = data[ch_idx[ch]]
        select_ch_dict[ch_names[ch]] = ch
    return select_data, select_ch_dict

def drop_channels(data, ch_dict, ch_names):
    """ 
    Deletes designated channels.
    """
    chan_list = list(ch_dict.keys())
    select_channels = [ch for ch in chan_list if ch not in ch_names]
    return select_channels(data, ch_dict, select_channels)
    
