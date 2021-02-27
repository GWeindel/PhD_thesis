# -*- coding: utf-8 -*-
"""
Created on Thu May 24 12:38:03 2018

@author: Laure
"""

import debut.events as evt
from debut.utils import utilsfunc


def events_to_lists(evts, t0_codes, stim_codes, resp_codes, \
                    onset_codes=['onset'], offset_codes=['offset'], \
                    other_codes=[], end_codes=[],\
                    min_stim=1, max_stim=1, min_resp=0, max_resp=1, tmax=None,\
                    stop_after_response=False, stop_after_offset=True):
    """Reads trials of a continuous Events object.

    Reads continuous events and detects trials starts and ends. The start of a
    new trial is detected each time an event with code defined in 't0_codes'
    is read. The end of trial is determined with 'idx_end_event' function. End of
    trial is detected at the first occurence of any of the following: 1) a new
    'T0_codes' event, 2) a new stimulus event (as defined by stim_codes), when the
    maximum number of stimulus has been reached, 3) when the maximum number of
    responses occured (if stop_after_response is True) or after offset(s) of all
    bursts starting before last response (if stop_after_offset is True), 4) an
    'end_codes' event, 5), if none of 1), 2), 3) or 4), trial ends when maximum
    duration after previous T0_codes event is reached (if maximum duration tmax
    is provided). Returns lists containing each trial's index ('trial_list'),
    stimulus event(s) ('stim_list'), response event(s) ('resp_list'), onset event(s)
    ('onset_list'), offset event(s) ('offset_list'), any other event(s) 
    ('other_evt_list'). Each list index corresponds to a new trial, and trial order 
    corresponds to the Events object.

    Parameters:
    -----------
    evts: class Events
        Events object to read.
    t0_codes: list | 1Darray
        Events codes for detecting trial start (e.g., use the the same events
        than for trial segmentation).
    stim_codes: list | 1Darray
        Codes of stimulus events (may also contain T0_codes).
    resp_codes : list | 1Darray
        Codes of responses events.
    onset_codes : list | 1Darray
        Codes of EMG onset events (default ['onset']).
    offset_codes : list | 1Darray
        Codes of EMG offset events. Must be provided if 'stop_after_offset'
        is True (default ['offset']).
    other_codes : list | 1Darray
        Codes of any other event that need to be stored. If equal to [], all
        other codes are stored, if None, no other code is stored (default []).
    end_codes : list | 1Darray
        Codes signaling the end of the trial (default []).
    min_stim : int
        Minimum number of stimulus events (as defined in stim_codes) to read in
        the trial (default 1).
    max_stim : int
        Maximum number of stimulus events (as defined in stim_codes) to read in
        the trial (default 1).
    min_resp : int
        Minimum number of response events (as defined in resp_codes) to read in
        the trial (default 0).
    max_resp : int
        Maximim number of response events (as defined in resp_codes) to read in
        the trial (default 1).
    tmax : float
        Maximum trial duration in second. If None, trial end is detected only if
        either 1), 2), 3) or 4) (see above) occured. If provided, end event is
        added at tmax after T0_codes event if none of 1), 2), 3) or 4) occured
        (default None).
    stop_after_resp : bool
        If True, the end trial marker is added after that the maximum number of
        responses occured (default False).
    stop_after_offset: bool
        If True, the end trial marker is added after offset(s) of all bursts
        starting before the last response occured (default True).

    Returns:
    --------
    trial_list : list
        List of trials indices.
    stim_list : list
        List of stimulus events.
    resp_list : list
        List of response events.
    onset_list : list
        List of onset events.
    offset_list : list
        List of offset events.
    other_list : list
        List of other events.
    """
    import numpy as np

    if other_codes == []:
        other_codes = [c for c in np.unique(evts.code) if c not in utilsfunc.in_list(stim_codes) + utilsfunc.in_list(resp_codes) + utilsfunc.in_list(onset_codes) + utilsfunc.in_list(offset_codes)]

    start_idx = evts.find_events(code=t0_codes)
    trial_idx = np.hstack((start_idx, evts.nb_events()))

    nr_trials = len(start_idx)
    trial_list = [None] * nr_trials
    stim_list = [None] * nr_trials
    resp_list = [None] * nr_trials
    onset_list = [None] * nr_trials
    offset_list = [None] * nr_trials
    other_list = [None] * nr_trials

    for idx in range(len(start_idx)):

        code = evts.code[trial_idx[idx]:trial_idx[idx+1]]
        chan = evts.chan[trial_idx[idx]:trial_idx[idx+1]]
        sample = evts.lat.sample[trial_idx[idx]:trial_idx[idx+1]]
        time = evts.lat.time[trial_idx[idx]:trial_idx[idx+1]]

        end_idx = idx_end_event(code, chan, time, stim_codes, resp_codes,\
                                onset_codes=onset_codes, offset_codes=offset_codes,\
                                min_stim=min_stim, max_stim=max_stim, min_resp=min_resp, max_resp=max_resp,\
                                tmax=tmax,\
                                stop_after_response=stop_after_response, stop_after_offset=stop_after_offset)

        evt_trial = evt.Events(sample=sample[:end_idx+1], code=code[:end_idx+1], chan=chan[:end_idx+1], sf=evts.sf)
        stim_evt, resp_evt, onset_evt, offset_evt, other_evt = read_trial_events(evt_trial, stim_codes, resp_codes, \
                                                                                 onset_codes=onset_codes, offset_codes=offset_codes,\
                                                                                 other_codes=other_codes)

        trial_list[idx] = idx
        stim_list[idx] = stim_evt
        resp_list[idx] = resp_evt
        onset_list[idx] = onset_evt
        offset_list[idx] = offset_evt
        other_list[idx] = other_evt

    return trial_list, stim_list, resp_list, onset_list, offset_list, other_list

def events_to_df(evts, t0_codes, stim_codes, resp_codes, \
                 onset_codes=['onset'], offset_codes=['offset'], \
                 other_codes=[], end_codes=[],\
                 min_stim=1, max_stim=1, min_resp=0, max_resp=1, tmax=None,\
                 stop_after_response=False, stop_after_offset=True):
    """ Gets pandas data frame from continuous events.

    Reads continuous events and detects trials starts and ends. The start of a
    new trial is detected each time an event with code defined in 't0_codes'
    is read. The end of trial is determined with 'idx_end_event' function. End of
    trial is detected at the first occurence of any of the following: 1) a new
    't0_codes' event, 2) a new stimulus event (as defined by stim_codes), when the
    maximum number of stimulus has been reached, 3) when the maximum number of
    responses occured (if stop_after_response is True) or after offset(s) of all 
    bursts starting before the last response (if stop_after_offset is True), 
    4) an 'end_codes' event, 5), if none of 1), 2), 3) or 4), trial ends when 
    maximum duration after previous T0_codes event is reached (if maximum duration 
    tmax is provided). Returns a pandas dataframe containing each trial's stimulus 
    code, sample and time, response code and time, onset(s) code, channel and 
    time, offset(s) code, channel and time and any other event(s) code and time.
    Each dataframe row corresponds to a trial, and trial order corresponds to
    the Events object.

    Parameters:
    -----------
    evts : class Events
        Events object to read.
    t0_codes : list | 1Darray
        Events codes for detecting trial start (e.g., use the the same events
        than for trial segmentation).
    stim_codes : list | 1Darray
        Codes of stimulus events (may also contain t0_codes).
    resp_codes : list | 1Darray
        Codes of responses events.
    onset_codes : list | 1Darray
        Codes of EMG onset events (default ['onset']).
    offset_codes : list | 1Darray
        Codes of EMG offset events. Must be provided if 'stop_after_offset'
        is True (default ['offset']).
    other_codes : list | 1Darray
        Codes of any other event that need to be stored. If equal to [], all
        other codes are stored, if None, no other code is stored (default []).
    end_codes : list | 1Darray
        Codes signaling the end of the trial (default []).
    min_stim : int
        Minimum number of stimulus events (as defined in stim_codes) to read in
        the trial (default 1).
    max_stim : int
        Maximum number of stimulus events (as defined in stim_codes) to read in
        the trial (default 1).
    min_resp : int
        Minimum number of response events (as defined in resp_codes) to read in
        the trial (default 0).
    max_resp : int
        Maximim number of response events (as defined in resp_codes) to read in
        the trial (default 1).
    tmax : float
        Maximum trial duration in second. If None, trial end is detected only if
        either 1), 2), 3) or 4) (see above) occured. If provided, end event is
        added at tmax after t0_codes event if none of 1), 2), 3) or 4) occured
        (default None).
    stop_after_resp : bool
        If True, the end trial marker is added after that the maximum number of
        responses occured (default False).
    stop_after_offset: bool
        If True, the end trial marker is added after offset(s) of all bursts
        starting before the last response occured (default True).

    Returns:
    --------
        Pandas DataFrame, with at least columns 'stim_code', 'stim_sample',
        'stim_time', 'resp_code', 'resp_sample', 'resp_time', 'onset_chan', 
        'onset_code', 'onset_sample', 'onset_time', 'offset_chan', 'offset_code',
        'offset_sample', 'offset_time'. Each line of the data frame contains 
        information corresponding to one trial.
    """
    import pandas as pd

    trial_list, stim_list, resp_list, onset_list, offset_list, other_list = events_to_lists(evts, t0_codes, stim_codes, resp_codes, \
                                                                                            onset_codes=onset_codes, offset_codes=offset_codes,\
                                                                                            other_codes=other_codes, end_codes=end_codes,\
                                                                                            min_stim=min_stim, max_stim=max_stim, min_resp=min_resp, max_resp=max_resp, tmax=tmax,\
                                                                                            stop_after_response=stop_after_response,\
                                                                                            stop_after_offset=stop_after_offset)

    stim_df = evts_list_to_df(stim_list, 'stim', keep_code=True, keep_chan=False, keep_time=True, keep_sample=True)
    if max_stim <= 1:
        for col, i in stim_df.iteritems():
            stim_df[col] = utilsfunc.remove_list(i)

    resp_df = evts_list_to_df(resp_list, 'resp', keep_code=True, keep_chan=False, keep_time=True, keep_sample=True)
    if max_resp <= 1:
        for col, i in resp_df.iteritems():
            resp_df[col] = utilsfunc.remove_list(i)

    onset_df = evts_list_to_df(onset_list, 'onset', keep_code=True, keep_chan=True, keep_time=True, keep_sample=True)
    offset_df = evts_list_to_df(offset_list, 'offset', keep_code=True, keep_chan=True, keep_time=True, keep_sample=True)
    other_df = evts_list_to_df(other_list, 'other', keep_code=True, keep_chan=True, keep_time=True, keep_sample=True)

    return pd.concat([stim_df, resp_df, onset_df, offset_df, other_df], axis=1)


def epochevents_to_lists(ep_evts, stim_codes, resp_codes, \
                         onset_codes=['onset'], offset_codes=['offset'], \
                         other_codes=[], end_codes=[],\
                         min_stim=1, max_stim=1, min_resp=0, max_resp=1, tmax=None, \
                         stop_after_response=False, stop_after_offset=True):
    """Returns separate lists for each trial index, stim, resp, onset, offset and other events.

    Reads each trial of the epoch events. Detects trials ends with 'idx_end_event' function
    to avoid overlapping trials. End of trial is detected at the first occurence
    of any of the following: 1) a new stimulus event (as defined by stim_codes), when
    the maximum number of stimulus has been reached, 2) when the maximum number
    of responses occured (if stop_after_response is True) or after offset(s) of
    all bursts starting before last response (if stop_after_offset is True),
    3) an 'end_codes' event, 4), at time = tmax, if none of 1), 2) or 3) occurs within
    maximum time duration after trial starts defined by tmax. Returns lists containing
    each trial's index ('trial_list'), stimulus event(s) ('stim_list'), response event(s)
    ('resp_list'), onset event(s) ('onset_list'), offset event(s) ('offset_list'),
    any other event(s) ('other_evt_list'). Each list index corresponds to a new trial,
    and trial order corresponds to the EpochEvents object.

    Parameters:
    -----------
    stim_codes : list | 1Darray
        Codes of stimulus events.
    resp_codes : list | 1Darray
        Codes of responses events.
    onset_codes : list | 1Darray
        Codes of EMG onset events (default ['onset']).
    offset_codes : list | 1Darray
        Codes of EMG offset events. Must be provided if 'stop_after_EMGoffset'
        is True (default ['offset']).
    other_codes : list | 1Darray
        Codes of any other event that need to be stored. If equal to [], all
        other _codes are stored, if None, no other code is stored (default []).
    end_codes : list | 1Darray
        Codes signaling the end of the trial (default []).
    min_stim : int
        Minimum number of stimulus events (as defined in stim_codes) to read in
        the trial (default 1).
    max_stim : int
        Maximum number of stimulus events (as defined in stim_codes) to read in
        the trial (default 1).
    min_resp : int
        Minimum number of response events (as defined in resp_codes) to read in
        the trial (default 0).
    max_resp : int
        Maximim number of response events (as defined in resp_codes) to read
        in the trial (default 1).
    tmax : float
        Maximum trial duration in second. If None, trial end is detected only
        if either 1), 2), 3) or 4) (see above) occured. If provided, end event
        is added at tmax after T0_codes event if none of 1), 2), 3) or 4)
        occured (default None).
    stop_after_resp : bool
        If True, the end trial marker is added after that the maximum number of
        responses occured (default False).
    stop_after_offset: bool
        If True, the end trial marker is added after offset(s) of all bursts
        starting before last response occured (default True).

    Returns:
    --------
    trial_list : list
        List of trials indices.
    stim_list : list
        List of stimulus events.
    resp_list : list
        List of response events.
    onset_list : list
        List of onset events.
    offset_list : list
        List of offset events.
    other_evt_list : list
        List of other events.
    """
    nr_trials = ep_evts.nb_trials()
    trial_list = [None] * nr_trials
    stim_list = [None] * nr_trials
    resp_list = [None] * nr_trials
    onset_list = [None] * nr_trials
    offset_list = [None] * nr_trials
    other_evt_list = [None] * nr_trials

    for idx in range(nr_trials):

        code = ep_evts.list_evts_trials[idx].code
        chan = ep_evts.list_evts_trials[idx].chan
        sample = ep_evts.list_evts_trials[idx].lat.sample
        time = ep_evts.list_evts_trials[idx].lat.time

        end_idx = idx_end_event(code, chan, time, stim_codes, resp_codes, onset_codes=onset_codes, offset_codes=offset_codes,\
                                min_stim=min_stim, max_stim=max_stim, min_resp=min_resp, max_resp=max_resp,\
                                tmax=tmax,\
                                stop_after_response=stop_after_response, stop_after_offset=stop_after_offset)

        evtTrial = evt.Events(sample=sample[:end_idx+1], time=time[:end_idx+1], code=code[:end_idx+1], chan=chan[:end_idx+1], sf=ep_evts.sf)
        stim_evt, resp_evt, onset_evt, offset_evt, other_evt = read_trial_events(evtTrial, stim_codes, resp_codes,\
                                                                                 onset_codes=onset_codes, offset_codes=offset_codes,\
                                                                                 other_codes=other_codes)

        trial_list[idx] = idx
        stim_list[idx] = stim_evt
        resp_list[idx] = resp_evt
        onset_list[idx] = onset_evt
        offset_list[idx] = offset_evt
        other_evt_list[idx] = other_evt

    return trial_list, stim_list, resp_list, onset_list, offset_list, other_evt_list

def epochevents_to_df(ep_evts, stim_codes, resp_codes,\
                      onset_codes=['onset'], offset_codes=['offset'],\
                      other_codes=[], end_codes=[],\
                      min_stim=1, max_stim=1, min_resp=0, max_resp=1, tmax=None,\
                      stop_after_response=False, stop_after_offset=True):
    """Creates pandas data frame from epoch events.

    Reads each trial of epoch events. Detects trials ends using 'idxEndEvent' 
    function. End of trial is detected at the first occurence of any of the following: 
    1) a new stimulus event (as defined by stim_codes), when the maximum number
    of stimulus has been reached, 2) when the maximum number of responses occured
    (if stop_after_response is True) or after offset(s) of all bursts starting
    before the last response (if stop_after_offset is True), 3) an
    'end_codes' event, 4) at time = tmax, if none of 1), 2) or 3) occurs within
    maximum time duration after trial starts defined by tmax. Returns a pandas 
    dataframe containing each trial's stimulus code, sample and time, response 
    code and time, onset(s) code, channel and time, offset(s) code, channel 
    and time and any other event(s) code and time. Each dataframe row corresponds 
    to a trial, and trial order corresponds to the EpochEvents object.

    Parameters:
    -----------
    ep_evts : class EpochEvents
        EpochEvents object to read.
    stim_codes : list | 1Darray
        Codes of stimulus events.
    resp_codes : list | 1Darray
        Codes of responses events.
    onset_codes : list | 1Darray
        Codes of EMG onset events (default ['onset']).
    offset_codes : list | 1Darray
        Codes of EMG offset events. Must be provided if 'stop_after_offset'
        is True (default ['offset']).
    other_codes : list | 1Darray
        Codes of any other event that need to be stored. If equal to [], all
        other codes are stored, if None, no other code is stored (default []).
    end_codes : list | 1Darray
        Codes signaling the end of the trial (default []).
    min_stim : int
        Minimum number of stimulus events (as defined in stim_codes) to read in
        the trial (default 1).
    max_stim : int
        Maximum number of stimulus events (as defined in stim_codes) to read in
        the trial (default 1).
    min_resp : int
        Minimum number of response events (as defined in resp_codes) to read in
        the trial (default 0).
    max_resp : int
        Maximim number of response events (as defined in resp_codes) to read in
        the trial (default 1).
    tmax : float
        Maximum trial duration in second. If None, trial end is detected only if
        either 1), 2), 3) or 4) (see above) occured. If provided, end event is
        added at tmax after T0_codes event if none of 1), 2), 3) or 4) occured
        (default None).
    stop_after_resp : bool
        If True, the end trial marker is added after that the maximum number of
        responses occured (default False).
    stop_after_offset: bool
        If True, the end trial marker is added after offset(s) of all bursts
        starting before the last response occured (default True).

    Returns:
    --------
        Pandas DataFrame, with at least columns 'stim_code', 'stim_sample',
        'stim_time', 'resp_code', 'resp_sample', 'resp_time', 'onset_chan', 
        'onset_code', 'onset_sample', 'onset_time', 'offset_chan', 'offset_code',
        'offset_sample', 'offset_time'. Each line of the data frame contains 
        information corresponding to one trial.
    """
    import pandas as pd

    trial_list, stim_list, resp_list, onset_list, offset_list, other_evt_list = epochevents_to_lists(ep_evts, stim_codes, resp_codes, \
                                                                                                     onset_codes=onset_codes, offset_codes=offset_codes,\
                                                                                                     other_codes=other_codes, end_codes=end_codes,\
                                                                                                     min_stim=min_stim, max_stim=max_stim, min_resp=min_resp, max_resp=max_resp, tmax=tmax,\
                                                                                                     stop_after_response=stop_after_response,\
                                                                                                     stop_after_offset=stop_after_offset)

    stim_df = evts_list_to_df(stim_list, 'stim', keep_code=True, keep_chan=False, keep_time=True, keep_sample=True)
    if max_stim <= 1:
        for col, i in stim_df.iteritems():
            stim_df[col] = utilsfunc.remove_list(i)

    resp_df = evts_list_to_df(resp_list, 'resp',keep_code=True, keep_chan=False, keep_time=True, keep_sample=True)
    if max_resp <= 1:
        for col, i in resp_df.iteritems():
            resp_df[col] = utilsfunc.remove_list(i)

    onset_df = evts_list_to_df(onset_list, 'onset', keep_code=True, keep_chan=False, keep_time=True, keep_sample=True)
    offset_df = evts_list_to_df(offset_list, 'offset', keep_code=True, keep_chan=False, keep_time=True, keep_sample=True)
    other_df = evts_list_to_df(other_evt_list, 'other_evt', keep_code=True, keep_chan=False, keep_time=True, keep_sample=True)

    return pd.concat([stim_df, resp_df, onset_df, offset_df, other_df], axis=1)


def evts_list_to_df(evt_list, name_evt, keep_code=True, keep_chan=False, keep_time=True, keep_sample=True):
    """Returns pandas.DataFrame containing Events information.
    """
    import pandas as pd
    name_evt = str(name_evt)
    big_dict = {}
    if keep_code is True:
        code = []
        for e in evt_list:
            code.append(e.code)
        big_dict[name_evt + '_code'] = code
    if keep_chan is True:
        chan = []
        for e in evt_list:
            chan.append(e.chan)
        big_dict[name_evt + '_chan'] = chan
    if keep_time is True:
        time = []
        for e in evt_list:
            time.append(e.lat.time)
        big_dict[name_evt + '_time'] = time
    if keep_sample is True:
        sample = []
        for e in evt_list:
            sample.append(e.lat.sample)
        big_dict[name_evt + '_sample'] = sample

    return pd.DataFrame(big_dict)


def read_trial_events(evt_trial, stim_codes, resp_codes,\
                      onset_codes=['onset'], offset_codes=['offset'], other_codes=[]):
    """Reads single trial's events.

    Reads evt_trial, which should contain Events object of a single trial and
    returns different Events object containing respectively: stimulus events,
    response events , onset events , offset events , other events.

    Parameters:
    -----------
    evt_trial : class Events
        Events object to read.
    stim_codes : list | 1Darray
        Codes of stimulus events.
    resp_codes : list | 1Darray
        Codes of responses events.
    onset_codes : list | 1Darray
        Codes of onset events.
    offset_codes : list | 1Darray
        Codes of offset events.
    other_evt_codes : list | 1Darray
        Codes of any other event that need to be stored.
    end_codes : list | 1Darray
        Codes signaling the end of the trial. If missing, use
        'add_end_event' to insert next_trial_codes.

    Returns:
    --------
    stim_evt : class Events
        Events object containing events which code is contained in
        stim_codes.
    resp_evt : class Events
        Events object containing events which code is contained in
        resp_codes.
    onset_evt : class Events
        Events object containing events which code is contained in
        onset_codes.
    offset_evt : class Events
        Events object containing events which code is contained in
        offset_codes.
    other_evt : class Events
        Events object containing events which code is contained in
        other_evt_codes.
    """
    stim_codes = utilsfunc.in_list(stim_codes)
    resp_codes = utilsfunc.in_list(resp_codes)
    onset_codes = utilsfunc.in_list(onset_codes)
    offset_codes = utilsfunc.in_list(offset_codes)

    other_codes = utilsfunc.in_list(other_codes)
    if other_codes == []:
        other_codes = [c for c in set(evt_trial.code) if c not in stim_codes + resp_codes + onset_codes + offset_codes]

    stim_evt = evt.Events(sf=evt_trial.sf)
    resp_evt = evt.Events(sf=evt_trial.sf)
    onset_evt = evt.Events(sf=evt_trial.sf)
    offset_evt = evt.Events(sf=evt_trial.sf)
    other_evt = evt.Events(sf=evt_trial.sf)

    resp = []
    for c in range(evt_trial.nb_events()):

        if evt_trial.code[c] in stim_codes:
            stim_evt.add_events(evt_trial.get_events(c))
        if evt_trial.code[c] in resp_codes:
            resp.append(evt_trial.code[c])
            resp_evt.add_events(evt_trial.get_events(c))

        if evt_trial.code[c] in onset_codes:
            onset_evt.add_events(evt_trial.get_events(c))
        if evt_trial.code[c] in offset_codes:
            offset_evt.add_events(evt_trial.get_events(c))

        if evt_trial.code[c] in other_codes:
            other_evt.add_events(evt_trial.get_events(c))

    if offset_evt.nb_events() > 0 and offset_evt.nb_events() != onset_evt.nb_events():
        print(('Warning: different number of onset and offset around time {} !').format(stim_evt.lat)) 
        
    return stim_evt, resp_evt, onset_evt, offset_evt, other_evt

def add_limits_events(evts, t0_codes, stim_codes, resp_codes, start_code, end_code, channel=-1,\
                      start_time=-.05, end_time=0,\
                      onset_codes=['onset'], offset_codes=['offset'],\
                      min_stim=1, max_stim=1, min_resp=0, max_resp=1, tmax=None,\
                      stop_after_response=False, stop_after_offset=True):
    """Adds a marker at the start and the end of each trial.

    Adds events marking each trial start and end. Returns a new Events object,
    containing originial events plus added start and end events. Event marking
    a trial start is added each time an event code whose code is defined in
    'T0_codes' occurs (e.g., can be codes that would be used for epoching). End
    of trial event is detected using 'idxEndEvent' function as: 1) the event
    preceding the occurence of next startCode, 2) the event preceding the
    occurence of next stimulus, if the maximum number of stimulus already occured,
    3) the last event response when the maximum number of responses occured (if
    stop_after_response is True) or the offset(s) of all bursts starting before
    the response occured (if stop_after_offset is True), 4) when none of 1), 2)
    or 3) occured within  duration after T0Code event (if maximum duration tmax
    is provided).

    Parameters:
    -----------
    evts : class Events
        Events object to read.
    t0_codes : list | 1Darray
        Codes of events signaling trial start (e.g., would be used for events epoching).
    stim_codes : list | 1Darray
        Codes of stimulus events (may also include t0_codes).
    resp_codes : list | 1Darray
        Codes of responses events.
    start_code : int | str
        Event code that will be added to mark trials start.
    end_code : int | str
        Event code that will be added to mark trials end.
    channel : list | 1Darray
        Event channel for start_code and end_code (default -1, can be set to 'All').
    start_time : float
        Time in second, relative to t0 event time, at which start event is added
        (default -.05s).
    end_time : float
        Time in second, relative to last trial event, at which end event is added.
        If endTime=0, end event is added at the sample following last trial event
        (default 0s).
    onset_codes : list | 1Darray
        Codes of EMG onset events. Must be provided if 'stop_after_EMGoffset'
        is True (default ['onset']).
    offset_codes : list | 1Darray
        Codes of EMG offset events. Must be provided if 'stop_after_EMGoffset'
        is True (default ['offset']).
    min_stim : int
        Minimum number of stimulus events (as defined in stim_codes) to read
        in the trial (default 1).
    max_stim : int
        Maximum number of stimulus events (as defined in stim_codes) to read
        in the trial (default 1).
    min_resp : int
        Minimum number of response events (as defined in resp_codes) to read
        in the trial (default 0).
    max_resp : int
        Maximum number of response events (as defined in resp_codes) to read
        in the trial (default 1).
    tmax : float
        Maximum trial duration in second. If None, end event is added only if
        either 1), 2) or 3) (see above) occured. If provided, end event is
        added at tmax after t0 event if none of 1), 2) or 3) occured (default
        None).
    stop_after_resp : bool
        If True, the end trial marker is added after that the maximum number of
        responses occured (default False).
    stop_after_offset : bool
        If True, the end trial marker is added after offset(s) of all bursts
        starting before the last response occured (default True).
    """
    import numpy as np

    trial_idx = evts.find_events(code=t0_codes)
    trial_idx = np.hstack((trial_idx, evts.nb_events()))

    start_sample = round(start_time * evts.sf)
    if end_time == 0:
        end_sample = 1
    else:
        end_sample = round(end_time * evts.sf)

    bound_evts = evts.copy()
    for idx in range(len(trial_idx)-1):

        if start_code is not None:
            bound_evts.add_events(evt.Events(sample=evts.lat.sample[trial_idx[idx]] + start_sample, code=start_code, chan=channel, sf=evts.sf))

        if end_code is not None:

            code = evts.code[trial_idx[idx]:trial_idx[idx+1]]
            chan = evts.chan[trial_idx[idx]:trial_idx[idx+1]]
            time = evts.lat.time[trial_idx[idx]:trial_idx[idx+1]]

            end_idx = idx_end_event(code, chan, time, stim_codes, resp_codes,\
                                    onset_codes=onset_codes, offset_codes=offset_codes,\
                                    min_stim=min_stim, max_stim=max_stim, min_resp=min_resp, max_resp=max_resp, tmax=tmax,\
                                    stop_after_response=stop_after_response, stop_after_offset=stop_after_offset)

            bound_evts.add_events(evt.Events(sample=evts.lat.sample[trial_idx[idx]+end_idx] + end_sample, code=end_code, chan=channel, sf=evts.sf))

    return bound_evts

def idx_end_event(code, chan, time, stim_codes, resp_codes,\
                  onset_codes=['onset'], offset_codes=['offset'],\
                  min_stim=1, max_stim=1, min_resp=0, max_resp=1, tmax=None,\
                  stop_after_response=False, stop_after_offset=True):
    """Finds the index at which trial ends.

    End of trial is detected: 1) before the occurence of next stimulus, if the
    maximum number of stimulus already occured, 2) after maximum number of
    responses  occured (if stop_after_response is True) or after offset(s) of
    all bursts starting before the last response occured (if stop_after_offset
    is True), 3) if none of 1) or 2) occured, end event is added when maximum
    duration is reached (if maximum duration tmax is provided).

    Parameters:
    -----------
    code : list
        List of events codes.
    chan : list
        List of corresponding events channels.
    time : list
        List of corresponding events times.
    stim_codes : list | 1Darray
        Codes of stimulus events (may also include t0_codes).
    resp_codes : list | 1Darray
        Codes of responses events.
    onset_codes : list | 1Darray
        Codes of EMG onset events. Must be provided if 'stop_after_EMGoffset'
        is True (default ['onset']).
    offset_codes : list | 1Darray
        Codes of EMG offset events. Must be provided if 'stop_after_EMGoffset'
        is True (default ['offset']).
    min_stim : int
        Minimum number of stimulus events (as defined in stim_codes) to read
        in the trial (default 1).
    max_stim : int
        Maximum number of stimulus events (as defined in stim_codes) to read
        in the trial (default 1).
    min_resp : int
        Minimum number of response events (as defined in resp_codes) to read
        in the trial (default 0).
    max_resp : int
        Maximum number of response events (as defined in resp_codes) to read
        in the trial (default 1).
    tmax : float
        Maximum trial duration in second. If None, end event is added only if
        either 1) or 2) (see above) occured. If provided, end event is added at
        tmax after time[0] (the time of the first provided event) occured (default
        None).
    stop_after_resp : bool
        If True, the end trial marker is added after that the maximum number of
        responses occured (default False).
    stop_after_offset : bool
        If True, the end trial marker is added after offset(s) of all bursts
        starting before the last response occured (default True).
    """
    import numpy as np

    stim_codes = utilsfunc.in_list(stim_codes)
    resp_codes = utilsfunc.in_list(resp_codes)
    onset_codes = utilsfunc.in_list(onset_codes)
    offset_codes = utilsfunc.in_list(offset_codes)

    stim = []
    resp = []
    emg = 0
    end_trial = False
    next_trial = False
    m = 0
    while (end_trial is False) and (next_trial is False) and (m < len(code)):

        if emg < 0: # ignore when starts by an offset
            emg = 0
        
        if code[m] in stim_codes:
            if len(stim) < max_stim:
                stim.append(code[m])
            else:
                next_trial = True

        if code[m] in resp_codes:
            resp.append(code[m])
            if (len(resp) == max_resp) and (min_stim <= len(stim) <= max_stim) and (stop_after_response):
                end_trial = True

        if code[m] in onset_codes:
            emg += 1

        if code[m] in offset_codes:
            emg -= 1
        if (stop_after_offset) and (min_stim <= len(stim) <= max_stim) and (len(resp) == max_resp):
#				if (resp[-1] in resp_chan_dict.keys()) and (resp_chan_dict[resp[-1]] == chan[m]):
            if emg == 0:
                end_trial = True

        m += 1

    if next_trial is True:
        m = m-2
    else:
        m = m-1

    if (tmax is not None) and (time[m]-time[0] > tmax):
        m = np.where(time-time[0] <= tmax)[0][-1]

    return m

def decode_accuracy(dataframe, stim_resp_dict, resp_chan_dict,\
                    stim_code_col='stim_code', resp_code_col='resp_code', resp_time_col='resp_time', \
                    onset_time_col='onset_time', onset_chan_col='onset_chan',\
                    offset_time_col='offset_time', offset_chan_col='offset_chan'):
    """Decodes response and channel (e.g., EMG) accuracy, assuming one stimulus and one response.

    Reads each trial of the pandas DataFrame and decodes accuracy using
    'decodeTrialAccuracy'. Returns dataframe, with new columns containing
    response and channel onset(s)/offset(s) accuracy. Channel onset(s)/offset(s)
    are also sorted as responding (associated with the response occurrence),
    and non responding (not associated with the response).

    Parameters:
    -----------
    dataframe : class pandas.DataFrame
        Pandas DataFrame with at least columns containing stimulus codes,
        response codes, onset latency(ies), onset channel(s), offset latency(ies),
        offset channel(s).
    stim_resp_dict : dict
        Dictionnary with possible stimulus codes as keys, and corresponding
        correct response codes as values.
    resp_chan_dict : dict
        Dictionnary with response codes as keys, and corresponding channels indices
        as values.
    stim_code_col : str
        Name of dataframe column containing stimulus code.
    resp_code_col : str
        Name of dataframe column containing response code.
    onset_time_col : str
        Name of dataframe column containing onset time latency(ies).
    onset_chan_col : str
        Name of dataframe column containing onset channel.
    offset_time_col : str
        Name of dataframe column containing offset time latency(ies).
    offset_chan_col : str
        Name of dataframe column containing offset channel.

    Returns:
    --------
    dataframe : class pandas.DataFrame
        Pandas DataFrame including columns 'accuracy', 'onset_correct',
        'offset_correct', 'onset_incorrect', 'offset_incorrect', 'onset_resp',
        'offset_resp', 'onset_non_resp', 'offset_non_resp'.
    """
    import pandas as pd

    accuracy_list = []
    onset_correct_list = []
    offset_correct_list = []
    onset_incorrect_list = []
    offset_incorrect_list = []

    onset_resp_list = []
    offset_resp_list = []
    onset_non_resp_list = []
    offset_non_resp_list = []

    for index, trial in dataframe.iterrows():

        if trial.isna()[stim_code_col]:
            stim = 'nan'
        else:
            stim = trial[stim_code_col]
        resp = trial[resp_code_col]
        resp_time = trial[resp_time_col]
        
        if trial.isna()[onset_time_col]:
            onset_time = []
        else: 
            onset_time = trial[onset_time_col]

        if trial.isna()[onset_chan_col]:
            onset_chan = []
        else:
            onset_chan = trial[onset_chan_col]

        if trial.isna()[offset_time_col]:
            offset_time = []
        else:
            offset_time = trial[offset_time_col]

        if trial.isna()[offset_chan_col]:
            offset_chan = []
        else:
            offset_chan = trial[offset_chan_col]

        accuracy, onset_correct, offset_correct, onset_incorrect, offset_incorrect = decode_trial_accuracy(stim, resp, onset_time, onset_chan,\
                                                                                                           offset_time, offset_chan, stim_resp_dict, resp_chan_dict)
        accuracy_list.append(accuracy)
        onset_correct_list.append(onset_correct)
        offset_correct_list.append(offset_correct)
        onset_incorrect_list.append(onset_incorrect)
        offset_incorrect_list.append(offset_incorrect)

        onset_resp, onset_non_resp, offset_resp, offset_non_resp = resp_onsets(resp, resp_time, onset_time, onset_chan, resp_chan_dict,\
                                                                               offset_time=offset_time, offset_chan=offset_chan)
        onset_resp_list.append(onset_resp)
        onset_non_resp_list.append(onset_non_resp)
        offset_resp_list.append(offset_resp)
        offset_non_resp_list.append(offset_non_resp)

    accuracy_df = pd.DataFrame(data={'accuracy' : accuracy_list,\
                                     'onset_correct' : onset_correct_list,\
                                     'offset_correct' : offset_correct_list,\
                                     'onset_incorrect' : onset_incorrect_list,\
                                     'offset_incorrect' : offset_incorrect_list,\
                                     'onset_resp' : onset_resp_list,\
                                     'offset_resp' : offset_resp_list,\
                                     'onset_non_resp' : onset_non_resp_list,\
                                     'offset_non_resp' : offset_non_resp_list,\
                                     })

    dataframe = pd.concat((dataframe, accuracy_df), axis=1)

    return dataframe

def decode_trial_accuracy(stim_code, resp_code, onset_time, onset_chan, offset_time, offset_chan,\
                          stim_resp_dict, resp_chan_dict):
    """Decodes trial response and channel (e.g., EMG) accuracy, assuming one stimulus and one response.

    Parameters:
    -----------
    stim_code: int | str
        Code of the stimulus event.
    resp_code: int | str
        Code of the response event.
    onset_time: list
        List containing time latencie(s) of EMG onsets.
    onset_chan: list
        List containing corresponding channel(s).
    offset_time: list
        List containing latencie(s) of EMG offsets.
    offset_chan: list
        List containing corresponding channel(s).
    stim_resp_dict: dict
        Dictionnary with possible stimulus codes as keys, and corresponding
        correct response codes as values.
    resp_chan_dict: dict
        Dictionnary with response codes as keys, and corresponding channels
        as values.

    Returns:
    --------
    accuracy : str
        Response accuracy: 'correct' or 'incorrect'.
    onset_correct : list
        List of correct onset time latencies.
    offset_correct : list
        List of correct offset time latencies.
    onset_incorrect : list
        List of incorrect onset time latencies.
    offset_incorrect : list
        List of incorrect offset time latencies.
    """

    if type(onset_chan) != type(onset_time):
        raise TypeError('onset_chan and onset_time must be of same type.')
    else:
        onset_chan = utilsfunc.in_list(onset_chan)
        onset_time = utilsfunc.in_list(onset_time)

    if type(offset_chan) != type(offset_time):
        raise TypeError('offset_chan and offset_time must be of same type.')
    else:
        offset_chan = utilsfunc.in_list(offset_chan)
        offset_time = utilsfunc.in_list(offset_time)

    if hasattr(stim_code, '__iter__') and not isinstance(stim_code, str):
        if len(stim_code) > 1:
            raise TypeError('Only one stimulus is assumed, stim_code must be str or int type')
        else:
            stim_code = utilsfunc.remove_list(stim_code)

    try:
        correct = stim_resp_dict[stim_code]
    except KeyError:
        raise KeyError('Stimulus code \'' + str(stim_code) + '\' is unknown, provide key entry for stimulus \'' + str(stim_code) + '\' in stimulus-response dictionnary (stim_resp_dict).')

    if hasattr(resp_code, '__iter__') and not isinstance(resp_code, str):
        if len(resp_code) > 1:
            raise TypeError('Only one response is assumed, resp_code must be str or int type')
        else:
            resp_code = utilsfunc.remove_list(resp_code)
    try:
        resp_chan_dict[correct]
    except KeyError:
        raise KeyError('Response code \'' + str(correct) + '\' is unknown, provide key entry for response \'' + str(resp_code) + '\' in response-EMGchannel dictionnary (resp_chan_dict).')

    onset_correct = []
    offset_correct = []
    onset_incorrect = []
    offset_incorrect = []

    if resp_code == correct:
        accuracy = 'correct'
    else:
        accuracy = 'incorrect'

    for c in range(len(onset_chan)):
        if onset_chan[c] == resp_chan_dict[correct]:
            onset_correct.append(onset_time[c])
        elif onset_chan[c] in resp_chan_dict.values():
            onset_incorrect.append(onset_time[c])
    for c in range(len(offset_chan)):
        if offset_chan[c] == resp_chan_dict[correct]:
            offset_correct.append(offset_time[c])
        elif offset_chan[c] in resp_chan_dict.values():
            offset_incorrect.append(offset_time[c])

    return accuracy, onset_correct, offset_correct, onset_incorrect, offset_incorrect

def classify_emg(dataframe, onset_correct_col='onset_correct', onset_incorrect_col='onset_incorrect', resp_time_col='resp_time', min_lat_difference=0,\
                 accuracy_col='accuracy', accuracy_chan_dict={'correct':'C','incorrect':'I'}, resp='R',\
                 print_warning=True):
    """Classifies EMG type of each trial of dataframe (e.g., 'PureC','IC').

    Reads each trial of the pandas DataFrame and determines onset sequence 
    based on 'sequenceOnset', then classifies EMG type using 'classifyTrialEMG'.
    Returns dataframe, with new columns containing: sequence of onset(s)
    (sequence_onset) and EMG type (EMGtype).

    Parameters:
    -----------
    dataframe : class pandas.DataFrame
        Pandas DataFrame with at least columns containing correct onset(s)
        latency(ies), incorrect onset(s) latency(ies), response latency(ies).
    onset_correct_col : str
        Name of dataframe column containing correct onset(s) latency(ies).
    onset_incorrect_col : str
        Name of dataframe column containing incorrect onset(s) latency(ies).
    resp_time_col : str
        Name of dataframe column containing response latency.
    min_lat_difference : float
        Minimal latency difference between successive onsets. If latency
        differnce between at least two onsets is smaller, 'onset_too_close' is
        returned.
    accuracy_col : str
        Name of dataframe column containing response accuracy.
    accuracy_chan_dict : dict
        Dictionnary with accuracy names (i.e., used in accuracy column) as keys,
        and code for corresponding channel onset(s) in sequence onset column.
    resp : str
        Code for response in sequence onset column.
    print_warning : bool
        If true, a warning is printed when 1) the response code is not in last
        position in onset sequence column, or 2) the last onset channel in onset
        sequence column does not match the response (e.g., accuracy is 'correct'
        but last onset is 'incorrect').

    Returns:
    --------
    dataframe : class pandas.DataFrame
        Pandas DataFrame including columns:
            - 'sequence_onset' in which each trial sequence of onset(s) and
              response is reported as a succession of codes defined by
              accuracy_chan_dict and resp,
            - 'emg_type' based on sequence_onset (from wich response code is 
              removed). The prefix 'pure' indicates that only one onset was 
              present, the prefix 'partial' indicates that no response was
              present.
    """
    import pandas as pd
    
    dataframe = sequence_onset(dataframe, onset_x_col=onset_correct_col, onset_y_col=onset_incorrect_col,\
                               name_chan_dict={'x':accuracy_chan_dict['correct'],'y':accuracy_chan_dict['correct'],'resp':resp},\
                               resp_time_col=resp_time_col, min_lat_difference=min_lat_difference)
    
    emg_type_list = []

    for index, trial in dataframe.iterrows():

        emg_type = classify_trial_emg(index, trial['sequence_onset'], trial[accuracy_col], accuracy_chan_dict=accuracy_chan_dict, resp=resp, print_warning=print_warning)
        emg_type_list.append(emg_type)
        
    new_df = pd.DataFrame(data={'emg_type': emg_type_list})
    dataframe = pd.concat((dataframe, new_df), axis=1)

    return dataframe


def classify_trial_emg(index, onset_sequence, accuracy, accuracy_chan_dict={'correct':'C','incorrect':'I'}, resp='R',\
                       print_warning=True):
    """Classifies EMG type (e.g., 'PC','IC').

    Parameters:
    -----------
    onset_sequence : str
        Trial's onset(s) sequence coded as a succesion of string (e.g., 'CIR').
    accuracy : str
        Trial's accuracy.
    accuracy_chan_dict: dict
        Dictionnary with accuracy names (i.e., corresponding to accuracy
        parameter) as keys, and code for corresponding channel onset(s) in
        onset_sequence.
    resp : str
        Code for response in onset_sequence.
    print_warning : bool
        If true, a warning is printed when 1) some events are observed in onset
        sequence column after the response, or 2) the last onset channel in onset
        sequence column does not match the response (e.g., accuracy is 'correct'
        but last onset is 'incorrect').

    Returns:
    --------
    emg_type : str
        EMG type name, corresponding to sequence_onset from wich response code
        is removed. The prefix 'pure' indicates that only one onset was
        present, the prefix 'partial' indicates that no response was present.
    """    
    codes = list(accuracy_chan_dict.values()) + utilsfunc.in_list(resp)
#    if any([])onset_sequence in unclass:
    if any([ _ not in codes for _ in onset_sequence]):
        emg_type = 'unclassified'
    
    # no EMG
    elif len(onset_sequence) == 0 :
        emg_type = 'no_emg'

    # no response was recorded
    elif onset_sequence.find(resp) == -1:
        emg_type = 'partial' + onset_sequence

    elif onset_sequence.find(resp) != len(onset_sequence) - 1:
        emg_type = 'unclassified'
        if print_warning is True:
            print(('Warning trial {} : some events occur after the response').format(index))
        
    else: 
        # remove response code
        onset_sequence = onset_sequence[:-1]
        
        # no onset (i.e., only resp)
        if len(onset_sequence) == 0:
            emg_type = 'no_emg'

        # checks whether last onset happens on responding channel
        elif onset_sequence[-1] != accuracy_chan_dict[accuracy]:
            emg_type = 'unclassified'
            if print_warning is True:
                print(('Warning trial {} : last onset channel does not match response').format(index))

        # only one onset before response
        elif len(onset_sequence) == 1:
            emg_type = 'pure' + onset_sequence
                
        else:
            emg_type = onset_sequence
    
    return emg_type
    

def sequence_onset(dataframe, onset_x_col='onset_correct', onset_y_col='onset_incorrect', name_chan_dict={'x':'C','y':'I','resp':'R'},\
                   resp_time_col='resp_time', min_lat_difference=0):
    """Sequences each trial onset(s) EMG type of each trial of dataframe (e.g., 'PC','IC').

    Reads each trial of the pandas DataFrame and defined the sequence of
    correct and incorrect onset(s) and response occurance using the function
    'sequenceOnsetTrial'. Returns dataframe, with new column containing the
    sequence onset (sequence_onset).

    Parameters:
    -----------
    dataframe : class pandas.DataFrame
        Pandas DataFrame with at least columnscontaining correct onset(s)
        latency(ies), incorrect onset(s) latency(ies), response latency(ies).
    onset_x_col : str
        Name of first dataframe column containing onset(s) latency(ies) to
        sequence.
    onset_y_col : str
        Name of second dataframe column containing onset(s) latency(ies) to
        sequence.
    name_chan_dict : dict
        Dictionnary defining codes representing each channel onset(s) in
        sequence onset. 'x' entry defines code for onset(s) from 'onset_x_col',
        'y' entry defines code for onset(s) from 'onset_y_col', 'resp' entry
        defines code for response.
    resp_time_col : str
        Name of dataframe column containing response latency.
    min_lat_difference : float
        Minimal latency difference between successive onsets. If one latency
        difference between at least two onsets is smaller,'onset_too_close' is
        returned.
    
    Returns:
    --------
    dataframe : class pandas.DataFrame
        Pandas DataFrame including columns 'sequence_onset' in which each trial
        sequence of onset(s) is reported as a succession of codes defined by
        name_chan_dict.
    """
    import pandas as pd
    
    seq_list = []

    for index, trial in dataframe.iterrows():

        if trial.isna()[onset_x_col]:
            onset_x = []
        else:
            onset_x = trial[onset_x_col]

        if trial.isna()[onset_y_col]:
            onset_y = []
        else:
            onset_y = trial[onset_y_col]

        if trial.isna()[resp_time_col]:
            resp_time = None
        else:
            resp_time = trial[resp_time_col]

        seq = sequence_onset_trial(onset_x, onset_y, resp_time, name_chan_dict={'x':'C','y':'I','resp':'R'}, min_lat_difference=min_lat_difference)
        seq_list.append(seq)
        
    new_df = pd.DataFrame(data={'sequence_onset': seq_list})
    dataframe = pd.concat((dataframe, new_df), axis=1)

    return dataframe


def sequence_onset_trial(onset_x, onset_y, resp_time, name_chan_dict={'x':'C','y':'I','resp':'R'}, min_lat_difference=0):
    """Sequences each trial onset(s) .

    Parameters:
    -----------
    onset_x : list
        Latency of onset(s) associated with response x.
    onset_y : list
        Latency of onset(s) associated with response y.
    resp_time : float
        Response latency, only one response is possible.
    name_chan_dict: dict
        Dictionnary defining codes representing each channel onset(s) in
        sequence onset. 'x' entry defines code for onset_x, 'y' entry defines
        the code for onset_y, and 'resp' entry defines the code for response.
    min_lat_difference : float
        Minimal latency difference between successive onsets. If latency 
        difference between at least two onsets is smaller, returns 
        'onset_too_close'.

    Returns:
    --------
    seq : str
        Sequence of x and y onsets, and response, as a succession of codes
        defined by name_chan_dict.
  """
    import numpy as np
    
    onset_x = utilsfunc.in_list(onset_x)
    onset_y = utilsfunc.in_list(onset_y)

    # no response was recorded
    if resp_time is None or np.isnan(resp_time) or resp_time == 'none' or (hasattr(resp_time, '__iter__') and len(resp_time) == 0):
        resp_time = []
    else: 
        resp_time = utilsfunc.in_list(resp_time)
    
    event_lat = np.array(onset_x + onset_y + resp_time)
    event_name = name_chan_dict['x'] * len(onset_x) + name_chan_dict['y'] * len(onset_y) + name_chan_dict['resp'] * len(resp_time)

    sort_idx = np.argsort(event_lat)
    if (np.diff(event_lat[sort_idx]) < min_lat_difference).any():
        seq='onset_too_close'
    else:
        seq = str()
        for idx in sort_idx:
            seq += event_name[idx]

    return seq


def resp_onsets(resp, resp_time, onset_time, onset_chan, resp_chan_dict, offset_time=[], offset_chan=[]):
    """Returns onsets and offsets, sorted as responding and non_responding.
    """
    import numpy as np
    
    # keeps only onset on desired channels
    idx = [i for i,c in enumerate(onset_chan) if c in resp_chan_dict.values()]
    # and transforms in ndarray
    onset_time = utilsfunc.in_array(onset_time)[idx]
    onset_chan = utilsfunc.in_array(onset_chan)[idx]
    # keeps only offset on desired channels
    idx = [i for i,c in enumerate(offset_chan) if c in resp_chan_dict.values()]
    # and transforms in ndarray
    offset_time = utilsfunc.in_array(offset_time)[idx]
    offset_chan = utilsfunc.in_array(offset_chan)[idx]

    onset_idx = np.arange(len(onset_time))
    try:
        # searchs the onset on responding channel closest but anterior to the response
        onset_resp_idx = np.where((onset_chan == resp_chan_dict[resp]) & ((onset_time - resp_time) < 0))[0]
    except KeyError:
        onset_resp_idx = []

    # if more than one, keep the one with longest latency (i.e., closest to response)
    if len(onset_resp_idx) > 1:
        onset_resp_idx = utilsfunc.in_list(onset_resp_idx[np.argmax(onset_time[onset_resp_idx])])
    onset_non_resp_idx = np.delete(onset_idx, onset_resp_idx)

    onset_resp = onset_time[onset_resp_idx]
    onset_non_resp = onset_time[onset_non_resp_idx]

    # same thing (almost) for offset
    offset_idx = np.arange(len(offset_time))
    if len(onset_resp) > 0:
        try:
            # searchs the offset on responding channel just responding onset
            offset_resp_idx = np.where((offset_chan == resp_chan_dict[resp]) & ((offset_time - onset_resp) > 0))[0]
        except KeyError:
            offset_resp_idx = []
    else:
        offset_resp_idx = []

    # if more than one, keep the one with shortest latency (i.e., closest to onset_resp)
    if len(offset_resp_idx) > 1:
        offset_resp_idx = utilsfunc.in_list(offset_resp_idx[np.argmin(offset_time[offset_resp_idx])])
    offset_non_resp_idx = np.delete(offset_idx, offset_resp_idx)

    offset_resp = offset_time[offset_resp_idx]
    offset_non_resp = offset_time[offset_non_resp_idx]

#    if len(offset_time) > 0:
#        offset_resp = offset_time[onset_resp_idx]
#        offset_non_resp = offset_time[np.invert(onset_resp_idx)]
#    else:
#        offset_resp = utilsfunc.in_array(offset_time)
#        offset_non_resp = utilsfunc.in_array(offset_time)

    return onset_resp, onset_non_resp, offset_resp, offset_non_resp


def get_channel_signal(dataframe, resp_chan_dict, data_signal_array, column_onset, column_resp='resp_code', keep_index=0, tmin=0, tmax=1, sf=None, remove_nan=True):
    """ Gets signal around defined event on the specified channel.
    
    Reads dataframe and gets signal from data_signal_array on interval defined 
    by tmin and tmax around events read in column_onset column of dataframe. 
    data_signal_array must be 2D, with channels on first dimension and time 
    samples on second dimension. For each trial, the channel on which signal 
    is extracted is given by resp_chan_dict using the response code read in 
    'column_resp' col. 
    
    Parameters:
    -----------
    dataframe : class pandas.DataFrame
        Pandas DataFrame with at least columns containing events around which 
        signal must be extracted and (i.e., EMG onsets) and events defining
        channel to read (i.e., response code).
    resp_chan_dict : dict
        Dictionnary with response codes as keys, and corresponding channels
        indices as values.
    data_signal_array : 2D array
        Channels x Time samples.
    column_onset : str
        Name of the dataframe column containing the events time instants around
        which signal must be read.
    column_resp :
        Name of the dataframe column containing event code defining channel index
        to read for event time instant (default 'resp_code' column). 
    keep_index : int
        Defines which event is kept if more than one event can be present in
        each dataframe row of column_onset (default 0, keeps the fisrt event).
    tmin : float
        Starting time (in s) to get signal (default 0).
    tmax : float
        Ending time (in s) to get signal (default 1s).
    sf: float
        Sampling frequency in data_signal_array, must be provided (default None).
    remove_nan : bool
        If True, skip nan rows in dataframe column_onset.

    Returns:
    -----------
    array : 
    
    """
    t0 = utilsfunc.remove_list(dataframe[column_onset], keep_index=keep_index)
    try:
        # we assume emg happened on emg channel corresponding to the response
        channels = [resp_chan_dict[r] for r in dataframe[column_resp]]
    except KeyError:
        print(('Response code is unknown, provide key entry in response-channel dictionnary (resp_chan_dict).'))
        raise 
    
    if remove_nan:
        import numpy as np
        t0 = np.array(t0)
        if (np.isnan(t0)).any():
            nan_lines = np.where(np.isnan(t0))[0]
            print(('nan values: reject trials {}').format(nan_lines))
            t0 = np.delete(t0,nan_lines)
            channels = np.delete(channels,nan_lines)
            
    array = get_signals(data_signal_array, t0, channels, tmin=tmin, tmax=tmax, sf=sf)

    return array

def get_signals(array, list_t0, list_channels, tmin=0, tmax=1, sf=None):

    import numpy as np
    
    events = evt.Events(time=list_t0, code=['trial']*len(list_t0), chan=list_channels, sf=sf)
    ep_evts = events.segment(pos_events0=range(events.nb_events()), tmin=tmin, tmax=tmax, print_epochs=False)

    last_trial = ep_evts.nb_trials()-1
    drop_trials = []
    while ep_evts.tmax.sample[last_trial] > array.shape[1]:
        drop_trials.append(last_trial)
        last_trial-=1
    if len(drop_trials)>0:
        print(('dataArray too short: reject trials {}').format(drop_trials))

    epochs_array = ep_evts.get_data(array)
    signal_array = np.empty((epochs_array.shape[0],epochs_array.shape[2]))
    for t in range(epochs_array.shape[0]):
        signal_array[t] = epochs_array[t,list_channels[t]]

    return signal_array

def getTrialsLists_fromContinuous(Evts, T0Codes, stimCodes, respCodes, \
                                  onsetCodes=['onset'], offsetCodes=['offset'], \
                                  otherCodes=[], endCodes=[],\
                                  minStim=1, maxStim=1, minResp=0, maxResp=1, tmax=None, \
                                  stop_after_response=False, stop_after_offset=True):
    """Reads trials of a continuous Events object.

    Reads continuous events and detects trials starts and ends. The start of a
    new trial is detected each time an event with code defined in 'T0Codes'
    is read. The end of trial is determined with 'idxEndEvent' function. End of
    trial is detected at the first occurence of any of the following: 1) a new
    'T0Codes' event, 2) a new stimulus event (as defined by stimCodes), when the
    maximum number of stimulus has been reached, 3) when the maximum number of
    responses occured (if stop_after_response is True) or after offset(s) of all
    bursts starting before last response (if stop_after_offset is True), 4) an
    'endCodes' event, 5), if none of 1), 2), 3) or 4), trial ends when maximum
    duration after previous T0Codes event is reached (if maximum duration tmax
    is provided). Returns lists containing each trial's index ('trialList'),
    stimulus event(s) ('stimList'), response event(s) ('respList'), onset event(s)
    ('onsetList'), offset event(s) ('offsetList'), any other event(s) 
    ('otherEvtList'). Each list index corresponds to a new trial, and trial order 
    corresponds to the Events object.

    Parameters:
    -----------
    Evts: class Events
        Events object to read.
    T0Codes: list | 1Darray
        Events codes for detecting trial start (e.g., use the the same events
        than for trial segmentation).
    stimCodes: list | 1Darray
        Codes of stimulus events (may also contain T0Codes).
    respCodes : list | 1Darray
        Codes of responses events.
    onsetCodes : list | 1Darray
        Codes of EMG onset events (default ['onset']).
    offsetCodes : list | 1Darray
        Codes of EMG offset events. Must be provided if 'stop_after_offset'
        is True (default ['offset']).
    otherCodes : list | 1Darray
        Codes of any other event that need to be stored. If equal to [], all
        other codes are stored, if None, no other code is stored (default []).
    endCodes : list | 1Darray
        Codes signaling the end of the trial (default []).
    minStim : int
        Minimum number of stimulus events (as defined in stimCodes) to read in
        the trial (default 1).
    maxStim : int
        Maximum number of stimulus events (as defined in stimCodes) to read in
        the trial (default 1).
    minResp : int
        Minimum number of response events (as defined in respCodes) to read in
        the trial (default 0).
    maxResp : int
        Maximim number of response events (as defined in respCodes) to read in
        the trial (default 1).
    tmax : float
        Maximum trial duration in second. If None, trial end is detected only if
        either 1), 2), 3) or 4) (see above) occured. If provided, end event is
        added at tmax after T0Codes event if none of 1), 2), 3) or 4) occured
        (default None).
    stop_after_resp : bool
        If True, the end trial marker is added after that the maximum number of
        responses occured (default False).
    stop_after_offset: bool
        If True, the end trial marker is added after offset(s) of all bursts
        starting before the last response occured (default True).

    Returns:
    --------
    trial_list : list
        List of trials indices.
    stim_list : list
        List of stimulus events.
    resp_list : list
        List of response events.
    onset_list : list
        List of onset events.
    offset_list : list
        List of offset events.
    otherEvt_list : list
        List of other events.
    """
    import warnings 
    warnings.simplefilter('default')
    warnings.warn('getTrialsLists_fromContinuous will be deleted in future version, use events_to_lists instead.', DeprecationWarning)
    import numpy as np

    if otherCodes == []:
        otherCodes = [c for c in np.unique(Evts.code) if c not in utilsfunc.in_list(stimCodes) + utilsfunc.in_list(respCodes) + utilsfunc.in_list(onsetCodes) + utilsfunc.in_list(offsetCodes)]

    startIdx = Evts.findEvents(code=T0Codes)
    trialIdx = np.hstack((startIdx, Evts.nbEvents()))

    nrTrials = len(startIdx)
    trial_list = [None] * nrTrials
    stim_list = [None] * nrTrials
    resp_list = [None] * nrTrials
    onset_list = [None] * nrTrials
    offset_list = [None] * nrTrials
    otherEvt_list = [None] * nrTrials

    for idx in range(len(startIdx)):

        code = Evts.code[trialIdx[idx]:trialIdx[idx+1]]
        chan = Evts.chan[trialIdx[idx]:trialIdx[idx+1]]
        sample = Evts.lat.sample[trialIdx[idx]:trialIdx[idx+1]]
        time = Evts.lat.time[trialIdx[idx]:trialIdx[idx+1]]

        endIdx = idx_end_event(code, chan, time, stimCodes, respCodes,\
                               onset_codes=onsetCodes, offset_codes=offsetCodes,\
                               min_stim=minStim, max_stim=maxStim, min_resp=minResp, max_resp=maxResp,\
                               tmax=tmax,\
                               stop_after_response=stop_after_response, stop_after_offset=stop_after_offset)

        EvtTrial = evt.Events(sample=sample[:endIdx+1], code=code[:endIdx+1], chan=chan[:endIdx+1], sf=Evts.sf)
        stimEvt, respEvt, onsetEvt, offsetEvt, otherEvt = read_trial_events(EvtTrial, stimCodes, respCodes, \
                                                                            onset_codes=onsetCodes, offset_codes=offsetCodes,\
                                                                            other_codes=otherCodes)

        trial_list[idx] = idx
        stim_list[idx] = stimEvt
        resp_list[idx] = respEvt
        onset_list[idx] = onsetEvt
        offset_list[idx] = offsetEvt
        otherEvt_list[idx] = otherEvt

    return trial_list, stim_list, resp_list, onset_list, offset_list, otherEvt_list
    
def getDataFrame_fromContinuous(Evts, T0Codes, stimCodes, respCodes, \
                                onsetCodes=['onset'], offsetCodes=['offset'], \
                                otherCodes=[], endCodes=[],\
                                minStim=1, maxStim=1, minResp=0, maxResp=1, tmax=None, \
                                stop_after_response=False, stop_after_offset=True):
    """ Gets pandas data frame from continuous events.

    Reads continuous events and detects trials starts and ends. The start of a
    new trial is detected each time an event with code defined in 'T0Codes'
    is read. The end of trial is determined with 'idxEndEvent' function. End of
    trial is detected at the first occurence of any of the following: 1) a new
    'T0Codes' event, 2) a new stimulus event (as defined by stimCodes), when the
    maximum number of stimulus has been reached, 3) when the maximum number of
    responses occured (if stop_after_response is True) or after offset(s) of all 
    bursts starting before the last response (if stop_after_offset is True), 
    4) an 'endCodes' event, 5), if none of 1), 2), 3) or 4), trial ends when 
    maximum duration after previous T0Codes event is reached (if maximum duration 
    tmax is provided). Returns a pandas dataframe containing each trial's stimulus 
    code, sample and time, response code and time, onset(s) code, channel and 
    time, offset(s) code, channel and time and any other event(s) code and time.
    Each dataframe row corresponds to a trial, and trial order corresponds to
    the Events object.

    Parameters:
    -----------
    Evts : class Events
        Events object to read.
    T0Codes : list | 1Darray
        Events codes for detecting trial start (e.g., use the the same events
        than for trial segmentation).
    stimCodes : list | 1Darray
        Codes of stimulus events (may also contain T0Codes).
    respCodes : list | 1Darray
        Codes of responses events.
    onsetCodes : list | 1Darray
        Codes of EMG onset events (default ['onset']).
    offsetCodes : list | 1Darray
        Codes of EMG offset events. Must be provided if 'stop_after_offset'
        is True (default ['offset']).
    otherCodes : list | 1Darray
        Codes of any other event that need to be stored. If equal to [], all
        other codes are stored, if None, no other code is stored (default []).
    endCodes : list | 1Darray
        Codes signaling the end of the trial (default []).
    minStim : int
        Minimum number of stimulus events (as defined in stimCodes) to read in
        the trial (default 1).
    maxStim : int
        Maximum number of stimulus events (as defined in stimCodes) to read in
        the trial (default 1).
    minResp : int
        Minimum number of response events (as defined in respCodes) to read in
        the trial (default 0).
    maxResp : int
        Maximim number of response events (as defined in respCodes) to read in
        the trial (default 1).
    tmax : float
        Maximum trial duration in second. If None, trial end is detected only if
        either 1), 2), 3) or 4) (see above) occured. If provided, end event is
        added at tmax after T0Codes event if none of 1), 2), 3) or 4) occured
        (default None).
    stop_after_resp : bool
        If True, the end trial marker is added after that the maximum number of
        responses occured (default False).
    stop_after_offset: bool
        If True, the end trial marker is added after offset(s) of all bursts
        starting before the last response occured (default True).

    Returns:
    --------
        Pandas DataFrame, with at least columns 'stim_code', 'stim_sample',
        'stim_time', 'resp_code', 'resp_sample', 'resp_time', 'onset_chan', 
        'onset_code', 'onset_sample', 'onset_time', 'offset_chan', 'offset_code',
        'offset_sample', 'offset_time'. Each line of the data frame contains 
        information corresponding to one trial.
    """
    import warnings 
    warnings.simplefilter('default')
    warnings.warn('getDataFrame_fromContinuous will be deleted in future version, use events_to_df instead.', DeprecationWarning)
    import pandas as pd

    trial_list, stim_list, resp_list, onset_list, offset_list, otherEvt_list = getTrialsLists_fromContinuous(Evts, T0Codes, stimCodes, respCodes, \
                                                                                                                   onsetCodes=onsetCodes, offsetCodes=offsetCodes,\
                                                                                                                   otherCodes=otherCodes, endCodes=endCodes,\
                                                                                                                   minStim=minStim, maxStim=maxStim, minResp=minResp, maxResp=maxResp, tmax=tmax,\
                                                                                                                   stop_after_response=stop_after_response,\
                                                                                                                   stop_after_offset=stop_after_offset)

    stimDF = evts_list_to_df(stim_list, 'stim', keep_code=True, keep_chan=False, keep_time=True, keep_sample=True)
    if maxStim <= 1:
        for col, i in stimDF.iteritems():
            stimDF[col] = utilsfunc.remove_list(i)

    respDF = evts_list_to_df(resp_list, 'resp', keep_code=True, keep_chan=False, keep_time=True, keep_sample=True)
    if maxResp <= 1:
        for col, i in respDF.iteritems():
            respDF[col] = utilsfunc.remove_list(i)

    onsetDF = evts_list_to_df(onset_list, 'onset', keep_code=True, keep_chan=False, keep_time=True, keep_sample=True)
    offsetDF = evts_list_to_df(offset_list, 'offset', keep_code=True, keep_chan=False, keep_time=True, keep_sample=True)
    otherEvtDF = evts_list_to_df(otherEvt_list, 'otherEvt', keep_code=True, keep_chan=False, keep_time=True, keep_sample=True)

    return pd.concat([stimDF, respDF, onsetDF, offsetDF, otherEvtDF], axis=1)
    
def getTrialsLists_fromEpochs(EpEvts, stimCodes, respCodes, \
                              onsetCodes=['onset'], offsetCodes=['offset'], \
                              otherCodes=[], endCodes=[],\
                              minStim=1, maxStim=1, minResp=0, maxResp=1, tmax=None, \
                              stop_after_response=False, stop_after_offset=True):

    """Reads trials of an Epoch Events object.

    Reads each trial of an epoch events. Detects trials ends with 'idxEndEvent' function
    to avoid overlapping trials. End of trial is detected at the first occurence
    of any of the following: 1) a new stimulus event (as defined by stimCodes), when
    the maximum number of stimulus has been reached, 2) when the maximum number
    of responses occured (if stop_after_response is True) or after offset(s) of
    all bursts starting before last response (if stop_after_offset is True),
    3) an 'endCodes' event, 4), at time = tmax, if none of 1), 2) or 3) occurs within
    maximum time duration after trial starts defined by tmax. Returns lists containing
    each trial's index ('trialList'), stimulus event(s) ('stimList'), response event(s)
    ('respList'), onset event(s) ('onsetList'), offset event(s) ('offsetList'),
    any other event(s) ('otherEvtList'). Each list index corresponds to a new trial,
    and trial order corresponds to the EpochEvents object.

    Parameters:
    -----------
    EpEvts : class EpochEvents
        EpochEvents object to read.
    stimCodes : list | 1Darray
        Codes of stimulus events.
    respCodes : list | 1Darray
        Codes of responses events.
    onsetCodes : list | 1Darray
        Codes of EMG onset events (default ['onset']).
    offsetCodes : list | 1Darray
        Codes of EMG offset events. Must be provided if 'stop_after_EMGoffset'
        is True (default ['offset']).
    otherCodes : list | 1Darray
        Codes of any other event that need to be stored. If equal to [], all
        other codes are stored, if None, no other code is stored (default []).
    endCodes : list | 1Darray
        Codes signaling the end of the trial (default []).
    minStim : int
        Minimum number of stimulus events (as defined in stimCodes) to read in
        the trial (default 1).
    maxStim : int
        Maximum number of stimulus events (as defined in stimCodes) to read in
        the trial (default 1).
    minResp : int
        Minimum number of response events (as defined in respCodes) to read in
        the trial (default 0).
    maxResp : int
        Maximim number of response events (as defined in respCodes) to read
        in the trial (default 1).
    tmax : float
        Maximum trial duration in second. If None, trial end is detected only
        if either 1), 2), 3) or 4) (see above) occured. If provided, end event
        is added at tmax after T0Codes event if none of 1), 2), 3) or 4)
        occured (default None).
    stop_after_resp : bool
        If True, the end trial marker is added after that the maximum number of
        responses occured (default False).
    stop_after_offset: bool
        If True, the end trial marker is added after offset(s) of all bursts
        starting before last response occured (default True).

    Returns:
    --------
    trialList : list
        List of trials indices.
    stimList : list
        List of stimulus events.
    respList : list
        List of response events.
    onsetList : list
        List of onset events.
    offsetList : list
        List of offset events.
    otherEvtList : list
        List of other events.
    """
    import warnings 
    warnings.simplefilter('default')
    warnings.warn('getTrialsLists_fromEpochs will be deleted in future version, use epochevents_to_lists instead.', DeprecationWarning)

    nrTrials = EpEvts.nbTrials()
    trialList = [None] * nrTrials
    stimList = [None] * nrTrials
    respList = [None] * nrTrials
    onsetList = [None] * nrTrials
    offsetList = [None] * nrTrials
    otherEvtList = [None] * nrTrials

    for idx in range(nrTrials):

        code = EpEvts.list_evts_trials[idx].code
        chan = EpEvts.list_evts_trials[idx].chan
        sample = EpEvts.list_evts_trials[idx].lat.sample
        time = EpEvts.list_evts_trials[idx].lat.time

        endIdx = idx_end_event(code, chan, time, stimCodes, respCodes, onset_codes=onsetCodes, offset_codes=offsetCodes,\
                               min_stim=minStim, max_stim=maxStim, min_resp=minResp, max_resp=maxResp,\
                               tmax=tmax,\
                               stop_after_response=stop_after_response, stop_after_offset=stop_after_offset)

        EvtTrial = evt.Events(sample=sample[:endIdx+1], time=time[:endIdx+1], code=code[:endIdx+1], chan=chan[:endIdx+1], sf=EpEvts.sf)
        stimEvt, respEvt, onsetEvt, offsetEvt, otherEvt = read_trial_events(EvtTrial, stimCodes, respCodes,\
                                                                            onset_codes=onsetCodes, offset_codes=offsetCodes,\
                                                                            other_codes=otherCodes)

        trialList[idx] = idx
        stimList[idx] = stimEvt
        respList[idx] = respEvt
        onsetList[idx] = onsetEvt
        offsetList[idx] = offsetEvt
        otherEvtList[idx] = otherEvt

    return trialList, stimList, respList, onsetList, offsetList, otherEvtList
    
def getDataFrame_fromEpochs(EpEvts, stimCodes, respCodes, \
                            onsetCodes=['onset'], offsetCodes=['offset'], \
                            otherCodes=[], endCodes=[],\
                            minStim=1, maxStim=1, minResp=0, maxResp=1, tmax=None, \
                            stop_after_response=False, stop_after_offset=True):
    """ Gets pandas data frame from epoch events.

    Reads each trial of epoch events. Detects trials ends using 'idxEndEvent' 
    function. End of trial is detected at the first occurence of any of the following: 
    1) a new stimulus event (as defined by stimCodes), when the maximum number
    of stimulus has been reached, 2) when the maximum number of responses occured
    (if stop_after_response is True) or after offset(s) of all bursts starting
    before the last response (if stop_after_offset is True), 3) an
    'endCodes' event, 4) at time = tmax, if none of 1), 2) or 3) occurs within
    maximum time duration after trial starts defined by tmax. Returns a pandas 
    dataframe containing each trial's stimulus code, sample and time, response 
    code and time, onset(s) code, channel and time, offset(s) code, channel 
    and time and any other event(s) code and time. Each dataframe row corresponds 
    to a trial, and trial order corresponds to the EpochEvents object.

    Parameters:
    -----------
    EpEvts : class EpochEvents
        EpochEvents object to read.
    stimCodes : list | 1Darray
        Codes of stimulus events (may also contain T0Codes).
    respCodes : list | 1Darray
        Codes of responses events.
    onsetCodes : list | 1Darray
        Codes of EMG onset events (default ['onset']).
    offsetCodes : list | 1Darray
        Codes of EMG offset events. Must be provided if 'stop_after_EMGoffset'
        is True (default ['offset']).
    otherCodes : list | 1Darray
        Codes of any other event that need to be stored. If equal to [], all
        other codes are stored, if None, no other code is stored (default []).
    endCodes : list | 1Darray
        Codes signaling the end of the trial (default []).
    minStim : int
        Minimum number of stimulus events (as defined in stimCodes) to read in
        the trial (default 1).
    maxStim : int
        Maximum number of stimulus events (as defined in stimCodes) to read in
        the trial (default 1).
    minResp : int
        Minimum number of response events (as defined in respCodes) to read in
        the trial (default 0).
    maxResp : int
        Maximim number of response events (as defined in respCodes) to read in
        the trial (default 1).
    tmax : float
        Maximum trial duration in second. If None, trial end is detected only if
        either 1), 2), 3) or 4) (see above) occured. If provided, end event is
        added at tmax after T0Codes event if none of 1), 2), 3) or 4) occured
        (default None).
    stop_after_resp : bool
        If True, the end trial marker is added after that the maximum number of
        responses occured (default False).
    stop_after_offset: bool
        If True, the end trial marker is added after offset(s) of all bursts
        starting before the last response occured (default True).

    Returns:
    --------
        Pandas DataFrame, with at least columns 'stim_code', 'stim_sample',
        'stim_time', 'resp_code', 'resp_sample', 'resp_time', 'onset_chan', 
        'onset_code', 'onset_sample', 'onset_time', 'offset_chan', 'offset_code',
        'offset_sample', 'offset_time'. Each line of the data frame contains 
        information corresponding to one trial.
    """
    import warnings 
    warnings.simplefilter('default')
    warnings.warn('getDataFrame_fromEpochs will be deleted in future version, use method epochevents_to_df instead.', DeprecationWarning)
    import pandas as pd

    trial_list, stim_list, resp_list, onset_list, offset_list, otherEvt_list = getTrialsLists_fromEpochs(EpEvts, stimCodes, respCodes, \
                                                                                                                 onsetCodes=onsetCodes, offsetCodes=offsetCodes,\
                                                                                                                 otherCodes=otherCodes, endCodes=endCodes,\
                                                                                                                 minStim=minStim, maxStim=maxStim, minResp=minResp, maxResp=maxResp, tmax=tmax,\
                                                                                                                 stop_after_response=stop_after_response,\
                                                                                                                 stop_after_offset=stop_after_offset)

    stimDF = evts_list_to_df(stim_list, 'stim', keep_code=True, keep_chan=False, keep_time=True, keep_sample=True)
    if maxStim <= 1:
        for col, i in stimDF.iteritems():
            stimDF[col] = utilsfunc.remove_list(i)

    respDF = evts_list_to_df(resp_list, 'resp', keep_code=True, keep_chan=False, keep_time=True, keep_sample=True)
    if maxResp <= 1:
        for col, i in respDF.iteritems():
            respDF[col] = utilsfunc.remove_list(i)

    onsetDF = evts_list_to_df(onset_list, 'onset', keep_code=True, keep_chan=False, keep_time=True, keep_sample=True)
    offsetDF = evts_list_to_df(offset_list, 'offset', keep_code=True, keep_chan=False, keep_time=True, keep_sample=True)
    otherEvtDF = evts_list_to_df(otherEvt_list, 'otherEvt', keep_code=True, keep_chan=False, keep_time=True, keep_sample=True)

    return pd.concat([stimDF, respDF, onsetDF, offsetDF, otherEvtDF], axis=1)
    
def decodeAccuracy(dataframe, stim_resp_dict, resp_chan_dict,\
                   stim_code_col='stim_code', resp_code_col='resp_code', resp_time_col='resp_time', \
                   onset_time_col='onset_time', onset_chan_col='onset_chan',\
                   offset_time_col='offset_time', offset_chan_col='offset_chan'):
    """Decodes response and channel (e.g., EMG) accuracy, assuming one stimulus and one response.

    Reads each trial of the pandas DataFrame and decodes accuracy using
    'decodeTrialAccuracy'. Returns dataframe, with new columns containing
    response and channel onset(s)/offset(s) accuracy. Channel onset(s)/offset(s)
    are also sorted as responding (associated with the response occurrence),
    and non responding (not associated with the response).

    Parameters:
    -----------
    dataframe : class pandas.DataFrame
        Pandas DataFrame with at least columns containing stimulus codes,
        response codes, onset latency(ies), onset channel(s), offset latency(ies),
        offset channel(s).
    stim_resp_dict : dict
        Dictionnary with possible stimulus codes as keys, and corresponding
        correct response codes as values.
    resp_chan_dict : dict
        Dictionnary with response codes as keys, and corresponding channels indices
        as values.
    stim_code_col : str
        Name of dataframe column containing stimulus code.
    resp_code_col : str
        Name of dataframe column containing response code.
    onset_time_col : str
        Name of dataframe column containing onset time latency(ies).
    onset_chan_col : str
        Name of dataframe column containing onset channel.
    offset_time_col : str
        Name of dataframe column containing offset time latency(ies).
    offset_chan_col : str
        Name of dataframe column containing offset channel.

    Returns:
    --------
    dataframe : class pandas.DataFrame
        Pandas DataFrame including columns 'accuracy', 'onset_correct',
        'offset_correct', 'onset_incorrect', 'offset_incorrect', 'onset_resp',
        'offset_resp', 'onset_non_resp', 'offset_non_resp'.
    """
    import warnings 
    warnings.simplefilter('default')
    warnings.warn('decodeAccuracy will be deleted in future version, use decode_accuracy instead.', DeprecationWarning)
    import pandas as pd

    accuracy_list = []
    onset_correct_list = []
    offset_correct_list = []
    onset_incorrect_list = []
    offset_incorrect_list = []

    onset_resp_list = []
    offset_resp_list = []
    onset_non_resp_list = []
    offset_non_resp_list = []

    for index, trial in dataframe.iterrows():

        if trial.isna()[stim_code_col]:
            stim = 'nan'
        else:
            stim = trial[stim_code_col]
        resp = trial[resp_code_col]
        resp_time = trial[resp_time_col]
        
        if trial.isna()[onset_time_col]:
            onset_time = []
        else: 
            onset_time = trial[onset_time_col]

        if trial.isna()[onset_chan_col]:
            onset_chan = []
        else:
            onset_chan = trial[onset_chan_col]

        if trial.isna()[offset_time_col]:
            offset_time = []
        else:
            offset_time = trial[offset_time_col]

        if trial.isna()[offset_chan_col]:
            offset_chan = []
        else:
            offset_chan = trial[offset_chan_col]

        accuracy, onset_correct, offset_correct, onset_incorrect, offset_incorrect = decode_trial_accuracy(stim, resp, onset_time, onset_chan,\
                                                                                                           offset_time, offset_chan, stim_resp_dict, resp_chan_dict)
        accuracy_list.append(accuracy)
        onset_correct_list.append(onset_correct)
        offset_correct_list.append(offset_correct)
        onset_incorrect_list.append(onset_incorrect)
        offset_incorrect_list.append(offset_incorrect)

        onset_resp, onset_non_resp, offset_resp, offset_non_resp = resp_onsets(resp, resp_time, onset_time, onset_chan, resp_chan_dict,\
                                                                               offset_time=offset_time, offset_chan=offset_chan)
        onset_resp_list.append(onset_resp)
        onset_non_resp_list.append(onset_non_resp)
        offset_resp_list.append(offset_resp)
        offset_non_resp_list.append(offset_non_resp)

    accuracyDf = pd.DataFrame(data={'accuracy' : accuracy_list,\
                                    'onset_correct' : onset_correct_list,\
                                    'offset_correct' : offset_correct_list,\
                                    'onset_incorrect' : onset_incorrect_list,\
                                    'offset_incorrect' : offset_incorrect_list,\
                                    'onset_resp' : onset_resp_list,\
                                    'offset_resp' : offset_resp_list,\
                                    'onset_non_resp' : onset_non_resp_list,\
                                    'offset_non_resp' : offset_non_resp_list,\
                                    })

    dataframe = pd.concat((dataframe, accuracyDf), axis=1)

    return dataframe
    
def classifyEMG(dataframe, onset_correct_col='onset_correct', onset_incorrect_col='onset_incorrect', resp_time_col='resp_time', min_lat_difference=0,\
                accuracy_col='accuracy', accuracy_chan_dict={'correct':'C','incorrect':'I'}, resp='R',\
                print_warning=True):
    """Classifies EMG type of each trial of dataframe (e.g., 'PureC','IC').

    Reads each trial of the pandas DataFrame and determines onset sequence 
    based on 'sequenceOnset', then classifies EMG type using 'classifyTrialEMG'.
    Returns dataframe, with new columns containing: sequence of onset(s)
    (sequence_onset) and EMG type (EMGtype).

    Parameters:
    -----------
    dataframe : class pandas.DataFrame
        Pandas DataFrame with at least columns containing correct onset(s)
        latency(ies), incorrect onset(s) latency(ies), response latency(ies).
    onset_correct_col : str
        Name of dataframe column containing correct onset(s) latency(ies).
    onset_incorrect_col : str
        Name of dataframe column containing incorrect onset(s) latency(ies).
    resp_time_col : str
        Name of dataframe column containing response latency.
    min_lat_difference : float
        Minimal latency difference between successive onsets. If latency
        differnce between at least two onsets is smaller, 'onset_too_close' is
        returned.
    accuracy_col : str
        Name of dataframe column containing response accuracy.
    accuracy_chan_dict : dict
        Dictionnary with accuracy names (i.e., used in accuracy column) as keys,
        and code for corresponding channel onset(s) in sequence onset column.
    resp : str
        Code for response in sequence onset column.
    print_warning : bool
        If true, a warning is printed when 1) the response code is not in last
        position in onset sequence column, or 2) the last onset channel in onset
        sequence column does not match the response (e.g., accuracy is 'correct'
        but last onset is 'incorrect').

    Returns:
    --------
    dataframe : class pandas.DataFrame
        Pandas DataFrame including columns:
            - 'sequence_onset' in which each trial sequence of onset(s) and
              response is reported as a succession of codes defined by
              accuracy_chan_dict and resp,
            - 'emg_type' based on sequence_onset (from wich response code is 
              removed). The prefix 'pure' indicates that only one onset was 
              present, the prefix 'partial' indicates that no response was
              present.
    """
    import warnings 
    warnings.simplefilter('default')
    warnings.warn('classifyEMG will be deleted in future version, use classify_emg instead.', DeprecationWarning)
    import pandas as pd
    
    dataframe = sequence_onset(dataframe, onset_X_col=onset_correct_col, onset_Y_col=onset_incorrect_col,\
                              name_chan_dict={'X':accuracy_chan_dict['correct'],'Y':accuracy_chan_dict['correct'],'resp':resp},\
                              resp_time_col=resp_time_col, min_lat_difference=min_lat_difference)
    
    emg_type_list = []

    for index, trial in dataframe.iterrows():

        emg_type = classify_trial_emg(index, trial['sequence_onset'], trial[accuracy_col], accuracy_chan_dict=accuracy_chan_dict, resp=resp, print_warning=print_warning)
        emg_type_list.append(emg_type)
        
    newDf = pd.DataFrame(data={'emg_type': emg_type_list})
    dataframe = pd.concat((dataframe, newDf), axis=1)

    return dataframe