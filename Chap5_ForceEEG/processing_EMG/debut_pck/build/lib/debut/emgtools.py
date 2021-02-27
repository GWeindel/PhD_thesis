# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:55:42 2016

@author: Laure Spieser and Boris Burle
Laboratoire de Neurosciences Cognitives
UMR 7291, CNRS, Aix-Marseille Université
3, Place Victor Hugo
13331 Marseille cedex 3
"""


def set_log_file(fname, overwrite=False):
    """Set the log to print to a file. 
    Largely inspired by the logging system of mne-python
    ###COPY FROM MNE###
    
    Set the log to print to a file using the 'logging' module. For more information
    about logging module, see:
        https://docs.python.org/dev/howto/logging.html

    Parameters
    ----------
    fname : str
        Filename of the log to print to.
    overwrite : bool
        Overwrite the log file (if it exists) if True. If False, statements
        will be appended to the log (default). 
        FIXME: define the kwargs used, for example "logging.INFO". 
        Laure: I don't think we should define logging.INFO, it is not a kwarg of our function (it can't be passed as a kwarg to set_log_file). I add 'using logging module'
    """
    import logging
    logger = logging.getLogger('emg_tools')
    logger.setLevel(logging.INFO)
    print(logging.INFO)
    handlers = logger.handlers
    for h in handlers:
        # only remove our handlers (get along nicely with nose)
        if isinstance(h, (logging.FileHandler, logging.StreamHandler)):
            if isinstance(h, logging.FileHandler):
                h.close()
            logger.removeHandler(h)
    mode = 'w' if overwrite else 'a'
    lh = logging.FileHandler(fname, mode=mode)
    logger.addHandler(lh)

def tkeo(array):
    """
    Returns the Teaker-Kayser transformation of the input signal.

    Parameters:
    -----------
    array : 1D array
        Signal on which Teager-Kaiser energy operator should be
        computed. If it has more than one dimension, a warning
        is issued and an array of zeros is returned.

    Returns:
    --------
    tkeo : array of same dimension as imput array

    References:
    -----------
    Li X, Zhou P, Aruin AS (2007) Teager–Kaiser energy operation of surface EMG
    improves muscle activity onset detection. Ann Biomed Eng 35:1532–1538.
    
    Solnik S, Rider P, Steinweg K, DeVita P, HortobaÂgyi T (2010) Teager-Kaiser
    energy operator signal conditioning improves EMG onset detection. Eur J Appl
    Physiol. 110(3):489-98. doi: 10.1007/s00421-010-1521-8

    Tenan MS, Tweedell AJ, Haynes CA (2017) Analysis of statistical and standard
    algorithms for detecting muscle onset with surface electromyography.
    PLoS One. 12(5):e0177312. doi: 10.1371/journal.pone.0177312.
    """
    import numpy as np
    if len(array.shape) > 1:
        print("only 1D vectors are accepted")
        return np.zeros(array.shape)
    else:
        tmp = np.zeros(array.shape)
        tmp[1:-1] = array[1:-1]**2 - (array[:-2] * array[2:])
        tmp[0] = tmp[1]
        tmp[-1] = tmp[-2]
        return tmp

def findTimes(timeValue, timeSerie):
    """
    Returns index such that timeSerie[index] is the closest value
    of timeSerie to timeValue.

    Parameters:
    -----------
    timeValue : float
    timeSerie : 1D array | list

    Returns:
    --------
    index : int

    """
    import warnings 
    warnings.simplefilter('default')
    warnings.warn('findTimes will be deleted in future version, use find_times instead.', DeprecationWarning)
    
    return find_times(timeValue, timeSerie)

def find_times(time_value, time_serie):
    """
    Returns index such that timeSerie[index] is the closest value
    of timeSerie to timeValue.

    Parameters:
    -----------
    time_value : float
    time_serie : 1D array | list

    Returns:
    --------
    index : int

    """
    import numpy as np
    return np.abs(time_serie - time_value).argmin()

def filtFilter(array, n=3, sf=None, cutoff=None, filter_name=None):
    """
    FIXME: sg.filtfilt is better than sg.lfilter especially for low-pass filtering so 
    I think we should use it instead of hfilter and lfilter, agreed ?

    """
    import warnings 
    warnings.simplefilter('default')
    warnings.warn('filtFilter will be deleted in future version, use filtfilter instead.', DeprecationWarning)
    return filtfilter(array, N=n, sf=sf, cutoff=cutoff, filter_name=filter_name)

def filtfilter(array, n=3, sf=None, cutoff=None, filter_name=None):
    """
    FIXME: sg.filtfilt is better than sg.lfilter especially for low-pass filtering so 
    I think we should use it instead of hfilter and lfilter, agreed ?

    """
    from numpy import zeros
    import scipy.signal as sg
    if sf is None:
        raise TypeError('Provide suitable value for sampling frequency (sf).')
    if filter_name is None:
        raise TypeError('Provide suitable value for filter_name (\'lowpass\' or \'highass\').')
    Wn = 2. * cutoff / sf
    b, a = sg.butter(n, Wn, filter_name)
    out = zeros(array.shape)
    out = sg.filtfilt(b, a, array)
    return out

def hpFilter(array, n=3, sf=None, cutoff=10):
    """Apply high pass filter to input signal.

    Parameters:
    -----------
    array : array_like (1D or 2D)
        Input signal to filter. For 2D arrays, must contain channels on axis 0
        and time samples on axis 1.
    n : int
        The order of the filter (Default 3).
    sf : float
        Sampling frequency.
    cutoff : int | float
        Cutoff frequency.

    Returns:
    --------
    out : array
        Filtered signal

    """
    import warnings 
    warnings.simplefilter('default')
    warnings.warn('hpFilter  will be deleted in future version, use hpfilter instead.', DeprecationWarning)
    return hpfilter(array, n=n, sf=sf, cutoff=cutoff)

def hpfilter(array, n=3, sf=None, cutoff=10):
    """Apply high pass filter to input signal.

    Parameters:
    -----------
    array : array_like (1D or 2D)
        Input signal to filter. For 2D arrays, must contain channels on axis 0
        and time samples on axis 1.
    N : int
        The order of the filter (Default 3).
    sf : float
        Sampling frequency.
    cutoff : int | float
        Cutoff frequency.

    Returns:
    --------
    out : array
        Filtered signal

    """
    from numpy import zeros
    import scipy.signal as sg
    if sf is None:
        raise TypeError('Provide suitable value for sampling frequency (sf).')
    Wn = 2. * cutoff / sf
    b, a = sg.butter(n, Wn, "highpass")
    out = zeros(array.shape)
    out = sg.lfilter(b, a, array)
    return out

def lpFilter(array, n=3, sf=None, cutoff=150):  
    """Apply low pass filter to input signal.

    Parameters:
    -----------
    array : array_like
        Input signal to filter.For 2D arrays, must contain channels on axis 0
        and time samples on axis 1.
    n : int
        The order of the filter (Default 3).
    sf : float
        Sampling frequency.
    cutoff : float
        Cutoff frequency.

    Returns:
    --------
    out : array
        Filtered signal

    """
    import warnings 
    warnings.simplefilter('default')
    warnings.warn('lpFilter  will be deleted in future version, use lpfilter instead.', DeprecationWarning)
    return lpfilter(array, n=n, sf=sf, cutoff=cutoff)
    
def lpfilter(array, n=3, sf=None, cutoff=150):  
    """
    Apply low pass filter to input signal.

    Parameters:
    -----------
    array : array_like
        Input signal to filter.For 2D arrays, must contain channels on axis 0
        and time samples on axis 1.
    N : int
        The order of the filter (Default 3).
    sf : float
        Sampling frequency.
    cutoff : float
        Cutoff frequency.

    Returns:
    --------
    out : array
        Filtered signal

    """
    from numpy import zeros
    import scipy.signal as sg
    Wn = 2. * cutoff / sf
    b, a = sg.butter(n, Wn, "lowpass")
    out = zeros(array.shape)
    out = sg.lfilter(b, a, array)
    return out

def notchFilter(array, n=3, sf=None, freq=50, band=10, ripple=None, filter_type='butter'):
    """
    Apply notch filter to input signal.

    Parameters:
    -----------
    array : array_like
        Input signal to filter. For 2D arrays, must contain channels on axis 0
        and time samples on axis 1.
    n : int
        The order of the filter (default 3). 
    sf : float
        Sampling frequency
    freq : float
        The centerline frequency to be filtered.
    band : float
        The bandwidth around the centerline frequency that you wish to
        filter.
    ripple :
        The maximum passband ripple that is allowed in db
    filter_type : 'butter' | 'bessel' | 'cheby1' | 'cheby2' | 'ellip'
        The filter type to use (default 'butter').

    Returns:
    --------
    out : array
        Filtered signal
    """
    import warnings 
    warnings.simplefilter('default')
    warnings.warn('notchFilter will be deleted in future version, use notch_filter instead.', DeprecationWarning)
    return notch_filter(array, n=n, sf=sf, freq=freq, band=band, ripple=ripple, filter_type=filter_type)

def notch_filter(array, n=3, sf=None, freq=50, band=10, ripple=None, filter_type='butter'):
    """
    Apply notch filter to input signal.

    Parameters:
    -----------
    array : array_like
        Input signal to filter. For 2D arrays, must contain channels on axis 0
        and time samples on axis 1.
    N : int
        The order of the filter (default 3). 
    sf : float
        Sampling frequency
    freq : float
        The centerline frequency to be filtered.
    band : float
        The bandwidth around the centerline frequency that you wish to
        filter.
    ripple :
        The maximum passband ripple that is allowed in db
    filter_type : 'butter' | 'bessel' | 'cheby1' | 'cheby2' | 'ellip'
        The filter type to use (default 'butter').

    Returns:
    --------
    out : array
        Filtered signal
    """
    import scipy.signal as sg
    nyq = sf / 2.0
    low = freq - band / 2.0
    high = freq + band / 2.0
    low = low / nyq
    high = high / nyq
    b, a = sg.iirfilter(n, [low, high], rp=ripple, btype='bandstop',
                        analog=False, ftype=filter_type)
    out = sg.lfilter(b, a, array)
    return out

def movingAverage(data_trial, window_size):
    """
    Computes signal moving average with specified window size.

    Parameters:
    -----------
    data_trial : 1D array_like
        Input signal.
    window_size : int
        Size of the moving average window in data samples.

    Returns:
    --------
    1D signal array after moving average.

    """
    import warnings 
    warnings.simplefilter('default')
    warnings.warn('movingAverage will be deleted in future version, use moving_avg instead.', DeprecationWarning)
    return moving_avg(data_trial, window_size)

def moving_avg(data_trial, window_size,centered=True):
    """
    Computes signal moving average with specified window size.

    Parameters:
    -----------
    data_trial : 1D array_like
        Input signal.
    window_size : int
        Size of the moving average window in data samples.

    Returns:
    --------
    1D signal array after moving average.

    """
    import numpy as np
    if centered:
        data_trial = np.insert(data_trial,0,np.ones(int(window_size - int(window_size)//2))*data_trial[0])
        data_trial = np.append(data_trial,np.ones(int(window_size)//2)*data_trial[-1],0)
    else:
        data_trial = np.insert(data_trial,-1,np.ones(window_size)*data_trial[-1])
    cs = np.cumsum(data_trial)
    return (cs[window_size:] - cs[:-window_size]) / float(window_size)

def integratedProfile(data_trial, times):
    """
    Return signal integrated profile.

    Compute the signal integrated profile, defined as the difference
    between the empirical cumulative sum of a dataset, and its uniform
    equivalent (straigth line).

    Parameters:
    -----------
    data_trial : 1D array
        Input signal
    times : 1D array
        Time instants

    Returns:
    --------
    d : 1D array
        Integrated profile

    """
    import warnings 
    warnings.simplefilter('default')
    warnings.warn('integratedProfile will be deleted in future version, use integrated_profile instead.', DeprecationWarning)
    return integrated_profile(data_trial, times)

def integrated_profile(data_trial, times):
    """
    Return signal integrated profile.

    Compute the signal integrated profile, defined as the difference
    between the empirical cumulative sum of a dataset, and its uniform
    equivalent (straigth line).

    Parameters:
    -----------
    data_trial : 1D array
        Input signal
    times : 1D array
        Time instants

    Returns:
    --------
    d : 1D array
        Integrated profile

    """
    import numpy as np
    cs = np.abs(data_trial).cumsum()
    a = cs[-1] / (times[-1] - times[0])
    l = a * times
    d = cs - l
    return d

def getOnsetIntegratedProfile(data_trial, times, samples='all', movingAvg_size = 1):
    """
    Return min and max of signal integrated profile.

    Compute the signal integrated profile and return the position of min
    and max, assumed to correspond to the onset and offset of activity.

    Parameters:
    -----------
    data_trial : 1D array
        Input signal.
    times : 1D array
        Time instants.
    samples : 1D array
        Time samples in which min and max are defined. If samples = 'all',
        all samples are used (default is 'all').
    movingAvg_size : int
        Defines the window size (in samples) for the moving average applied 
        on integrated profile before to define min and max. If movingAvg_size 
        is equal to 1, this has no effect (default is 1).

    Returns:
    --------
    onset,offset : int
        Indices of data_trial corresponding to EMG onset and offset

    References:
    -----------
    
    Liu J, Liu Q (2016) Use of the integrated profile for voluntary muscle
    activity detection using EMG signals with spurious background spikes: A
    study with incomplete spinal cord injury. Biomed Signal Process Control
    24:19-24.
    
    Santello M, McDonagh MJN (1998) The control of timing and amplitude of EMG
    activity in landing movements in humans. Exp Physiol 83:857-874.
    """
    import warnings 
    warnings.simplefilter('default')
    warnings.warn('getOnsetIntegratedProfile will be deleted in future version, use get_onset_ip instead.', DeprecationWarning)
    return get_onset_ip(data_trial, times, samples=samples, moving_avg_size=movingAvg_size)

def get_onset_ip(data_trial, times, samples='all', moving_avg_size = 1):
    """
    Return min and max of signal integrated profile.

    Compute the signal integrated profile and return the position of min
    and max, assumed to correspond to the onset and offset of activity.

    Parameters:
    -----------
    data_trial : 1D array
        Input signal.
    times : 1D array
        Time instants.
    samples : 1D array
        Time samples in which min and max are defined. If samples = 'all',
        all samples are used (default is 'all').
    moving_avg_size : int
        Defines the window size (in samples) for the moving average applied 
        on integrated profile before to define min and max. If movingAvg_size 
        is equal to 1, this has no effect (default is 1).

    Returns:
    --------
    onset,offset : int
        Indices of data_trial corresponding to EMG onset and offset

    References:
    -----------
    
    Liu J, Liu Q (2016) Use of the integrated profile for voluntary muscle
    activity detection using EMG signals with spurious background spikes: A
    study with incomplete spinal cord injury. Biomed Signal Process Control
    24:19-24.
    
    Santello M, McDonagh MJN (1998) The control of timing and amplitude of EMG
    activity in landing movements in humans. Exp Physiol 83:857-874.
    """
    import numpy as np
    if (type(samples) is str) and samples == 'all' : samples = np.arange(len(data_trial))

    ip = integrated_profile(data_trial, times)
    ip = moving_avg(ip[samples],moving_avg_size)
    onset = np.argmin(ip)
    offset = onset + np.argmax(ip[onset:])
    onset = onset + samples[0]
    offset = offset + samples[0]
    return onset, offset

def getGlobalVariance(data_epochs, times, tmin=None, tmax=None, useTkeo=True,
                      corVar=None):
    """
    Return mean and standard deviation of the input signal between
    tmin and tmax.

    Parameters:
    -----------
    data_epochs : 3D array
        First dimension should be trials, second dimension shoud be
        channels, and last dimension is times
    times : 1D array
        Time instants
    tmin : float | None
        Start time for mean/variance computation. If None start at
        first sample (default None).
    tmax : float | None
        End time for mean/variance computation. If None end at time 0
        (default None).
    useTkeo : bool
        If True, compute mean and variance on the Teaker-Kayser
        transformation of the input signal (default True).
    corVar : float | None
        If float, correct for outliers. The return mean and variance are
        computed on all sample signals whose abs(z-score) < corVar
        (default None).

    Returns:
    --------
    mBl,sBl : list
        Mean and standard deviation on each channel.

    """
    import warnings 
    warnings.simplefilter('default')
    warnings.warn('getGlobalVariance will be deleted in future version, use global_var instead.', DeprecationWarning)
    return global_var(data_epochs, times, tmin=tmin, tmax=tmax, use_tkeo=useTkeo, cor_var=corVar)

def global_var(data_epochs, times, tmin=None, tmax=None, use_tkeo=True, cor_var=None):
    """Return mean and standard deviation of the input signal between
    tmin and tmax.

    Parameters:
    -----------
    data_epochs : 3D array
        First dimension should be trials, second dimension shoud be
        channels, and last dimension is times
    times : 1D array
        Time instants
    tmin : float | None
        Start time for mean/variance computation. If None start at
        first sample (default None).
    tmax : float | None
        End time for mean/variance computation. If None end at time 0
        (default None).
    use_tkeo : bool
        If True, compute mean and variance on the Teaker-Kayser
        transformation of the input signal (default True).
    cor_var : float | None
        If float, correct for outliers. The return mean and variance are
        computed on all sample signals whose abs(z-score) < corVar
        (default None).

    Returns:
    --------
    mbsl,stbsl : list
        Mean and standard deviation on each channel.

    """
    import numpy as np
    if len(data_epochs.shape) != 3:
        print("Data do not have the correct dimensions. Exit !")
        exit()
    mbsl = []
    stbsl = []
    if tmin == None:
        tmin = 0
    if tmax == None:
        tmax = find_times(0, times)
    n_chan = data_epochs.shape[1]
    n_epochs = data_epochs.shape[0]
    for chan in range(n_chan):
        if use_tkeo:
            tkbsl = np.zeros((n_epochs, tmax - tmin))
            for e in range(n_epochs):
                tkbsl[e, :] = tkeo(data_epochs[e, chan, tmin:tmax])
            fbsl = tkbsl.flatten()
        else:
            fbsl = data_epochs[:, chan, tmin:tmax].flatten()
        # correct for outliers
        if cor_var != None:
            mbsl.append(
                fbsl[np.abs((fbsl - fbsl.mean()) / fbsl.std()) < cor_var].mean())
            stbsl.append(
                fbsl[np.abs((fbsl - fbsl.mean()) / fbsl.std()) < cor_var].std())
        else:
            mbsl.append(fbsl.mean())
            stbsl.append(fbsl.std())
    return mbsl, stbsl

def detectorVariance(data_trial, times, th = 3.5, time_limit = .025, 
                     min_samples = 3, varying_min = 1, mBl = None, stBl = None,
                     sf = None, use_derivative = False, movingAvg_size = 1):
    """
    Return the indices of time windows containing signal exceeding the 
    specified variance.

    Define the intervals during which one or several portions of input
    signal amplitude exceeds mBl + (th * stBl). Intervals are merged when 
    the delay between the interval offset and the next interavl onset is
    inferior to the time limit. Intervals containing less than the defined
    minimal number of samples are excluded.
    For EMG, this gives time windows where EMG bursts are supposed to occur.

    Parameters:
    -----------
    data_trial : 1D array
        Input signal. Note that data_trial signal is always rectified before
        detection.
    times : 1D array
        Time instants.
    th : float
        Threshold to multiply by stBl (default 3.5).
    time_limit : float
        Lowest time delay (in second) between bursts. Bursts intervals are 
        merged When less time separates interval offset from next interval 
        onset (default 25ms).
    min_samples : float
        Minimal number of data points exceeding mBl + (th * stBl) in signal 
        intervals. If less points are present, interval is excluded. 
        This is applied after merging intervals separated by less than the 
        time limit (default 3).
    varying_min : float
        If True, the minimal number of data points exceeding threshold required 
        to define a burst is increased in noisy trials. The noise is estimated 
        by the frequency of "small" bursts (freq_small), i.e., bursts containing
        less than twice the defined minimal number of samples. The minimal 
        number of samples is then increased up to min_samples + 
        (freq_small * varying_min) (Default is 1, set to 0 to have no 
        change of min_samples).
    mBl : float | None
        If None, use the mean baseline value of the trial (between first
        time sample and time sample 0). To use global mBl, use 
        getGlobalVariance function, applied on same data type as input 
        signal (e.g., raw EMG, teager-kayser...)(default None).
    stBl : float | None
        If None, use the baseline standard deviation of the trial
        (between first time sample and time sample 0). To use global stBl,
        apply getGlobalVariance on the same data type as input signal 
        (e.g., raw EMG, teager-kayser...)(default None).
    sf : float
        Sampling frequency.
    UseDerivative : bool
        If True signal derivative is used to determine offset (default False).
    movingAvg_size : int
        Defines the window size (in samples) for the moving average applied 
        on signal before detection. If movingAvg_size is equal to 1, this has
        no effect (default is 1).

    Returns:
    --------
    2D array
        Each line corresponds to one burst interval. First column contains 
        onset, second column contains offset.
    """
    import warnings 
    warnings.simplefilter('default')
    warnings.warn('detectorVariance will be deleted in future version, use detector_var instead.', DeprecationWarning)
    return detector_var(data_trial, times, th=th, time_limit=time_limit, 
                        min_samples=min_samples, varying_min=varying_min, 
                        mbsl=mBl, stbsl=stBl, sf=sf, use_derivative=use_derivative,
                        moving_avg_size=movingAvg_size)

def detector_var(data_trial, times, th=3.5, time_limit=.025, 
                 min_samples=3, varying_min=1, mbsl=None, stbsl=None,
                 sf=None, use_derivative=False, use_derivative_onset=False, moving_avg_size=1,moving_avg_size_onset=1):
    """Return the indices of time windows containing signal exceeding the 
    specified variance.

    Define the intervals during which one or several portions of input
    signal amplitude exceeds mbsl + (th * stbsl). Intervals are merged when 
    the delay between the interval offset and the next interavl onset is
    inferior to the time limit. Intervals containing less than the defined
    minimal number of samples are excluded.
    For EMG, this gives time windows where EMG bursts are supposed to occur.

    Parameters:
    -----------
    data_trial : 1D array
        Input signal. Note that data_trial signal is always rectified before
        detection.
    times : 1D array
        Time instants.
    th : float
        Threshold to multiply by stbsl (default 3.5).
    time_limit : float
        Lowest time delay (in second) between bursts. Bursts intervals are 
        merged When less time separates interval offset from next interval 
        onset (default 25ms).
    min_samples : float
        Minimal number of data points exceeding mbsl + (th * stbsl) in signal 
        intervals. If less points are present, interval is excluded. 
        This is applied after merging intervals separated by less than the 
        time limit (default 3).
    varying_min : float
        If True, the minimal number of data points exceeding threshold required 
        to define a burst is increased in noisy trials. The noise is estimated 
        by the frequency of "small" bursts (freq_small), i.e., bursts containing
        less than twice the defined minimal number of samples. The minimal 
        number of samples is then increased up to min_samples + 
        (freq_small * varying_min) (Default is 1, set to 0 to have no 
        change of min_samples).
    mbsl : float | None
        If None, use the mean baseline value of the trial (between first
        time sample and time sample 0). To use global mbsl, use 
        getGlobalVariance function, applied on same data type as input 
        signal (e.g., raw EMG, teager-kayser...)(default None).
    stbsl : float | None
        If None, use the baseline standard deviation of the trial
        (between first time sample and time sample 0). To use global stbsl,
        apply getGlobalVariance on the same data type as input signal 
        (e.g., raw EMG, teager-kayser...)(default None).
    sf : float
        Sampling frequency.
    Use_derivative : bool
        If True signal derivative is used to determine offset (default False).
    moving_avg_size : int
        Defines the window size (in samples) for the moving average applied 
        on signal before detection. If moving_avg_size is equal to 1, this has
        no effect (default is 1).

    Returns:
    --------
    2D array
        Each line corresponds to one burst interval. First column contains 
        onset, second column contains offset.
    """
    import numpy as np
    
    if sf is None:
        raise ValueError('Sampling frequency sf must be provided.')
    
    t0 = find_times(0, times)
    original_data = data_trial
    data_trial = moving_avg(data_trial, moving_avg_size)
    data_trial = np.abs(data_trial)

    if mbsl is None: mbsl = data_trial[:t0].mean()
    if stbsl is None: stbsl = data_trial[:t0].std()

    above_threshold = data_trial > (mbsl + th * stbsl)
    onsets = np.where((above_threshold[:-1] == False) & (above_threshold[1:] == True))[0] + 1
    offsets = np.where((above_threshold[:-1] == True) & (above_threshold[1:] == False))[0] + 1

    if len(onsets) > 0 or len(offsets) > 0 :
        if use_derivative_onset :
            for onset in np.arange(len(onsets)):
                 deriv_data_trial = moving_avg(np.flip(np.diff(original_data[:onsets[onset]])),moving_avg_size_onset, centered=False)
                 onsets[onset] = onsets[onset]-np.where(deriv_data_trial <= 0)[0][0] 
            
        if use_derivative and onsets[0] < (data_trial.shape[0] - 1): # we exclude if onset is on last sample

            data_trial = np.abs(np.diff(original_data))
            
            mbsl = data_trial[:t0].mean()
            stbsl = data_trial[:t0].std()

            max_derivative = data_trial[onsets[0]:].argmax() + onsets[0]
            if data_trial[max_derivative] > (mbsl + th * stbsl):
                offset_derivative = np.where(data_trial[max_derivative:] > (mbsl + th * stbsl))[0]
                if len(offset_derivative) > 0:
                    offsets = offset_derivative[-1:] + max_derivative
                    onsets = onsets[:1]

        if len(offsets) == len(onsets)-1 : #end of signal is above threshold
                offsets = np.append(offsets,len(data_trial)-1) 
        elif len(offsets) == len(onsets)+1 : 
            onsets = np.insert(onsets,0,0) #start of signal is above threshold
    
        intervals = np.where((onsets[1:] - offsets[:-1]) > time_limit * sf)[0]
        onsets = np.hstack((onsets[0],onsets[intervals+1])) ; offsets = np.hstack((offsets[intervals],offsets[-1])) 
 
        samples = np.array([sum(above_threshold[onsets[burst]:offsets[burst]]) for burst in range(len(onsets))])
        
        small_bursts = samples <= min_samples * 2
        per_sec = sum(small_bursts) * sf / float(len(times))
        min_samples = min_samples + (varying_min * per_sec)
        
        onsets = onsets[samples >= min_samples] ; offsets = offsets[samples >= min_samples]
        
    return np.transpose((onsets,offsets))

def detector_dbl_th(data_trial, times, th=3, window_size=.020, 
                    min_above_threshold=.5, min_samples=3, 
                    mbsl = None, stbsl = None, sf = None):
    """Return the indices of time windows containing signal exceeding the 
    double threshold.    

    Define signal onsets and offsets using a double theshold procedure as 
    proposed by Bonato et al. (1998). Signal onset is detected when a minimum 
    number of points in the specified time window exceeds mbsl + (th * stbsl). 
    Signal offset is detected when the minimum number of points in the 
    time window falls below mbsl + (th * stbsl). Intervals containing less 
    than the defined minimal number of samples are excluded.
    For EMG, this gives time windows where EMG bursts are supposed to occur.

    Parameters:
    -----------
    data_trial : 1D array
        Input signal.
    times : 1D array
        Time instants.
    th : float
        Threshold to multiply by stbsl (default 3).
    window_size : float
        Duration of the upfront time window (in second) in which samples 
        exceeding the threshold are detected. 
    min_above_threshold : float
        Minimum proportion of time samples exceeding the theshold in the 
        time window to detect signal (default 50 %).
    min_samples : float
        Minimal number of data points exceeding mbsl + (th * stbsl) in signal 
        intervals. If less points are present, interval is excluded (default 3).
    mbsl : float | None
        If None, use the mean baseline value of the trial (between first
        time sample and time sample 0). To use global mbsl, use 
        getGlobalVariance function, applied on same data type as input 
        signal (e.g., raw EMG, teager-kayser...)(default None).
    stbsl : float | None
        If None, use the baseline standard deviation of the trial
        (between first time sample and time sample 0). To use global stbsl,
        apply getGlobalVariance on the same data type as input signal 
        (e.g., raw EMG, teager-kayser...)(default None).
    sf : float
        Sampling frequency.

    Returns:
    --------
    2D array
        Each line corresponds to one burst interval. First column contains onset, second column contains offset.

    Reference:
    ----------
    Bonato P, D'Alessio T, Knaflitz M (1998) A statistical method for the measurement
    of muscle activation intervals from surface myoelectric signal during gait.
    IEEE Trans Biomed Eng. 45(3):287-99.
    """
    import numpy as np
    t0 = find_times(0, times)
    window_size = int(window_size * sf)
    min_above_threshold = round(min_above_threshold * window_size)
    onsets = np.array([],dtype=int) ; offsets = np.array([],dtype=int)
    
    data_trial = np.abs(data_trial)
    if mbsl is None: mbsl = data_trial[:t0].mean()
    if stbsl is None: stbsl = data_trial[:t0].std()

    above_threshold = data_trial > (mbsl + th * stbsl)
    cs = np.cumsum(above_threshold)
    on = np.where((cs[window_size:] - cs[:-window_size]) > min_above_threshold)[0]
    off = np.where((cs[window_size:] - cs[:-window_size]) < min_above_threshold)[0] + window_size
    
    if any(on) : 
        onsets = np.append(onsets,on[0]) 
        if any(off[off > (onsets[-1] + window_size)] < len(data_trial-1)) :
            offsets = np.append(offsets,[off[off > (onsets[-1] + window_size)][0]])
        else: offsets = np.append(offsets,len(data_trial)-1) #end of signal is above threshold
       
        while any(on > offsets[-1]) :
            onsets = np.append(onsets,on[on > offsets[-1]][0])
            if any(off[off > (onsets[-1] + window_size)] < len(data_trial-1)) :
                offsets = np.append(offsets,off[off > (onsets[-1] + window_size)][0] )
            else: offsets = np.append(offsets,len(data_trial)-1) #end of signal is above threshold
    
    samples = np.array([sum(above_threshold[onsets[burst]:offsets[burst]]) for burst in range(len(onsets))])
    onsets = onsets[samples > min_samples] ; offsets = offsets[samples > min_samples]
        
    return np.transpose((onsets,offsets))


def signal_windows(sections, start, end, warn=False):
    """
    Defines mean temporal windows around the given time sections.
    
    The time window for each section starts at the average value between 
    the end of the previous section and the beginning of the current section. 
    The end of the time window corresponds to the start of the next time 
    window, i.e., the average between the end of the current section and 
    the beginning of the next section. Start of first window and end of last 
    windows are defined by the average with start_sample and end_sample.
    If any windows starts before start value, start value is changed to 0. If 
    any window ends after end value, end value is changed to the maximal value
    of sections.
    This function is used for instance to define the different time windows 
    used for EMG onset search.

    Parameters:
    -----------
    sections : 2D array
        Each line corresponds to one section. First column gives the section's
        start, second column gives the section's end. 
    start : float
        The beginning of the first time window corresponds to the average
        value between start and the start of first section.
    end : float
        The end of the last time window corresponds to the average
        value between the end of last section and end.
    warn : bool
        If True, a warning is displayed when start value is changed to 0 or
        when end value is changed to maximal value of sections (default False).

    Returns:
    --------
    2D array
        Each line corresponds to one time window. First column contains 
        window's start, second columns contains window's end.
    
    """
    import numpy as np

    early_section = sections < start
    late_section = sections > end 

    if any(early_section.flatten()) > 0 : 
        start = 0
        if warn:
            print("Warning: Window starts before start point, new start point is " + str(start))
    if any(late_section.flatten()) > 0 : 
        end  = np.max(sections.flatten())
        if warn:
            print("Warning: Window ends after end point, new end point is" + str(end))
        
    borders = np.transpose((np.hstack((start,sections[:-1,1])),np.hstack((sections[1:,0],end))))
    windows = np.mean((sections,borders),axis=0)
    
    return windows


def getOnsets(data_trial, times, 
              useRaw = True, thRaw = 3.5, time_limitRaw = .025, min_samplesRaw = 3, 
              varying_minRaw = 1, mBlRaw = None, stBlRaw = None, 
              useTkeo = True, thTkeo = 8, time_limitTkeo = .025,  min_samplesTkeo = 10, 
              varying_minTkeo = 0, mBlTkeo = None, stBlTkeo = None, 
              sf = None, ip_search = [-.050,.050], 
              movingAvg_window = .015):
    """
    Return burst(s) onset(s) and offset(s).
    
    Identifies the time windows in data_trial containing EMG bursts, by looking 
    for signal exceeding the specified variance, using 'detectorVariance' function. 
    Then searchs for minimal and maximal values of each window integrated 
    profile.
    Finally, onset is defined as the earliest between the time sample of the 
    minimal value of integrated profile and the EMG burst start defined by 
    'detectorVariance' (i.e., signal exceeding the specified variance). Likewise, 
    offset is defined as the latest between the sample of maximal value of 
    integrated profile and the end of EMG burst as defined by 'detectorVariance'. 

    Parameters:
    -----------
    data_trial : 1D array
        Input signal.
    times : 1D array
        Time instants.
    useRaw : bool
        If True, apply 'detectorVariance' on raw (i.e., data_trial) signal.
    thRaw : float
        Threshold to multiply by variance, used in 'detectorVariance' applied
        on raw signal (default 3.5).
    time_limitRaw : float
        Lowest time delay (in second) between bursts for raw signal. Bursts
        intervals are merged When less time separates interval offset from next
        interval onset. (default is 25ms) 
    min_samplesRaw : float
        Minimal number of data points exceeding threshold to define a burst for
        raw signal. Bursts intervals containing less points exceeding the
        specified variance are excluded. This is applied after merging intervals 
        separated by less than the time limit (default is 3).
    varying_minRaw : float
        If True, the minimal number of data points exceeding threshold required 
        to define a burst is increased in noisy trials for raw signal. The noise
        is estimated by the frequency of "small" bursts (freq_small), i.e.,
        bursts containing less than twice the defined minimal number of samples.
        The minimal number of samples is then increased up to min_samples + 
        (freq_small * varying_min) (Default is 1, set to 0 to have no change of
        min_samples).
    mBlRaw : float | None
        If None, use the mean baseline value of the raw signal trial (between
        first time sample and time sample 0). To use global mBl, use 
        'getGlobalVariance' function (default is None).
    stBlRaw : float | None
        If None, use the baseline standard deviation of the raw signal trial
        (between first time sample and time sample 0). To use global stBl,
        apply 'getGlobalVariance' (default is None).
    useTkeo : bool
        If True, apply 'detectorVariance' on Teager-Kaiser transformation of
        data_trial signal.
    thTkeo : float
        Threshold to multiply by variance, used in 'detectorVariance' applied
        on Teager-Kaiser transformation of data_trial signal (default is 8).
    time_limitTkeo : float
        Lowest time delay (in second) between bursts for Teager-Kaiser
        transformation of data_trial signal. Bursts intervals are merged When
        less time separates interval offset from next interval onset
        (default is 25ms). 
    min_samplesTkeo : float
        Minimal number of data points exceeding threshold to define a burst for
        Teager-Kaiser transformation of data_trial signal. Bursts intervals
        containing less points exceeding the specified variance are excluded.
        This is applied after merging intervals separated by less than the time
        limit (default is 10).
    varying_minTkeo : float
        If True, the minimal number of data points exceeding threshold required 
        to define a burst is increased in noisy trials for Teager-Kaiser
        transformation of data_trial signal. The noise is estimated by the
        frequency of "small" bursts (freq_small), i.e., bursts containing less
        than twice the defined minimal number of samples. The minimal number of
        samples is then increased up to min_samples + (freq_small * varying_min)
        (Default is 0, meaning no change of min_samples).
    mBlTkeo : float | None
        If None, use the mean baseline value of the Teager-Kaiser transformation
        of data_trial signal (between first time sample and time sample 0). To
        use global mBl, use 'getGlobalVariance' function on the Teager-Kaiser
        signal transformation (default is None).
    stBlTkeo : float | None
        If None, use the baseline standard deviation of the Teager-Kaiser
        transformation of data_trial signal (between first time sample and time
        sample 0). To use global stBl, apply 'getGlobalVariance' on Teager-Kaiser
        signal transformation (default is None).
    sf : float
        Sampling frequency.
    ip_search : 1D array
        Time limits (in second) around EMG burst intervals (as defined by 
        'detectorVariance') to search for minimal and maximal integrated profile 
        values (Default [-.050 0.050]).
    movingAvg_window : float
        Time window size (in second) of moving average applied to smooth 
        integrated profile before to define minimal and maximal values. To 
        not smooth at all, set movingAvg_window to 1/sf (Default is 0.015).
       
    Returns:
    --------
    onsets,offsets : lists 
        Indices of data_trial corresponding to EMG onsets and offsets

    """
    import warnings 
    warnings.simplefilter('default')
    warnings.warn('getOnsets will be deleted in future version, use get_onsets instead.', DeprecationWarning)
    return get_onsets(data_trial, times, 
                      use_raw=useRaw, th_raw=thRaw, time_limit_raw=time_limitRaw, min_samples_raw=min_samplesRaw,
                      varying_min_raw=varying_minRaw, mbsl_raw=mBlRaw, stbsl_raw=stBlRaw, 
                      use_tkeo=useTkeo, th_tkeo=thTkeo, time_limit_tkeo=time_limitTkeo,  min_samples_tkeo=min_samplesTkeo, 
                      varying_min_tkeo=varying_minTkeo, mbsl_tkeao=mBlTkeo, stbsl_tkeo=stBlTkeo, 
                      sf=sf, ip_search=ip_search, moving_avg_window=movingAvg_window)

def get_onsets(data_trial, times, 
               use_raw=True, th_raw=3.5, time_limit_raw=.025, min_samples_raw=3, 
               varying_min_raw=1, mbsl_raw=None, stbsl_raw=None, 
               use_tkeo=True, th_tkeo=8, time_limit_tkeo=.025, min_samples_tkeo=10, 
               varying_min_tkeo=0, mbsl_tkeo=None, stbsl_tkeo=None, 
               sf=None, ip_search=[-.050,.050], moving_avg_window = .015):
    """Return burst(s) onset(s) and offset(s).
    
    Identifies the time windows in data_trial containing EMG bursts, by looking 
    for signal exceeding the specified variance, using 'detector_var' function. 
    Then searchs for minimal and maximal values of each window integrated 
    profile.
    Finally, onset is defined as the earliest between the time sample of the 
    minimal value of integrated profile and the EMG burst start defined by 
    'detector_var' (i.e., signal exceeding the specified variance). Likewise, 
    offset is defined as the latest between the sample of maximal value of 
    integrated profile and the end of EMG burst as defined by 'detector_var'. 

    Parameters:
    -----------
    data_trial : 1D array
        Input signal.
    times : 1D array
        Time instants.
    use_raw : bool
        If True, apply 'detector_var' on raw (i.e., data_trial) signal.
    th_raw : float
        Threshold to multiply by variance, used in 'detector_var' applied
        on raw signal (default 3.5).
    time_limit_raw : float
        Lowest time delay (in second) between bursts for raw signal. Bursts
        intervals are merged When less time separates interval offset from next
        interval onset. (default is 25ms) 
    min_samples_raw : float
        Minimal number of data points exceeding threshold to define a burst for
        raw signal. Bursts intervals containing less points exceeding the
        specified variance are excluded. This is applied after merging intervals 
        separated by less than the time limit (default is 3).
    varying_min_raw : float
        If True, the minimal number of data points exceeding threshold required 
        to define a burst is increased in noisy trials for raw signal. The noise
        is estimated by the frequency of "small" bursts (freq_small), i.e.,
        bursts containing less than twice the defined minimal number of samples.
        The minimal number of samples is then increased up to min_samples + 
        (freq_small * varying_min) (Default is 1, set to 0 to have no change of
        min_samples).
    mbsl_raw : float | None
        If None, use the mean baseline value of the raw signal trial (between
        first time sample and time sample 0). To use global mbsl, use 
        'global_var' function (default is None).
    stbsl_raw : float | None
        If None, use the baseline standard deviation of the raw signal trial
        (between first time sample and time sample 0). To use global stbsl,
        apply 'global_var' (default is None).
    use_tkeo : bool
        If True, apply 'detector_var' on Teager-Kaiser transformation of
        data_trial signal.
    th_tkeo : float
        Threshold to multiply by variance, used in 'detector_var' applied
        on Teager-Kaiser transformation of data_trial signal (default is 8).
    time_limit_tkeo : float
        Lowest time delay (in second) between bursts for Teager-Kaiser
        transformation of data_trial signal. Bursts intervals are merged When
        less time separates interval offset from next interval onset
        (default is 25ms). 
    min_samples_tkeo : float
        Minimal number of data points exceeding threshold to define a burst for
        Teager-Kaiser transformation of data_trial signal. Bursts intervals
        containing less points exceeding the specified variance are excluded.
        This is applied after merging intervals separated by less than the time
        limit (default is 10).
    varying_min_tkeo : float
        If True, the minimal number of data points exceeding threshold required 
        to define a burst is increased in noisy trials for Teager-Kaiser
        transformation of data_trial signal. The noise is estimated by the
        frequency of "small" bursts (freq_small), i.e., bursts containing less
        than twice the defined minimal number of samples. The minimal number of
        samples is then increased up to min_samples + (freq_small * varying_min)
        (Default is 0, meaning no change of min_samples).
    mbsl_tkeo : float | None
        If None, use the mean baseline value of the Teager-Kaiser transformation
        of data_trial signal (between first time sample and time sample 0). To
        use global mbsl, use 'global_var' function on the Teager-Kaiser
        signal transformation (default is None).
    stbsl_tkeo : float | None
        If None, use the baseline standard deviation of the Teager-Kaiser
        transformation of data_trial signal (between first time sample and time
        sample 0). To use global stbsl, apply 'global_var' on Teager-Kaiser
        signal transformation (default is None).
    sf : float
        Sampling frequency.
    ip_search : 1D array
        Time limits (in second) around EMG burst intervals (as defined by 
        'detector_var') to search for minimal and maximal integrated profile 
        values (Default [-.050 0.050]).
    moving_avg_window : float
        Time window size (in second) of moving average applied to smooth 
        integrated profile before to define minimal and maximal values. To 
        not smooth at all, set moving_avg_window to 1/sf (Default is 0.015).
       
    Returns:
    --------
    onsets,offsets : lists 
        Indices of data_trial corresponding to EMG onsets and offsets
    """
    import numpy as np
    
    if use_raw: emg_sections = detector_var(data_trial, times, th=th_raw, time_limit=time_limit_raw, min_samples=min_samples_raw, varying_min=varying_min_raw, mbsl=mbsl_raw, stbsl=stbsl_raw, sf=sf)
    else: emg_sections = np.transpose(([],[]))
    if use_tkeo: tkeo_sections = detector_var(tkeo(data_trial), times, th=th_tkeo, time_limit=time_limit_tkeo, min_samples=min_samples_tkeo, varying_min=varying_min_tkeo, mbsl=mbsl_tkeo, stbsl=stbsl_tkeo, sf=sf)
    else: tkeo_sections = np.transpose(([],[]))
    
    for tk in tkeo_sections:
        non_overlap = [b for b in range(len(emg_sections)) if ((emg_sections[b][1] < tk[0]) | (tk[1] < emg_sections[b][0]))]
        emg_sections = emg_sections[non_overlap]

    emg_sections = np.vstack((emg_sections,tkeo_sections))
                            
    onsets = [] ; offsets = []
    if emg_sections.shape[0] > 0:

        emg_sections = emg_sections[np.argsort(emg_sections[:,0])]

        t0 = find_times(times,0)
        ip_windows = np.array(signal_windows(emg_sections, t0, len(data_trial)),dtype = int)
        bsl_idx = np.hstack((np.arange(0,ip_windows[0,0]),np.arange(ip_windows[-1,1],len(data_trial))))
            
        for b in range(emg_sections.shape[0]):
            
            section_idx = np.hstack((bsl_idx,np.arange(ip_windows[b,0],ip_windows[b,1])))
            section_idx.sort()
            
            sample_search_start = int(emg_sections[b,0] + np.floor(ip_search[0]*sf) - (ip_windows[b,0] - ip_windows[0,0]))
            sample_search_end = int(emg_sections[b,1] + np.ceil(ip_search[1]*sf) - (ip_windows[b,0] - ip_windows[0,0]))
            sample_search = np.arange(np.max((sample_search_start, ip_windows[b,0] - (ip_windows[b,0] - ip_windows[0,0]))), np.min((sample_search_end, ip_windows[b,1] - (ip_windows[b,0] - ip_windows[0,0]))))
            
            on_ip,off_ip = get_onset_ip(data_trial[section_idx], times[:len(section_idx)], samples=sample_search, moving_avg_size=int(moving_avg_window * sf))

            onset = np.min((emg_sections[b,0],on_ip + (ip_windows[b,0] - ip_windows[0,0])))
            offset = np.min((emg_sections[b,1],off_ip + (ip_windows[b,0] - ip_windows[0,0])))
            # FIXME for Laure: decide something between those two and adjust the docstring
#            onset = onIp + (ip_windows[b,0] - ip_windows[0,0])
#            offset = offIp + (ip_windows[b,0] - ip_windows[0,0])

            onsets.append(onset)
            offsets.append(offset)
            
    return np.array(onsets,dtype=int), np.array(offsets,dtype=int)

def get_onsets_dbl_th(data_trial, times, window_size=.020, min_above_threshold=.5, 
                      use_raw=True, th_raw=3, mbsl_raw=None, stbsl_raw=None,
                      use_tkeo=True, th_tkeo=6, mbsl_tkeo=None, stbsl_tkeo=None, 
                      min_samples=3, sf=None, ip_search=[-.050,.050], moving_avg_window=.015):
    """
    Return burst(s) onset(s) and offset(s).
    
    Identifies the time windows in data_trial containing EMG bursts, by looking 
    for signal exceeding the specified variance, using 'detectorDblthreshold' 
    function. 
    Then searchs for minimal and maximal values of each window integrated 
    profile.
    Finally, onset is defined as the earliest between the time sample of the 
    minimal value of integrated profile and the EMG burst start defined by 
    'detector_dbl_th' (i.e., signal exceeding the specified variance). 
    Likewise, offset is defined as the earliest between the sample of maximal 
    value of integrated profile and the end of EMG burst as defined by 
    'detector_dbl_th'. 

    Parameters:
    -----------
    data_trial : 1D array
        Input signal.
    times : 1D array
        Time instants.
    window_size : float
        Duration of the upfront time window (in second) in which samples 
        exceeding the threshold are detected. 
    min_above_threshold : float
        Minimum proportion of time samples exceeding the theshold in the 
        time window to detect signal (default 50 %).
    use_raw : bool
        If True, 'detector_dbl_th' is applied on raw (i.e., data_trial) signal.
    th_raw : float
        Threshold to multiply by stbsl for raw signal (default 3).
    mbsl_raw : float | None
        If None, use the mean baseline value of raw signal (between first time
        sample and time sample 0). To use global mbsl, use 'global_var'
        (default is None).
    stbsl_raw : float | None
        If None, use the baseline standard deviation of raw signal (between
        first time sample and time sample 0). To use global stbsl, apply
        'global_var' (default is None).
    use_tkeo : bool
        If True, 'detector_dbl_th' is applied on Teager-Kaiser transformation 
        of data_trial signal.
    th_tkeo : float
        Threshold to multiply by stbsl for Teager-Kaiser transformation of
        data_trial signal (default 6).
    mbsl_tkeo : float | None
        If None, use the mean baseline value of the Teager-Kaiser transformation
        of data_trial signal (between first time sample and time sample 0). To
        use global mbsl, use 'global_var' function on the Teager-Kaiser
        signal transformation (default is None).
    stbsl_tkeo : float | None
        If None, use the baseline standard deviation of the Teager-Kaiser
        transformation of data_trial signal (between first time sample and time
        sample 0). To use global stbsl, apply 'global_var' on Teager-Kaiser
        signal transformation (default is None).
    min_samples : float
        Minimal number of data points exceeding mbsl + (th * stbsl) in signal 
        intervals. If less points are present, interval is excluded (default 3).
    sf : float
        Sampling frequency.
    ip_search : 1D array
        Time limits (in second) around EMG burst intervals (as defined by 
        'detector_dbl_th') to search for minimal and maximal integrated profile 
        values (Default [-.050 0.050]).
    moving_avg_window : float
        Time window size (in second) of moving average applied to smooth 
        integarted profile before to define minimal and maximal values. To 
        not smooth at all, set moving_avg_window to 1/sf (Default is 0.015).
       
    Returns:
    --------
    onsets,offsets : lists 
        Indices of data_trial corresponding to EMG onsets and offsets
    """
    import numpy as np
    
    moving_avg_window = int(moving_avg_window * sf)

    if use_raw: emg_sections = detector_dbl_th(data_trial, times, th=th_raw, window_size=window_size, min_above_threshold=min_above_threshold, min_samples=min_samples, mbsl=mbsl_raw, stbsl=stbsl_raw, sf=sf)
    else: emg_sections = np.transpose(([],[]))
    if use_tkeo: tkeo_sections = detector_dbl_th(tkeo(data_trial), times, th=th_tkeo, window_size=window_size, min_above_threshold=min_above_threshold, min_samples=min_samples, mbsl=mbsl_tkeo, stbsl=stbsl_tkeo, sf=sf)
    else: tkeo_sections = np.transpose(([],[]))
    
    for tk in tkeo_sections:
        non_overlap = [b for b in range(len(emg_sections)) if ((emg_sections[b][1] < tk[0]) | (tk[1] < emg_sections[b][0]))]
        emg_sections = emg_sections[non_overlap]

    emg_sections = np.vstack((emg_sections,tkeo_sections))
                            
    onsets = [] ; offsets = []
    if emg_sections.shape[0] > 0:

        emg_sections = emg_sections[np.argsort(emg_sections[:,0])]

        t0 = find_times(times,0)
        ip_windows = np.array(signal_windows(emg_sections, t0, len(data_trial)),dtype=int)
        bsl_idx = np.hstack((np.arange(0,ip_windows[0,0]),np.arange(ip_windows[-1,1],len(data_trial))))
            
        for b in range(emg_sections.shape[0]):
            
            section_idx = np.hstack((bsl_idx,np.arange(ip_windows[b,0],ip_windows[b,1])))
            section_idx.sort()
            
            sample_search_start = int(emg_sections[b,0] + np.floor(ip_search[0]*sf) - (ip_windows[b,0] - ip_windows[0,0]))
            sample_search_end = int(emg_sections[b,1] + np.ceil(ip_search[1]*sf) - (ip_windows[b,0] - ip_windows[0,0]))
            sample_search = np.arange(np.max((sample_search_start, ip_windows[b,0] - (ip_windows[b,0] - ip_windows[0,0]))),np.min((sample_search_end, ip_windows[b,1] - (ip_windows[b,0] - ip_windows[0,0]))))
            
            on_ip,off_ip = get_onset_ip(data_trial[section_idx], times[:len(section_idx)], samples=sample_search, moving_avg_size=moving_avg_window)

#            onset = np.min((emg_sections[b,0],onIp + (ip_windows[b,0] - ip_windows[0,0])))
#            offset = np.min((emg_sections[b,1],offIp + (ip_windows[b,0] - ip_windows[0,0])))
            onset = on_ip + (ip_windows[b,0] - ip_windows[0,0])
            offset = off_ip + (ip_windows[b,0] - ip_windows[0,0])

            onsets.append(onset)
            offsets.append(offset)
            
    return np.array(onsets,dtype=int), np.array(offsets,dtype=int)

def get_signal_portions(array, start_samples, stop_samples):
    """
    Get signals between each start_sample and stop_sample.

    Parameters:
    -----------
    array : 1D array
        Input signal array.
    start_samples : list | array of int
        Start samples of signal portions to get.
    stop_samples : list | array of int
        Stop (i.e., end) samples of signal portions to get.
    
    Returns:
    -----------
    list_signals : list
        List of arrays containing signal portions.
    """
    import numpy as np
    start_samples = np.array(start_samples)
    stop_samples = np.array(stop_samples)
    
    if (stop_samples < start_samples).any():
        raise ValueError('All start samples must be inferior to stop samples.')
        
    list_signals = []
    for p in range(len(start_samples)):
        list_signals.append(array[start_samples[p]:stop_samples[p]])
    
    return list_signals
    
def get_signal_max(list_signals):
    """
    Get signal max of each signal portion.

    Parameters:
    -----------
    list_signals : list
        List of arrays containing signal portions (e.g., obtained using
        'get_signal_portions').
    
    Returns:
    -----------
    val : 1D array
        Array containing maximum value for each array of list_signals.
    lat : 1D array
        Array containing latency of maximum value for each array of list_signals.
    """
    import numpy as np
    
    val = []
    lat = []
    for signal in list_signals:
        val.append(np.max(signal))
        lat.append(np.argmax(signal))
    return np.asarray(val), np.asarray(lat)

def somf(array, W):
    # Laure : do you have a ref? or time to explain ? I have no idea how this
    # is supposed to work + does not run at is in Python 3 because of the division
    
    # FIXME : not thoroughly tested. A warning should be raised!
    #TODO: make (W/2) a variable and use it instead of recomputing all the time
    #TODO2 ? : can't we use an np.array for s instead of a list? If we can 
    # determine its size a priori, it would be faster
    #TODO3: to speed-up, try to use Cython?
    import numpy as np
    s = [(W**2) / 12.]
    array = np.abs(array)
    for ti in np.arange((W / 2), array.shape[0] - (W / 2)):
        num = 0
        denom = 0
        for tn in np.arange(ti - (W / 2), ti + (W / 2)):
            num += (np.abs(array[tn])) * ((tn - ti)**2)
        denom = np.abs(array[ti - (W / 2):ti + (W / 2)]).sum()
        s.append(num / (denom * 1.))
    s = np.array(s)
    return s


def get_onset_somf(data, somf, W=900, L=.5):
    import numpy as np
    muSomf = (W**2) / 12
    th = L * muSomf
    logical = np.zeros(data.shape)

    remainingSignal = somf
    tons = [0]
    toffs = [0]
    while remainingSignal.shape[0] > 0:
        t = np.where(remainingSignal < th)[0]
        if t.shape[0] > 0:
            ton = t.min()
            print('ton = ' + str(ton))
            tons.append(ton + toffs[-1])
            t2 = np.where(remainingSignal[ton:] > th)[0]
            if t2.shape[0] > 0:
                toff = t2.min()
                print('toff = ' + str(toff))
            else:
                toff = data.shape[0]
            toffs.append(toff + tons[-1])
            logical[tons[-1] + W / 2:toffs[-1] + W / 2] = 1
            remainingSignal = somf[toffs[-1]:]
        else:
            remainingSignal = np.array([])
    return tons, toffs


def show_trial(data_trial, times, 
               use_raw=True, th_raw=3.5, time_limit_raw=.025, min_samples_raw=3, 
               varying_min_raw=1, mbsl_raw=None, stbsl_raw=None, 
               use_tkeo=True, th_tkeo=8, time_limit_tkeo=.025, min_samples_tkeo=10, 
               varying_min_tkeo=0, mbsl_tkeo=None, stbsl_tkeo=None, 
               sf=None, ip_search=[-.050,.050], 
               moving_avg_window=.015):
    """Show data trial burst detection and onset/offset detection.
    See 'get_onsets' function for complete description of the parameters.
    """
    import numpy as np
    import pylab as plt
    
    onsets,offsets = get_onsets(data_trial, times, 
                                use_raw=use_raw, th_raw=th_raw, time_limit_raw=time_limit_raw, min_samples_raw=min_samples_raw,
                                varying_min_raw=varying_min_raw, mbsl_raw=mbsl_raw, stbsl_raw=stbsl_raw, 
                                use_tkeo=use_tkeo, th_tkeo=th_tkeo, time_limit_tkeo=time_limit_tkeo, min_samples_tkeo=min_samples_tkeo,
                                varying_min_tkeo=varying_min_tkeo, mbsl_tkeo=mbsl_tkeo, stbsl_tkeo=stbsl_tkeo, 
                                sf=sf, ip_search=ip_search,
                                moving_avg_window=moving_avg_window)

    if use_raw:
        emg_sections = detector_var(data_trial, times, th=th_raw, time_limit=time_limit_raw, min_samples=min_samples_raw, varying_min=varying_min_raw, mbsl=mbsl_raw, stbsl=stbsl_raw, sf=sf)
    else:
        emg_sections = np.transpose(([],[]))
    if use_tkeo:
        tkeo_sections = detector_var(tkeo(data_trial), times, th=th_tkeo, time_limit=time_limit_tkeo, min_samples=min_samples_tkeo, varying_min=varying_min_tkeo, mbsl=mbsl_tkeo, stbsl=stbsl_tkeo, sf=sf)
    else:
        tkeo_sections = np.transpose(([],[]))
    
    for tk in tkeo_sections:
        non_overlap = [b for b in range(len(emg_sections)) if ((emg_sections[b][1] < tk[0]) | (tk[1] < emg_sections[b][0]))]
        emg_sections = emg_sections[non_overlap]

    emg_sections = np.vstack((emg_sections,tkeo_sections))

    
    plt.figure()
    plt.plot(times, data_trial, 'b', linewidth = .75)
    for section in emg_sections:
        plt.vlines(times[section[0]], data_trial.min()*1.1, data_trial.max()*1.1, linestyles='dotted')
        plt.vlines(times[section[1]], data_trial.min()*1.1, data_trial.max()*1.1, linestyles='dotted')

    for o in onsets:
        plt.plot(times[o],data_trial[o],'rx', markersize = 5)

    for o in offsets:
        plt.plot(times[o],data_trial[o],'rx', markersize = 5)

