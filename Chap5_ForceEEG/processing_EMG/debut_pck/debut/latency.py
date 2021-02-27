# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:40:41 2019

@author: Laure Spieser and Boris Burle
Laboratoire de Neurosciences Cognitives
UMR 7291, CNRS, Aix-Marseille Université
3, Place Victor Hugo
13331 Marseille cedex 3
"""

from .utils import utilsfunc

def samples(tmin, tmax, sf):
    """Computes sample serie.

    Parameters:
    -----------
    tmin : float
        Starting time.
    tmax : float
        Ending time.
    sf : int | float
        Sampling frequency.

    Returns:
    --------
    sample_serie : 1D array
        Samples array.
    """
    import numpy as np

    if tmin > tmax:
        raise ValueError('tmax must be superior or equal to tmin')

    samplemin = int(np.floor(tmin * sf))
    sample_serie = np.arange(samplemin, int(np.ceil(tmax * sf)) + 1)
    return sample_serie

def times(tmin, tmax, sf):
    """Computes times serie.

    Parameters:
    -----------
    tmin : float
        Starting time.
    tmax : float
        Ending time.
    sf : int | float
        Sampling frequency.

    Returns:
    --------
    time_serie : 1D array
        Times array in seconds.
    """
    return samples(tmin, tmax, sf) / float(sf)


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

##############################################################################
# Class Latency ##############################################################
##############################################################################
class Latency():
    """ Latency class, contains sample and time latency of events.

    Given latencies are stored in Latency.sample in samples and in Latency.time
    in seconds. Either sample or time can be provided, the other one is
    computed automatically based on sampling frequency (sf). Latency.sample and
    Latency.time are numpy.ndarray, in which one given event is stored at the
    same position (i.e., latency of event 'n' is stored in Latency.sample[n]
    and Latency.sample[n]).

    Parameters:
    -----------
    sample : int | list of int | 1D array of int
        Latency in samples.
    time : float | list of float | 1D array of float
        Latency in second.
    sf : float
        Sampling frequency of the corresponding data recording.
    """

    def __init__(self, sample=None, time=None, sf=None):
        import numpy as np

        if sf is None:
            sf = input('Enter sampling frequency: ')
        self.sf = float(sf)

        if (sample is None) & (time is None):
            sample = []
            time = []

        elif (sample is None) & (time is not None):
            time = np.array(utilsfunc.in_list(time))
            if (np.isnan(time)).any():
                print(('Warning: nan in time[{}]!').format(np.where(np.isnan(time))[0]))
            sample = np.array(np.round(time * self.sf), dtype=int)
        elif (sample is not None) & (time is None):
            if (np.isnan(sample)).any():
                print(('Warning: nan in sample[{}]!').format(np.where(np.isnan(sample))[0]))
            sample = np.array(utilsfunc.in_list(sample), dtype=int)
            time = sample / float(self.sf)
        else:
            sample = utilsfunc.in_list(sample)
            time = utilsfunc.in_list(time)

        self.sample = np.array(sample, dtype=int)
        self.time = np.array(time)
        
        if (self.sample.size) != (self.time.size):
            raise TypeError('sample and time must be of same length.')

    def __repr__(self):
        return "class Latency, {} events, sf = {}, sample: {}, time: {}".format(len(self.sample), self.sf, self.sample, self.time)

    def nb_lat(self):
        """Returns the number of latencies.
        """
        return len(self.sample)

    def sort_sample(self):
        """Returns the indices of sorted samples.
        """
        import numpy as np
        return np.argsort(self.sample)

    def sort_time(self):
        """Returns the indices of sorted times.
        """
        import numpy as np
        return np.argsort(self.time)

    def add_lat(self, lat):
        """Adds Latency.
        """
        import numpy as np
        if self.sf != lat.sf:
            raise ValueError('Sampling frequencies do not match !')
        self.sample = np.concatenate((self.sample, lat.sample))
        self.time = np.concatenate((self.time, lat.time))

    def del_lat(self, idx_lat):
        """Deletes Latency.
        """
        import numpy as np
        self.sample = np.delete(self.sample, idx_lat)
        self.time = np.delete(self.time, idx_lat)

    def get_lat(self, idx_lat):
        """Gets the specified latencies.
        """
        return Latency(sample=self.sample[idx_lat], time=self.time[idx_lat], sf=self.sf)

    def copy(self):
        """Returns a copy of the current Latency.
        """
        import numpy as np
        copy_sample = np.copy(self.sample)
        copy_time = np.copy(self.time)
        return Latency(sample=copy_sample, time=copy_time, sf=self.sf)

    def find_lat(self, sample=None, time=None):
        """Finds the specified latencies and returns the position(s).

        Finds the indices of latencies whose sample and/or time are equal to the
        given value(s). If not None, sample and time must be of same length.If 
        both sample and time are filled, finds latencies satisfying both 
        requirements: Latency.find_lat(sample = 512, time=0.5) returns latencies 
        indices whose sample == 512 AND time == 0.5. If more than one sample and
        one time are given, searchs for latencies satisfying each pair of sample/time: 
        Latency.find_lat(sample = [512, 1536], time=[0.5, 1.5]) returns indices of 
        latencies whose (sample == 512 AND time == 0.5) or (sample == 1536 AND
        time == 1.5). Note that using Events.find_events(sample = [512, 1536], time=[0.5, 1.5])
        however returns event latencies whose (sample == 512 OR sample == 1536)
        AND (time == .5 OR time == 1.5).      

        Parameters:
        -----------
        sample : int | list of  int | 1D array of int
            Sample value(s) to look for.
        time : float | list of float | | 1D array of float
            Time value(s) to look for.

        Returns:
        -----------
            Position(s) of the found latencies.
        """

        import numpy as np
        
        if time is None:
            idx_lat = np.flatnonzero(np.in1d(self.sample, sample))
        elif sample is None:
            idx_lat = np.flatnonzero(np.in1d(self.time, time))
        else: 
            sample = utilsfunc.in_list(sample)
            time = utilsfunc.in_list(time)
            if len(sample) != len(time):
                raise ValueError ('Sample and time must of same length.')

            idx_lat = []
            for e in range(len(sample)):
                sample_index = np.flatnonzero(self.sample == sample[e])
                # keeps only those whose time is also equal
                idx_lat.extend([s for s in sample_index if self.time[s] == time[e]])
            idx_lat = np.asarray(idx_lat)
            
        return idx_lat
    
