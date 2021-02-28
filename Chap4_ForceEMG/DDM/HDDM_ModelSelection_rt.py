import scipy.stats as spss
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob,os
import pandas as pd
import ipyparallel
import hddm
from kabuki.analyze import gelman_rubin
import sys
import time
from IPython.display import clear_output

print(os.getcwd())

def wait_watching_stdout(ar, dt=100):
    ## ar: vmap output of the models being run
    ## dt: number of seconds between checking output
    while not ar.ready():
        stdouts = ar.stdout
        if not any(stdouts):
            continue
        # clear_output doesn't do much in terminal environments
        clear_output()
        print '-' * 30
        print "%.3fs elapsed" % ar.elapsed
        print ""
        for out in ar.stdout: print(out);
        sys.stdout.flush()
        time.sleep(dt)

def run_model(id):
    from patsy import dmatrix
    from pandas import Series
    import numpy as np
    import hddm
    dataHDDM = hddm.load_csv('DDM/dataHDDM_rt.csv')
    dataHDDM["subj_idx"] = dataHDDM["participant"]
    del dataHDDM["participant"]
    dataHDDM["givenResp"] = dataHDDM["response"]
    dataHDDM["stim"] = dataHDDM.apply(lambda row: 1 if row['stim'] == 'Right' else 0, axis=1)
    dataHDDM["response"] = dataHDDM.apply(lambda row: 1 if row['givenResp'] == 'Right' else 0, axis=1)
    if id < 4:
        ############## M1
        deps = {'sz' : 'SAT',
                'v' : 'contrast',
                't' : ['contrast','SAT', 'FC'],
                'a' : 'SAT'}
        inc = ['z','sv','sz','st']
        model_name = "M1"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")
    elif id > 3 and id < 8:
        ############## M2
        deps = {'sz' : 'SAT',
                'v' : ['contrast','SAT'],
                't' : ['contrast','SAT', 'FC'],
                'a' : 'SAT'}
        inc = ['z','sv','sz','st']
        model_name = "M2"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")
    elif id > 7 and id < 12:
        ############## M3
        deps = {'sz' : 'SAT',
                'v' : ['contrast','FC'],
                't' : ['contrast','SAT', 'FC'],
                'a' : 'SAT'}
        inc = ['z','sv','sz','st']
        model_name = "M3"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")
    elif id > 11 and id < 16:
        ############## M4
        deps = {'sz' : 'SAT',
                'v' : ['contrast','SAT','FC'],
                't' : ['contrast','SAT', 'FC'],
                'a' : 'SAT'}
        inc = ['z','sv','sz','st']
        model_name = "M4"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")
    elif id > 15 and id < 20:
        ############## M5
        deps = {'sz' : 'SAT',
                'v' : 'contrast',
                't' : ['contrast','SAT', 'FC'],
                'a' : ['SAT', 'FC']}
        inc = ['z','sv','sz','st']
        model_name = "M5"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")
    elif id > 19 and id < 24:
        ############## M6
        deps = {'sz' : 'SAT',
                'v' : ['contrast','SAT'],
                't' : ['contrast','SAT', 'FC'],
                'a' : ['SAT', 'FC']}
        inc = ['z','sv','sz','st']
        model_name = "M6"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")
    elif id > 23 and id < 28:
        ############## M7
        deps = {'sz' : 'SAT',
                'v' : ['contrast','FC'],
                't' : ['contrast','SAT', 'FC'],
                'a' : ['SAT', 'FC']}
        inc = ['z','sv','sz','st']
        model_name = "M7"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")
    elif id > 27 and id < 32:
        ############## M8
        deps = {'sz' : 'SAT',
                'v' : ['contrast','SAT','FC'],
                't' : ['contrast','SAT', 'FC'],
                'a' : ['SAT', 'FC']}
        inc = ['z','sv','sz','st']
        model_name = "M8"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")
    elif id > 31 and id < 36:
        ############## M9
        deps = {'sz' : 'SAT',
                'v' : 'contrast',
                't' : ['contrast','SAT', 'FC'],
                'a' : 'SAT',
		'z' : 'FC'}
        inc = ['z','sv','sz','st']
        model_name = "M9"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")
    elif id > 35 and id < 40:
        ############## M10
        deps = {'sz' : 'SAT',
                'v' : ['contrast','SAT'],
                't' : ['contrast','SAT', 'FC'],
                'a' : 'SAT', 
                'z' : 'FC'}
        inc = ['z','sv','sz','st']
        model_name = "M10"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")
    elif id > 39 and id < 44:
        ############## M11
        deps = {'sz' : 'SAT',
                'v' : ['contrast','FC'],
                't' : ['contrast','SAT', 'FC'],
                'a' : 'SAT', 
                'z' : 'FC'}
        inc = ['z','sv','sz','st']
        model_name = "M11"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")
    elif id > 43 and id < 48:
        ############## M12
        deps = {'sz' : 'SAT',
                'v' : ['contrast','SAT','FC'],
                't' : ['contrast','SAT', 'FC'],
                'a' : 'SAT', 
                'z' : 'FC'}
        inc = ['z','sv','sz','st']
        model_name = "M12"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")
    elif id > 47 and id < 52:
        ############## M13
        deps = {'sz' : 'SAT',
                'v' : 'contrast',
                't' : ['contrast','SAT', 'FC'],
                'a' : ['SAT', 'FC'], 
                'z' : 'FC'}
        inc = ['z','sv','sz','st']
        model_name = "M13"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")
    elif id > 51 and id < 56:
        ############## M14
        deps = {'sz' : 'SAT',
                'v' : ['contrast','SAT'],
                't' : ['contrast','SAT', 'FC'],
                'a' : ['SAT', 'FC'], 
                'z' : 'FC'}
        inc = ['z','sv','sz','st']
        model_name = "M14"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")
    elif id > 55 and id < 60:
        ############## M15
        deps = {'sz' : 'SAT',
                'v' : ['contrast','FC'],
                't' : ['contrast','SAT', 'FC'],
                'a' : ['SAT', 'FC'], 
                'z' : 'FC'}
        inc = ['z','sv','sz','st']
        model_name = "M15"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")
    elif id > 59 :
        ############## M16
        deps = {'sz' : 'SAT',
                'v' : ['contrast','SAT','FC'],
                't' : ['contrast','SAT', 'FC'],
                'a' : ['SAT', 'FC'], 
                'z' : 'FC'}
        inc = ['z','sv','sz','st']
        model_name = "M16"
        m = hddm.HDDMStimCoding(dataHDDM, depends_on=deps, include=inc,
            group_only_nodes=['sv', 'sz','st', "sz_SAT"], split_param='v', stim_col = "stim")
    else :
        return np.nan()
    name = 'light_RT_%s_%s' %(model_name, str(id))
    m.find_starting_values()
    m.sample(iter=10000, burn=8500, thin=1, dbname='DDM/traces/db_%s'%name, db='pickle')
    m.save('DDM/Fits/%s'%name)
    return m

v = ipyparallel.Client(profile="MS_RT")[:]#sept
jobs = v.map(run_model, range(4 * 16))#4 chains for each model
wait_watching_stdout(jobs)
models = jobs.get()
