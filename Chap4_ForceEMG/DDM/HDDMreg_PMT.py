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
    dataHDDM = hddm.load_csv('DDM/dataHDDM_pmt.csv')
    dataHDDM["subj_idx"] = dataHDDM["participant"]
    del dataHDDM["participant"]
    dataHDDM["SAT"] = dataHDDM.apply(lambda row: 0 if row['SAT'] == "Accuracy" else 1, axis=1)
    dataHDDM["FC"] = dataHDDM.apply(lambda row: -0.5 if row['FC'] == "low" else 0.5, axis=1)
    dataHDDM["contrast"] = dataHDDM.contrast.replace([1,2,3,4,5,6], [-.5,-.3,-.1,.1,.3,.5])
    dataHDDM["givenResp"] = dataHDDM["response"]
    dataHDDM["stim"] = dataHDDM.apply(lambda row: 1 if row['stim'] == 'Right' else 0, axis=1)
    dataHDDM["response"] = dataHDDM.apply(lambda row: 1 if row['givenResp'] == 'Right' else 0, axis=1)

    def v_link_func(x, data=dataHDDM):
        stim = (np.asarray(dmatrix('0 + C(s, [[1], [-1]])',
                               {'s': data.stim.ix[x.index]})))
        return x*stim
    if id < 4:
        ############## M1
        LM = [{'model':'t ~ SAT  + FC + contrast + SAT:FC + SAT:contrast + FC:contrast + SAT:FC:contrast', 'link_func': lambda x: x} ,
			  {'model':'v ~ contrast', 'link_func':v_link_func} ,
			  {'model':'a ~ FC + SAT + SAT:FC', 'link_func': lambda x: x} ]
	deps = {'sz' : 'SAT'}
        inc = ['sv','sz','st','z']
        model_name = "Joint_t0"
    else :
        return np.nan()
    name = 'light_reg_PMT_%s' %str(id)
    m = hddm.HDDMRegressor(dataHDDM, LM , depends_on = deps,
            include=inc, group_only_nodes=['sv', 'sz','st', "sz_SAT"], group_only_regressors=False, keep_regressor_trace=True)
    m.find_starting_values()
    m.sample(iter=10000, burn=8500, thin=1, dbname='DDM/traces/db_%s'%name, db='pickle')
    m.save('DDM/Fits/%s'%name)
    return m

v = ipyparallel.Client(profile="reg_PMT")[:]#sept
jobs = v.map(run_model, range(4 * 1))#4 chains for each model
wait_watching_stdout(jobs)
models = jobs.get()

