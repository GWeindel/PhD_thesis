#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 16:29:31 2015

@author: weindel
"""
from __future__ import division
import os #handy system and path functions
from psychopy import core, visual, event, gui, data, monitors
import psychopy.logging as logging
from datetime import datetime
import csv
import numpy as np
from psychopy.constants import *
import ctypes #for parallel port
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sciOpt
from scipy.special import expit
from scipy.stats import binom, norm


_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)


expName = 'SPEMG'
expInfo = {'participant':'test1'}

dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False: core.quit()
expInfo['date'] = data.getDateStr()
expInfo['expName'] = expName

filename = _thisDir + os.sep + u'Data' + os.sep + 'PRDM' + os.sep + '%s_%s' %(expInfo['participant'], expInfo['date'])

 #An ExperimentHandler isn't essential but helps with data savin
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath=None,
    dataFileName=filename)
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)

#Definition moniteur sujet
mon1 = monitors.Monitor('Sujet_EMG')
mon1.setDistance(110) #cm
mon1.setWidth(33) #cm
mon1.setSizePix([1024, 768])
mon1.saveMon() #now when you launch Mon Center it should be there

#create a window to draw in
win = visual.Window(size=[1024, 768], monitor=mon1, gamma=1.194, allowGUI=False, units='deg',fullscr=True)

class KeyResponse:
    def __init__(self):
        self.keys=[]
        self.corr=0
        self.rt=None
        self.clock=None
        self.num_trial = None
        self.num_block = None
port_gauche = [104]
port_droit = [112]
lptno = 0x379#parallel port adress

def send_trigger(code):
    ctypes.windll.inpout32.Out32(0xBCC8, code)
    core.wait(0.002)
    ctypes.windll.inpout32.Out32(0xBCC8, 0)
send_trigger(0)
def send_trigger(code):
    print('coucou')
    

base = -0.2
L50 = [(-0.4,base),(-0.4,.5+base),(0,0.5+base),(0,base)]
R50 = [(0.4,base),(0.4,.5+base),(0,0.5+base),(0,base)]
L70 = [(-0.4,base),(-0.4,.7+base),(0,0.7+base),(0,base)]
R70 = [(0.4,base),(0.4,.7+base),(0,0.7+base),(0,base)]
L30 = [(-0.4,base),(-0.4,.3+base),(0,0.3+base),(0,base)]
R30 = [(0.4,base),(0.4,.3+base),(0,0.3+base),(0,base)]

Lcue  = visual.ShapeStim(win,
            fillColor=None, size=2, lineColor='white',lineWidth=3)

Rcue  = visual.ShapeStim(win, 
            fillColor=None, size=2, lineColor='white',lineWidth=3)

fixation = visual.GratingStim(win=win,#Fixation Cross
            mask='cross', size=0.2,
            pos=[0,0], sf=0, color='black')

gabor = visual.GratingStim(win,tex="sin", #Gabor patch
            mask="gauss",texRes=256,  pos=[0,0],
            size=2.5, sf=[1.2,0], ori = 0, name='gabor')
            
mask = visual.GratingStim(win,tex="sin", #Gabor patch
            mask="gauss",texRes=256,  pos=[0,0],  contrast=.5,
            size=3, sf=[0,0], ori = 0, name='gabor')

pause=visual.TextStim(win, ori=0,height = 0.5, name='pause',
    text=u"Prenez un petit temps de pause, reposez-vous les yeux.... puis appuyez sur n'importe quel bouton pour continuer.")

texte_fin=visual.TextStim(win, height = 0.5, ori=0, name='texte_fin',
    text=u"Fin de l'expérience. Merci d'attendre les instructions de l'expérimentateur..")


correct =visual.TextStim(win, height = 0.5, ori=0, name='correct',
    text=u"Correct !")
    
incorrect =visual.TextStim(win, height = 0.5, ori=0, name='incorrect',
    text=u"Incorrect !")


instructions = visual.TextStim(win, ori=0,height = 0.5, name='instructions',  pos=[0,1],
    text=u"Bienvenue !\n A chaque essai vous verrez un stimulus noir et blanc au centre de l'écran \n Appuyez sur le bouton DROITE si vous pensez que le stimulus est orienté vers la DROITE (sens horaire), appuyez sur le bouton GAUCHE si vous pensez que le stimulus est orienté vers la GAUCHE (sens anti-horaire). \n\n\n\n\n\n\n Veuillez répondre le plus vite et le plus correctement possible. Donnez toujours une réponse, même si vous êtes indécis et gardez les yeux sur la croix de fixation.")

RTTextMeanSub = visual.TextStim(win,
                        units='norm',height = 0.1,
                        text='Mean : ...',color='white',autoLog = False)

PrecTextMeanSub = visual.TextStim(win,
                        units='norm',height = 0.1, pos=[0, -0.5],
                        text='Mean : ...',color='white',autoLog = False)

conditionText = visual.TextStim(win,
                        units='norm',height = 0.1, pos=[0, 0],
                        text='...',color='white',autoLog = False)


################################### INSTRUCTIONS
corr_ans_inst = ["left","right"]
orientations = [-0.5, 1.5,  -2, 0.5, -0.75]
instructionLoop = True
inst = True
while instructionLoop:
    instructions.draw()
    win.flip()
    if event.getKeys(["space"]):#passé par l'expérimentateur
        for i in np.arange(4):
            win.flip()
            core.wait(.5)#ISIm
            fixation.draw()
            win.flip()
            corr_ans = np.random.choice(corr_ans_inst)
            orientation = np.random.choice(orientations)
            if corr_ans == "left":
                gabor.ori = -orientation
            else : 
                gabor.ori = -orientation
            core.wait(0.5)#Fix - stim
            gabor.draw()
            win.flip()
            continueRoutine = True
            while continueRoutine:
                theseKeys=[]
                lecture_port = ctypes.windll.inpout32.Inp32(0x379)
                if lecture_port  in port_gauche:
                #if event.getKeys("q"): 
                    send_trigger(100)
                    theseKeys.append("left")
                #elif  event.getKeys("m"): 
                elif lecture_port  in port_droit:
                    send_trigger(200)
                    theseKeys.append("right")
                if len(theseKeys) > 0:
                    if theseKeys[0] == corr_ans:
                        correct.draw()
                    else:
                        incorrect.draw()
                    win.flip()
                    core.wait(2)
                    continueRoutine = False
        pause.draw()
        win.flip()
        event.waitKeys()
        break
    if event.getKeys(["escape"]): core.quit()
    lecture_port = ctypes.windll.inpout32.Inp32(0x379)
    if lecture_port  in port_gauche or lecture_port in port_droit and inst == True:
        gabor.draw()
        instructions.draw()
        win.flip()
        event.waitKeys()
        inst=False


#############################TRIALS
#Trial Handler
trials=data.TrialHandler(nReps=1, method=u'random',
    originPath=None, extraInfo=expInfo,
    trialList=data.importConditions('Stimuli/Stimuli_PRDM.csv'))
thisTrial=trials.trialList[0]
if thisTrial!=None:
    for paramName in thisTrial.keys():
        exec(paramName+'=thisTrial.'+paramName)

#Init
trialClock = core.Clock()
counter = 0
bloc_counter = 0
block = 1
rep = KeyResponse()
rep.clock = core.Clock()


#Trial loop
sumRT= 0
currentRT=0
sumPrec = 0
currentPrec = 0
for thisTrial in trials:
    event.clearEvents(eventType='keyboard')
    if thisTrial!=None:
        for paramName in thisTrial.keys():
            exec(paramName+'=thisTrial.'+paramName)
#composants boucle
    counter += 1
    bloc_counter += 1
    rep.num_trial = counter
    rep.num_block = block

#Fixation
    if counter > 1:
        mask.draw()
    win.flip()
    core.wait(.1) #ISI
    win.flip()
    core.wait(.4)#ISIm
    fixation.draw()
    win.flip()

#Definition contraste et côté de réponse
    gabor.ori = orientation
    rep.status = NOT_STARTED
    gabor.status = STARTED
    continueRoutine = True
    core.wait(0.5)#Fix - stim
    while continueRoutine:
        if rep.status == NOT_STARTED:
            rep.status = STARTED
            gabor.draw()
            win.flip()
            rep.clock.reset()
        if event.getKeys(["escape"]): core.quit()
        if  rep.status == STARTED:
            theseKeys=[]
            lecture_port = ctypes.windll.inpout32.Inp32(0x379)
            if lecture_port  in port_gauche:
            #if event.getKeys("q"): 
                send_trigger(100)
                theseKeys.append("left")
            #elif  event.getKeys("m"): 
            elif lecture_port  in port_droit:
                send_trigger(200)
                theseKeys.append("right")
            if len(theseKeys) > 0:
                if "escape" in theseKeys:
                    win.close()
                    core.quit()
                rep.rt = rep.clock.getTime()
                win.flip()
                rep.keys=[]
                rep.keys = theseKeys[0]
                if rep.keys == corr_ans:
                    rep.corr = 1
                else:
                    rep.corr = 0
        #composants moyennes
                currentRT = rep.rt
                currentPrec = rep.corr
                sumPrec = sumPrec + currentPrec
                sumRT = sumRT + currentRT

        #Fin de boucle
                #thisExp.addData('p_right',p_right) 
                thisExp.addData('direction', corr_ans)
                thisExp.addData('keys',rep.keys)
                thisExp.addData('precision',rep.corr)
                thisExp.addData('rt',rep.rt)
                thisExp.addData('orientation', orientation)
                thisExp.addData('trial',rep.num_trial)
                thisExp.addData('time',datetime.time(datetime.now()))
                #thisExp.addData('trigger', trigger)
                #thisExp.addData('force', force)
                thisExp.nextEntry()
                continueRoutine = False

    if event.getKeys(["escape"]):
        win.close
        core.quit()

#routine fin
finPending = True
continueRoutine=True
while continueRoutine:
    if  finPending:
        core.wait(2.0)
        texte_fin.draw()
        win.flip()
        core.wait(5.0)
        finPending = False
    if not continueRoutine:
        break
    continueRoutine = False
    if event.getKeys(["escape"]):
        core.quit()

thisExp.saveAsWideText(filename+'.txt')
df = pd.read_table(filename+'.txt')

win.close()

#==============================================================================
# Déclaration de la fonction objectives + dérivées
#==============================================================================
def HyperSec(x):
    '''
    Fonction pour calculer une sécante hyperbolique
    '''
    return 1/(np.cosh(x))
    
def PsychMet(P, x):
    '''
    Adapté de Palmer et coll. (2005), fonction psychométrique à deux paramètres,
    x = force du stimulus.
    '''
    a,k,t = P    
    return expit(2*a*k*x) #fonction logistique : 1 / (1 + np.exp(-2*a*k*x))

def ChronoMet(P, x):
    '''
    Adapté de Palmer et coll. (2005), fonction chronométrique à trois paramètres,
    x = force du stimulus.
    '''
    a,k,t = P  
    return ((a/(k*x))*np.tanh(a*k*x))+t

def Var(P, x, pt, y):
    '''
    Donne la variance estimée pour le TR prédit par les paramètres
    '''
    a,k,t = P
    mu = x*k
    varT = t*0.1 #Fixing varTer to sdTr/E(TR) = .1 (see Appendix Palmer, Huk & Shadlen, 2005)
    return (((a*np.tanh(a*mu))-((a*mu*HyperSec(a*mu))**2))/mu**3) + varT #0.1 = approximation de varT/ MàJ : varTr crée échec minimize

def LRT(P, x, y,n):
    '''
    Calcul de la vraisemblance d'observer TR prédit sachant TR observé
    '''
    a,k,t = P
    pt= ChronoMet(P,x)#Moyenne TR prédits + variance
    var = Var(P, x, pt, y)
    sem = np.sqrt((var/n)) #SEM prédite
    return norm.pdf(y, pt, sem)

def LP(P, x, r, n):
    '''
    Calcul de la vraisemblance d'observer proportion correcte prédite sachant PC observé
    '''
    a,k,t = P
    n = np.array(n)
    r = np.array(r)
    pc = PsychMet(P,x)
    b = binom(n, pc)    
    result = b.pmf(r)
    return result

def ComLogLik(P,x,y,r,n):
    '''
    Fonction objective maximisée, somme des log des vraisemblances d'observer les TR 
    et la précision prédits par les fonctions pour des paramètres donnés
    '''
    a,k,t = P
    ap, kp, tp = a/1, k/1, t/1
    P = [ap,kp,tp]
    LRTi = LRT(P, x, y, n)
    LRTi = np.log(LRTi) #Transformation LRTi en log
    LPi = LP(P,x, r, n) #LPï déjà retourné en log
    LPi = np.log(LPi)
    return (LRTi+LPi).sum()

df = df[(df.rt < 2.5) & (df.rt > .1)].reset_index()
#df.precision = df.apply(lambda row: 0  if row['precision'] == 1 else 1, axis=1 )
df.orientation = np.abs(df.orientation)
orientation = np.array(sorted(df['orientation'].unique())) #Les valeurs de contrastes manipulées
minOri= orientation[0]
maxOri = orientation[-1]

#==============================================================================
# Les initial guess, à corriger avec un curve_fit
#==============================================================================
P_guess = [0.30, 30, 0.30]
nll = lambda *args: -ComLogLik(*args) #Déclaration de la fonction objective, le signe - = maximiser la fonction

dFit = pd.DataFrame(columns = ['participant','seuil', 'drift','Ter','Status', 'LogLik'])
j = 0

for m in df.participant.unique():
    df_ = df[df.participant == m]
    #==============================================================================
    # Les paramètres utilisés par minimize
    #==============================================================================
    n,r,y = [],[],[] #array pour le nombre d'essais, le TR moyen, le nombre de réponses correctes pour Speed et Acc
    for i in orientation:
        y.append(df_[df_.orientation==i]['rt'].mean())#TR moyen pour chaque valeur de contraste (i)
        r.append(int(df_[df_.orientation==i]['precision'].sum()))#Nbr réponses correctes pour chaque valeur de contraste (i)
        n.append(len(df_[df_.orientation==i]))#Nbr essais pour chaque valeur de contraste (i)
    #==============================================================================
    # Minimize, méthode Nelder-Mead à vérifier
    #==============================================================================
    res = sciOpt.minimize(nll, P_guess,method = 'Nelder-Mead', args=(orientation, y, r, n))
    dFit.loc[j] = m, res.x[0], res.x[1], res.x[2],res.success, res.fun
    j += 1

print dFit
k = 0
x_ = np.random.uniform(low=minOri, high=maxOri, size=10000)
f, axarr = plt.subplots(2,1, sharex=True)
yPrec_s, yRT_s, yPrec_a,yRT_a = [],[],[],[] #Array pour stockage des données chrono et psycho simulées
ParamS = [dFit.seuil[k],dFit.drift[k],dFit.Ter[k]]
for i in x_: #générer une valeurs y pour chaque x avec les paramètres fournis par les fits
    y_s = PsychMet(ParamS,i)#paramètres issus fit 
    y__s = ChronoMet(ParamS,i)#paramètres issus fit  
    yPrec_s.append(y_s)
    yRT_s.append(y__s)

dfGenP_s = pd.DataFrame([x_,yPrec_s])#stockage des données générées pour x associé
dfGenP_s = dfGenP_s.transpose()# formattage pour données en colonne
dfGenP_s = dfGenP_s.sort(0)
dfGenRT_s = pd.DataFrame([x_,yRT_s])
dfGenRT_s = dfGenRT_s.transpose()
dfGenRT_s = dfGenRT_s.sort(0)

axarr[0].plot(dfGenRT_s[0],dfGenRT_s[1])
axarr[0].plot(df.groupby("orientation").mean().reset_index().sort("orientation").orientation,df.groupby("orientation").mean().reset_index().sort("orientation").rt, 'o')
axarr[1].plot(dfGenP_s[0],dfGenP_s[1])
axarr[1].plot(df.groupby("orientation").mean().reset_index().sort("orientation").orientation,df.groupby("orientation").mean().reset_index().sort("orientation").precision, 'o')
print(np.mean(dfGenP_s[np.round(dfGenP_s[1], decimals=2)==.90][0]))

if np.mean(dfGenP_s[np.round(dfGenP_s[1], decimals=2)==.90][0]) != np.nan:
    f = open(_thisDir+os.sep+'Data'+os.sep+'PRDM'+os.sep+expInfo['participant']+'_PRDM.txt','w')
    f.write('{}'.format(np.mean(dfGenP_s[np.round(dfGenP_s[1], decimals=2)==.90][0])))
    f.close()
else :
    print('Not recorded')

plt.savefig(_thisDir+os.sep+'Data'+os.sep+'PRDM'+os.sep+expInfo['participant']+'.png')
plt.show()

core.quit()