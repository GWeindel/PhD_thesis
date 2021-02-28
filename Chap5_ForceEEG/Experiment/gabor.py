#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 16:29:31 2015

@author: weindel
"""
import os #handy system and path functions
from psychopy import core, visual, event, gui, data, monitors
import psychopy.logging as logging
import csv
import numpy as np
from psychopy.constants import *
import ctypes #for parallel port
import serial

_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

c1,c2,c3,c4 = ["weak","speed"], ["weak","accuracy"], ["strong","speed"], ["strong","accuracy"]

Counterbal_dict = {"pilot":c1,"S1":c1,"S5":c1,"S9":c1,"S13":c1,"S17":c1,
"S2":c2,"S6":c2,"S10":c2,"S14":c2,"S18":c2,
"S3":c3,"S7":c3,"S11":c3,"S15":c3,"S19":c3,
"S4":c4,"S8":c4,"S12":c4,"S16":c4,"S20":c4,"S16_bis":c2}



delta = .07 #Contrast difference between Left and right
NtrialsSession = 2448 #number of trials per session
alternate = NtrialsSession/12 #number of trials per successive condition
changeForce = NtrialsSession/8 #number of trials before changing force level
Ntrials = NtrialsSession/24 #number of trials before feedback


expName = 'ForceEMG'
expInfo = {'participant':'pilot','offset_R':'110',"offset_L":"370", "mx":''}

dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False: core.quit()
expInfo['date'] = data.getDateStr()
expInfo['expName'] = expName

filename = _thisDir + os.sep + u'Data' + os.sep + '%s_%s' %(expInfo['participant'], expInfo['date'])

 #An ExperimentHandler isn't essential but helps with data savin
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath=None,
    dataFileName=filename)
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
dataFile = open(filename+'.csv', 'w')
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

ser = serial.Serial("COM3", baudrate=9600, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS, timeout=1)
ser.close()
def write_force(level, offset_L, offset_R):
    value_L = int(np.round((85.84 + 0.04427* level)+level))+offset_L
    value_R = int(np.round((25.69 + 0.05531* level)+level))+offset_R
    print(value_L)
    ser.open()
    ser.write(str(value_L)+'g')#g for gauche
    ser.write(str(value_R)+'d')
    ser.close()

fixation = visual.GratingStim(win=win,#Fixation Cross
            mask='cross', size=0.2,
            pos=[0,0], sf=0, color='black')

gabor = visual.GratingStim(win,tex="sin", #Gabor patch
            mask="gauss",texRes=256,  pos=[0,0],
            size=2.5, sf=[1.2,0], ori = 0, name='gabor')

pause=visual.TextStim(win, ori=0,height = 0.5, name='pause',
    text=u"Prenez un petit temps de pause, reposez-vous les yeux.... puis appuyez sur n'importe quel bouton pour continuer.")

texte_fin=visual.TextStim(win, height = 0.5, ori=0, name='texte_fin',
    text=u"Fin de l'expérience. Merci d'attendre les instructions de l'expérimentateur..")

instructions = visual.TextStim(win, ori=0,height = 0.5, name='instructions',
    text=u"Début de l'expérience [REC]")

txtForce =visual.TextStim(win, units='norm',height = 0.1, pos=[0, -.5], name='pause',
    text=u"Force")

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
while True:
    instructions.draw()
    win.flip()
    instr_event= event.getKeys() #passé par l'expérimentateur
    if instr_event:
        break
    if event.getKeys(["escape"]): core.quit()


#############################TRIALS
#Trial Handler
trials=data.TrialHandler(nReps=NtrialsSession/Ntrials, method=u'random',
    originPath=None, extraInfo=expInfo,
    trialList=data.importConditions('Stimuli.csv'))
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

condition = Counterbal_dict[expInfo['participant']][-1]
force = Counterbal_dict[expInfo['participant']][0]

strong = int(expInfo['mx'])
weak = int(strong*.1)



if force == "weak":
    forcetrigger = 1
    write_force(weak,int(expInfo['offset_L']),int(expInfo['offset_R']))
    txtForce.text = "Force : Faible"
else :
    forcetrigger = 2
    write_force(strong,int(expInfo['offset_L']),int(expInfo['offset_R']))
    txtForce.text = "Force : Forte"

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
#Pause

    if counter > 1 and counter % Ntrials == 0 :
        fixation.autoDraw = False
        meanRT = sumRT / bloc_counter
        meanPrec = float(sumPrec) / float(bloc_counter)
        block +=1
        pausePending = False
        FBPending = True
        continueRoutine = True
        while continueRoutine:
            event.clearEvents(eventType='keyboard')
            if FBPending:
                RTTextMeanSub.text= u"Votre temps de réaction moyen est de %i millisecondes" %(meanRT*1000)
                RTTextMeanSub.draw()
                PrecTextMeanSub.text=u"Votre taux de réponses correctes est de %i %%" %(meanPrec*100)
                PrecTextMeanSub.draw()
                win.flip()
                if event.getKeys(["escape"]): core.quit()
                if event.getKeys(["space"]):#seul l'experimentateur peut passer cet écran
                    FBPending = False
                    pausePending = True
                    boolChangeCond = True
                    boolChangeForce = True
                    event.clearEvents(eventType='keyboard')
#                code1= ctypes.windll.inpout32.Inp32(lptno)
#                if code1 in port_gauche :#Bypass
#                    core.wait(0.2)
#                    code2 = ctypes.windll.inpout32.Inp32(lptno)
#                    if code2 in port_droit :
#                        pausePending = True
#                        FBPending = False
            if pausePending:
                if counter % alternate == 0 and boolChangeCond == True :
                    if condition == 'accuracy':#alternates conditions
                        condition = 'speed'
                    else:
                        condition = 'accuracy'
                    boolChangeCond = False
                if counter % changeForce == 0 and boolChangeForce == True:#automatic force level + trigger
                    if force == "weak":
                        force = "strong"
                        write_force(strong,int(expInfo['offset_L']),int(expInfo['offset_R']))
                        forcetrigger = 2
                        txtForce.text = "Force : Forte"
                    elif force == "strong":
                        force = "weak"
                        write_force(weak,int(expInfo['offset_L']),int(expInfo['offset_R']))
                        forcetrigger = 1
                        txtForce.text = "Force : Faible"
                    boolChangeForce = False
                pause.draw()
                win.flip()
                if event.getKeys(["escape"]): core.quit()
                lecture_port = ctypes.windll.inpout32.Inp32(0x379)
                if lecture_port  in port_gauche or lecture_port in port_droit or event.getKeys(["space"]) :
                    bypassPause=0
                    meanRT=0
                    sumRT = 0
                    sumPrec = 0
                    meanPrec=0
                    bloc_counter = 0
                    win.flip()
                    pausePending = False
                    continueRoutine=False
    if counter == 0 or counter % Ntrials == 0:#Affichage de la condition
        event.clearEvents(eventType='keyboard')
        if condition == "accuracy":
            conditionText.text = u"Prétez attention à votre précision"
        elif condition == "speed":
            conditionText.text = u"Prétez attention à votre vitesse"
        conditionText.draw()    
        txtForce.draw()    
        win.flip()
        core.wait(5)
        win.flip()
#composants boucle
    counter += 1
    bloc_counter += 1
    rep.num_trial = counter
    rep.num_block = block
    fixation.autoDraw = True
#Fixation
    win.flip()
    core.wait(0.5) #ISI
    win.flip()
#Definition contraste et côté de réponse
    gabor.contrast = intensity
    if corr_ans == 'right':
        corr_pos = [1.2,0]
        incorr_pos = [-1.2,0]
    elif corr_ans == 'left' :
        corr_pos = [-1.2,0]
        incorr_pos = [1.2,0]
    if condition == 'speed':
        addToTrigger = 1
    else:
        addToTrigger = 2
    trigger = str(addToTrigger)+str(partial_trigger)
#boucle
    
    rep.status = NOT_STARTED
    gabor.status = STARTED
    continueRoutine = True
    core.wait(0.45)#Fix - stim
    send_trigger(int(forcetrigger))#sends force trigger to acquisition computer
    core.wait(0.05)
    while continueRoutine:
        if event.getKeys(["p"]):#Stops experiment, discard trial
            pause2.draw()
            fixation.autoDraw = False
            win.flip()
            while True:
                pause_event= event.getKeys() #passé par l'expérimentateur
                if pause_event:
                    thisExp.addData('contraste', intensity)
                    thisExp.addData('direction', corr_ans)
                    thisExp.addData('keys',np.nan)
                    thisExp.addData('precision',np.nan)
                    thisExp.addData('rt',np.nan)
                    thisExp.addData('condition', condition)
                    thisExp.addData('trial',rep.num_trial)
                    thisExp.addData('block',rep.num_block)
                    thisExp.addData('trigger', trigger)
                    thisExp.addData('force', force)
                    thisExp.addData('mx', expInfo['mx'])
                    thisExp.nextEntry()
                    continueRoutine = False
                    break
        if gabor.status == STARTED:
            gabor.contrast += delta #manipulation symétrique du contraste
            gabor.pos=corr_pos
            gabor.draw()
            gabor.contrast -= delta
            gabor.pos=incorr_pos
            gabor.draw()
            win.flip()
            send_trigger(int(trigger))
            gabor.status = NOT_STARTED
        if rep.status == NOT_STARTED:
            rep.status = STARTED
            rep.clock.reset()
        if event.getKeys(["escape"]): core.quit()
        if  rep.status == STARTED:
            theseKeys=[]
            lecture_port = ctypes.windll.inpout32.Inp32(0x379)
            if lecture_port  in port_gauche:
                send_trigger(100)
                theseKeys.append("left")
            elif lecture_port in port_droit:
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
                thisExp.addData('contraste', intensity)
                thisExp.addData('direction', corr_ans)
                thisExp.addData('keys',rep.keys)
                thisExp.addData('precision',rep.corr)
                thisExp.addData('rt',rep.rt)
                thisExp.addData('condition', condition)
                thisExp.addData('trial',rep.num_trial)
                thisExp.addData('block',rep.num_block)
                thisExp.addData('trigger', trigger)
                thisExp.addData('force', force)
                thisExp.addData('mx', expInfo['mx'])
                thisExp.nextEntry()
                continueRoutine = False

    if event.getKeys(["escape"]):
        win.close
        core.quit()


#routine fin
finPending = True
continueRoutine=True
fixation.autoDraw = False
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

win.close()
core.quit()
