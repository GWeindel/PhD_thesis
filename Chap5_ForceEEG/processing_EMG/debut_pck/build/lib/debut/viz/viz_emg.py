# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 17:23:41 2014

@author: Laure Spieser and Boris Burle
Laboratoire de Neurosciences Cognitives
UMR 7291, CNRS, Aix-Marseille Université
3, Place Victor Hugo
13331 Marseille cedex 3
"""

import sys
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QWidget , QTabWidget
import pyqtgraph as pg
import numpy as np
from ..events import Events
from ..latency import Latency
from copy import deepcopy


## Always start by initializing Qt (only once per application)

class CustomEventPlot(pg.PlotItem):
    
    def __init__(self, movable_1, movable_2, sf=None, color_events=['y','b',pg.mkPen(color=(0, 201, 255, 255)),'r']):
        super(CustomEventPlot,self).__init__()
        self.enableAutoRange = False
        self.label = pg.LabelItem(justify='right')
        self.movable_1 = movable_1
        self.movable_2 = movable_2
        self.color_events = color_events
        self.list_plot_lines = []
 
    def new_events_epoch(self, events_epoch):
        self.list_plot_lines = []
        for e in range(len(events_epoch.code)):

            if events_epoch.code[e] == self.movable_1:
                line = self.addLine( x=events_epoch.lat.time[e], movable=True, pen=self.color_events[1])
                line.code = events_epoch.code[e]
                self.list_plot_lines.append(line)

            elif events_epoch.code[e] == self.movable_2:
                line = self.addLine( x=events_epoch.lat.time[e], movable=True, pen=self.color_events[2])
                line.code = events_epoch.code[e]
                self.list_plot_lines.append(line)

            else:
                line = self.addLine( x=events_epoch.lat.time[e], movable=False, pen=self.color_events[0])
                line.code = events_epoch.code[e]
                self.list_plot_lines.append(line)
 

#    def onClick(self,eventMouse,eventKey):
    def on_click(self,event_mouse):    
        
        vb = self.vb
        modifiers = QApplication.keyboardModifiers()
      
#        if event_mouse.double() : print('double click')
#        else: print('not double')
        
        if ( (event_mouse.button() == QtCore.Qt.LeftButton) ):#& (eventKey.key() == QtCore.Qt.Key_Control) ):
#            event_mouse.accept()
            if modifiers == QtCore.Qt.ControlModifier:
                click_pos = vb.mapToView(event_mouse.lastPos())
                self.add_event(click_pos, self.movable_1, movable=True, line_color=self.color_events[1])
            else:
                self.mousePressEvent(event_mouse)
        elif ( (event_mouse.button() == QtCore.Qt.RightButton) ):#& (eventKey.key() == QtCore.Qt.Key_Control) ):
#            event_mouse.accept()
            if modifiers == QtCore.Qt.ControlModifier:
                click_pos = vb.mapToView(event_mouse.lastPos())
                self.add_event(click_pos, self.movable_2, movable=True, line_color=self.color_events[2])
            else:
                line_under_mouse = [ line for line in self.list_plot_lines if (line.movable) & (line.isUnderMouse())]
                if len(line_under_mouse) > 0:
                    self.remove_event(line_under_mouse[0])
                else:
                    self.mousePressEvent(event_mouse)
        else:
            self.mousePressEvent(event_mouse)

            
    def add_event(self, pos, code, movable=False, line_color='y'):
        line = self.addLine(x=pos, movable=movable, pen=line_color)
        line.code = code
        self.list_plot_lines.append(line)

    def remove_event(self, line):
        self.removeItem(line)
        self.list_plot_lines.remove(line)

    def get_events_pos(self,idx):
        if type(idx) is int : idx = [idx]     
        return np.array([self.list_plot_lines[n].pos()[0] for n in idx])

    def get_events_code(self,idx):
        if type(idx) is int : idx = [idx]     
        return np.array([self.list_plot_lines[n].code for n in idx])

#    def setKeyboardPrevNext(self,listKeys):
#        self.keyGoToPrev = listKeys[0]        
#        self.keyGoToNext = listKeys[1]
#        
#    def pressKey(self, eventKey):
#         key = eventKey.key()
#         print(key)
##         eventKey.accept()
#         if key == QtCore.Qt.Key_Space:
#            print('space')


class VizUi(QWidget):
    
    def __init__(self, write_evt_file=True, fname_evts='viz_events.csv'):
        super(VizUi,self).__init__()
        
        self.nb_epoch = 0
        self.nb_chan = 0
        self.tmin_plot = ''
        self.tmax_plot = ''
        self.vmin_plot = ''
        self.vmax_plot = ''
        self.chan_scale = range(self.nb_chan)
        
        global current_epoch 
        current_epoch = ''

        self.write_evt_file = write_evt_file
        if self.write_evt_file:
            self.fname_evts = fname_evts
            f = open(self.fname_evts,'w')
            f.write("trial_idx,sample,time,code,chan,trial_raw_tmin.sample,trial_raw_tmin.time,trial_raw_t0.sample,trial_raw_t0.time,trial_raw_tmax.sample,trial_raw_tmax.time\n")
            f.close()
        
        self.initUi()
        self.set_keyboard_prev_next()
        self.keyPressEvent = self.press_key


#    def keyPressEvent(self, eventKey):
#         key = eventKey.key()
#         print(key)

        
    def press_key(self, eventKey):
         key = eventKey.key()
#         print(key)
         if key == self.key_go_to_prev: self.go_to_prev()
         elif key == self.key_go_to_next: self.go_to_next()
#         else: self.keyPressEvent()
         
    def load_data(self,data_epochs, times, ep_evts, channels='all', code_movable_1='onset', code_movable_2='offset', sync_chan=[[0,1],[0,1],[2,3],[2,3]], group_chan=True, list_trials=None, random_order=False):
        self.t = times 
        self.t0sample =  np.abs(0 - self.t).argmin()
        self.tmin_plot = self.t[0]
        self.tmax_plot = self.t[-1]
        self.data = data_epochs
        self.nb_epoch = data_epochs.shape[0]
        if self.nb_epoch == 0:
            raise ValueError('No trial found in data!')
        
        self.events = deepcopy(ep_evts)
        if self.events.nb_trials() == 0:
            raise ValueError('No trial found in epoch events!')
        elif self.events.nb_trials() != self.nb_epoch:
            raise ValueError('Number of trials in data do not match epoch events.')
#        if sf is None: sf = input('Enter sampling frequency: ')
        self.sf = ep_evts.sf
#        if len(sync_chan) > max(self.channels):

        if list_trials is None:
            self.list_epochs = np.arange(self.nb_epoch)
        else: 
            self.list_epochs = np.asarray(list_trials)
            
        if random_order is True:
            self.list_epochs = np.random.permutation(self.list_epochs)
            self.user_info_epoch = np.arange(self.nb_epoch)
        else:
            self.user_info_epoch = self.list_epochs
            
        if (self.list_epochs >= self.nb_epoch).any():
            raise ValueError(('Data contains only {} trials, can not plot trials {} .').format(self.nb_epoch, self.list_epochs[self.list_epochs >= self.nb_epoch]))
        self.nb_epoch = len(self.list_epochs)
        
        if channels == 'all': 
            self.nb_chan = data_epochs.shape[1]
            self.channels = range(self.nb_chan)
        else: 
            self.channels = [chan for chan in channels if chan < data_epochs.shape[1]]
            self.nb_chan = len(self.channels)
            
        self.sync_chan = []
        for chan in self.channels:
            if len(sync_chan) > chan:
                self.sync_chan.append([c for c in sync_chan[chan] if c in self.channels])
            else:
                self.sync_chan.append([chan])
#        else:
#            raise IndexError('sync_chan parameter must contain information for all displayed channels, channels {} are missing.'.format([_ for _ in self.channels if _ >= len(sync_chan)]) )

        for c in self.sync_chan : 
            if any(np.abs(np.diff(c)) > 1 ) & (group_chan): 
                print('Warning: Can not group non contiguous channels, group channels set to False')
                group_chan = False
        if group_chan :
            self.group_chan = [] 
            for group in self.sync_chan: 
                if group not in self.group_chan: self.group_chan.append(group) 
        else: self.group_chan = [range(self.nb_chan)]

        self.vmin_plot = [] 
        self.vmax_plot = [] 
        for c in range(self.nb_chan):
#            self.vmin_plot.append(self.data[:,self.sync_chan[c],:].min()*1.1)
#            self.vmax_plot.append(self.data[:,self.sync_chan[c],:].max()*1.1)        
            self.vmin_plot.append(self.data[:,self.sync_chan[c],:].min() - np.abs(self.data[:,self.sync_chan[c],:].min()*.01))
            self.vmax_plot.append(self.data[:,self.sync_chan[c],:].max() + np.abs(self.data[:,self.sync_chan[c],:].max()*.01))        

        self.movable_1 = code_movable_1
        self.movable_2 = code_movable_2
        
        # Makes sure no movable event is attached to more than one channel
        for e in enumerate(ep_evts.list_evts_trials):
            if any(e[1].find_events(code = [self.movable_1,self.movable_2], chan = -1)):
                raise ValueError("Epoch {}: only events linked to one channel can be movable, please link each {} and {} to only one channel.".format(e[0], self.movable_1,self.movable_2))
        
        global current_epoch 
        current_epoch = 0

        self.event_plot = []
        self.vb = []
        for group in self.group_chan:
            group_graph = pg.GraphicsLayoutWidget()
            for c in group:
                self.event_plot.append(CustomEventPlot(self.movable_1, self.movable_2))
                self.vb.append(self.event_plot[-1].vb)
                self.vb[-1].mousePressEvent = self.event_plot[-1].on_click
#                self.vb[-1].mouseDoubleClickEvent = self.event_plot[-1].onDoubleClick
    #            self.vb[-1].keyPressEvent = self.event_plot[-1].pressKey

                self.event_plot[-1].setXLink(self.event_plot[0])
                self.event_plot[-1].setYLink(self.event_plot[self.channels.index(self.sync_chan[self.channels.index(c)][0])])
    
                group_graph.addItem(self.event_plot[-1])
                group_graph.nextRow()
            self.win.addWidget(group_graph)
#            self.win.nextRow()
            
#            self.event_plot[c].getViewBox().setLimits(yMin = self.data[:,self.sync_chan[c],:].min()*1.1, yMax = self.data[:,self.sync_chan[c],:].max()*1.1)
           
#            if c > 0: 
#                self.event_plot[c].setXLink(self.event_plot[0])
#                self.event_plot[c].setYLink(self.event_plot[self.sync_chan[c][0]])
        
        self.plot_epoch()
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.nb_epoch-1)
        self.combo_chan_scale.clear()
        self.combo_chan_scale.addItems(['all channels'] + [ str(c) for c in range(self.nb_chan)])


    def data_plot(self):
        global current_epoch
        for c in range(self.nb_chan):
            self.event_plot[c].plot(x =self.t, y=self.data[self.list_epochs[current_epoch],self.channels[c],:])
            self.event_plot[c].getViewBox().setYRange(self.vmin_plot[c],self.vmax_plot[c],padding=0)
            self.event_plot[c].getViewBox().setXRange(self.tmin_plot,self.tmax_plot,padding=0)
            
        
    def plot_epoch(self):
        global current_epoch
        for c in range(self.nb_chan):
            chan_events = self.events.list_evts_trials[self.list_epochs[current_epoch]].find_and_get_events(chan=[-1, self.channels[c]], print_find_evt=False)
            self.event_plot[c].clear()
            self.event_plot[c].new_events_epoch(chan_events)
        self.data_plot()
        self.box_go_to.setText(str(self.user_info_epoch[current_epoch]))
        self.slider.setValue(self.user_info_epoch[current_epoch])

    def get_plot_events(self, chan, movable_events=None, data_time_range=True):
        if movable_events is True: selected_lines = [ li[0] for li in enumerate(self.event_plot[chan].list_plot_lines) if (li[1].movable)]
        elif movable_events is False: selected_lines = [ li[0] for li in enumerate(self.event_plot[chan].list_plot_lines) if (li[1].movable is False)]
        else: selected_lines = range(len(self.event_plot[chan].list_plot_lines))
        if data_time_range is True : selected_lines = [ li for li in selected_lines if self.t[0] <= self.event_plot[chan].get_events_pos(li) <= self.t[-1]]
        
        # create latencies to enter time and sample separately below
        evt_lat = Latency(time=self.event_plot[chan].get_events_pos(selected_lines), sf=self.sf)        
        return Events(time=evt_lat.time, sample=evt_lat.sample + self.t0sample, chan=[self.channels[chan]]*len(selected_lines), code=self.event_plot[chan].get_events_code(selected_lines), sf=self.sf)
#        return Events(time = self.event_plot[chan].getEventsPos(selectedLines), chan = [self.channels[chan]] * len(selectedLines), code = self.event_plot[chan].getEventsCode(selectedLines), sf = self.sf)
    
    def update_events(self):
        global current_epoch
        self.events.list_evts_trials[self.list_epochs[current_epoch]].find_and_del_events(code=[self.movable_1,self.movable_2], print_del_evt=False)
        for c in range(self.nb_chan):
            self.events.list_evts_trials[self.list_epochs[current_epoch]].add_events(self.get_plot_events(c, movable_events=True, data_time_range=True))
        if self.write_evt_file : self.write_evt()
            
    def write_evt(self):
        global current_epoch
        f = open(self.fname_evts,'a')
        for t in range(self.events.list_evts_trials[self.list_epochs[current_epoch]].nb_events()):
            f.write(str(current_epoch) + ',' + \
                    str(self.events.list_evts_trials[self.list_epochs[current_epoch]].lat.sample[t]) + ',' + \
                    str(self.events.list_evts_trials[self.list_epochs[current_epoch]].lat.time[t]) + ',' + \
                    str(self.events.list_evts_trials[self.list_epochs[current_epoch]].code[t]) + ',' + \
                    str(self.events.list_evts_trials[self.list_epochs[current_epoch]].chan[t]) + ',' + \
                    str(self.events.tmin.sample[self.list_epochs[current_epoch]]) + ',' + \
                    str(self.events.tmin.time[self.list_epochs[current_epoch]]) + ',' + \
                    str(self.events.t0.sample[self.list_epochs[current_epoch]]) + ',' + \
                    str(self.events.t0.time[self.list_epochs[current_epoch]]) + ',' + \
                    str(self.events.tmax.sample[self.list_epochs[current_epoch]]) + ',' + \
                    str(self.events.tmax.time[self.list_epochs[current_epoch]]) + '\n')
        f.close()
   
    def go_to_prev(self):
        global current_epoch
        self.update_events()
        if current_epoch >0:
            current_epoch-=1
            self.plot_epoch()
            
    def go_to_next(self):
        global current_epoch
        self.update_events()
        if current_epoch + 1 < self.nb_epoch:
            current_epoch+=1
            self.plot_epoch()

    def go_to_box(self):
        global current_epoch
        self.update_events()
        try:
            new_epoch = int(self.box_go_to.text())
        except ValueError: 
#            print('Enter a valid epoch number (type: integer)')
#            self.boxGoTo.setText('Only integer type')
            pass
        if 0 <= new_epoch < self.nb_epoch:
            current_epoch = np.where(self.user_info_epoch == new_epoch)[0][0]
            self.plot_epoch()
 
    def go_to_slider(self):
        global current_epoch
        if self.user_info_epoch[current_epoch] != self.slider.value(): #does everything only if change was made through slider (otherwise it is already done)
            self.update_events()
            current_epoch = np.where(self.user_info_epoch == self.slider.value())[0][0]
            self.plot_epoch()
    
    def set_plot_time(self):
        try:
            self.tmin_plot = float(self.box_show_from.text())
            self.tmax_plot = float(self.box_show_until.text())
        except ValueError: 
            pass
        for c in range(self.nb_chan):
            self.event_plot[c].getViewBox().setXRange(self.tmin_plot,self.tmax_plot,padding=0)

    def set_chan_scale(self):
        try: self.chan_scale = self.sync_chan[int(self.combo_chan_scale.currentText())]
        except: self.chan_scale = range(self.nb_chan)
        self.set_vertical_scale()
            
    def set_vertical_scale(self):
        try: 
            for c in self.chan_scale: self.vmin_plot[c] = float(self.box_vmin.text())
        except: pass
        try: 
            for c in self.chan_scale: self.vmax_plot[c] = float(self.box_vmax.text())
        except: pass

        for c in self.chan_scale:
            self.event_plot[c].getViewBox().setYRange(self.vmin_plot[c],self.vmax_plot[c],padding=0)
            
    def reset_scale(self):
        for c in range(self.nb_chan):
#            self.vmin_plot[c] = self.data[:,self.sync_chan[c],:].min()*1.1
#            self.vmax_plot[c] = self.data[:,self.sync_chan[c],:].max()*1.1        
            self.vmin_plot[c] = self.data[:,self.sync_chan[c],:].min() - np.abs(self.data[:,self.sync_chan[c],:].min()*.01)
            self.vmax_plot[c] = self.data[:,self.sync_chan[c],:].max() + np.abs(self.data[:,self.sync_chan[c],:].max()*.01)   

            self.event_plot[c].getViewBox().setYRange(self.vmin_plot[c],self.vmax_plot[c],padding=0)
        self.box_vmin.clear()
        self.box_vmax.clear()
        self.combo_chan_scale.setCurrentIndex(0)

    def set_keyboard_prev_next(self):        
        key_idx = self.combo_prev_next.currentIndex()
        if key_idx == 0 : 
            self.key_go_to_prev = QtCore.Qt.Key_Alt
            self.key_go_to_next = QtCore.Qt.Key_Space
        elif key_idx == 1 : 
            self.key_go_to_prev = QtCore.Qt.Key_Left
            self.key_go_to_next = QtCore.Qt.Key_Right
        elif key_idx == 2 : 
            self.key_go_to_prev = QtCore.Qt.Key_A
            self.key_go_to_next = QtCore.Qt.Key_Z
        elif key_idx == 3 : 
            self.key_go_to_prev = QtCore.Qt.Key_Q
            self.key_go_to_next = QtCore.Qt.Key_W
    
    def delete_movable(self):
        global current_epoch
        for c in range(self.nb_chan):
            movable_lines = [ line for line in self.event_plot[c].list_plot_lines if line.movable ]
            for line in movable_lines: self.event_plot[c].remove_event(line)

    def reset_movable(self):
        global current_epoch
        self.delete_movable()
        for c in range(self.nb_chan):
            chan_events = self.events.list_evts_trials[self.list_epochs[current_epoch]].find_and_get_events(chan=[self.channels[c]], print_find_evt=False)
            self.event_plot[c].new_events_epoch(chan_events)        
        
    def initUi(self):
        ## Define a top-level widget to hold everything
        global current_epoch        
#        self.w = QtGui.QWidget()

        ## Create some widgets to be placed inside
        tab_widget = QTabWidget()
        
        navigation_widget = QtGui.QWidget()
        navigation_layout = QtGui.QGridLayout()
        self.btn_back = QtGui.QPushButton('<<')
        self.btn_back.clicked.connect(self.go_to_prev)
        self.btn_fwd = QtGui.QPushButton('>>')
        self.btn_fwd.clicked.connect(self.go_to_next)

        label_current = QtGui.QLabel('Epoch ', alignment = QtCore.Qt.AlignCenter)
        self.box_go_to = QtGui.QLineEdit(str(current_epoch))
        self.box_go_to.returnPressed.connect(self.go_to_box)
 
        self.btn_delete_mrk = QtGui.QPushButton('Delete all markers')
        self.btn_delete_mrk.clicked.connect(self.delete_movable)
        self.btn_reset_mrk = QtGui.QPushButton('Reset markers')
        self.btn_reset_mrk.clicked.connect(self.reset_movable)
        
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.go_to_slider)
        
        navigation_layout.addWidget(self.btn_back, 1, 0, 1, 3)
        navigation_layout.addWidget(label_current, 1, 3, 1, 1)   # label goes in down-middle
        navigation_layout.addWidget(self.box_go_to, 1, 4, 1 , 1)   # text box goes in down-middle
        navigation_layout.addWidget(self.btn_fwd, 1, 5, 1, 3)   # button goes in down-right
        navigation_layout.addWidget(self.btn_delete_mrk, 2, 0, 1, 4)   # button goes in down-right
        navigation_layout.addWidget(self.btn_reset_mrk, 2, 4, 1, 4)   # button goes in down-right
        navigation_layout.addWidget(self.slider, 3, 0, 1, 8)   # button goes in down-right

        navigation_layout.setColumnStretch(0,1)
        navigation_layout.setColumnStretch(1,1)
        navigation_layout.setColumnStretch(2,1)
        navigation_layout.setColumnStretch(3,1)
        navigation_layout.setColumnStretch(4,1)
        navigation_layout.setColumnStretch(5,1)
        navigation_layout.setColumnStretch(6,1)
        navigation_layout.setColumnStretch(7,1)
        
        navigation_widget.setLayout(navigation_layout)
        tab_widget.addTab(navigation_widget,'Navig.')
                   
        config_widget = QtGui.QWidget()
        config_layout = QtGui.QGridLayout()
        
        label_time_range= QtGui.QLabel('Time range:', alignment = QtCore.Qt.AlignCenter)
        self.box_show_from = QtGui.QLineEdit(self.tmin_plot)
        self.box_show_from.setPlaceholderText('Xmin')
        self.box_show_from.returnPressed.connect(self.set_plot_time)
        self.box_show_until = QtGui.QLineEdit(self.tmax_plot)
        self.box_show_until.setPlaceholderText('Xmax')
        self.box_show_until.returnPressed.connect(self.set_plot_time)

        label_vertical_scale = QtGui.QLabel('Scale range:', alignment = QtCore.Qt.AlignCenter)
        self.combo_chan_scale = QtGui.QComboBox()
        self.combo_chan_scale.addItems(['all channels'] + [ str(c) for c in range(self.nb_chan)])
        self.combo_chan_scale.currentIndexChanged.connect(self.set_chan_scale)
        self.box_vmin = QtGui.QLineEdit()
        self.box_vmin.setPlaceholderText('Ymin')
        self.box_vmin .returnPressed.connect(self.set_vertical_scale)
        self.box_vmax = QtGui.QLineEdit()
        self.box_vmax.setPlaceholderText('Ymax')
        self.box_vmax .returnPressed.connect(self.set_vertical_scale)
        self.btn_reset_scale = QtGui.QPushButton('Reset scale')
        self.btn_reset_scale.clicked.connect(self.reset_scale)


        label_prev_next= QtGui.QLabel('Go to Prev / Next:', alignment = QtCore.Qt.AlignCenter)
        self.combo_prev_next = QtGui.QComboBox()
        self.combo_prev_next.addItems(['Alt / space bar'] + [u'\u2190 / \u2192'] + ['A / Z'] + ['Q / W'])
        self.combo_prev_next.currentIndexChanged.connect(self.set_keyboard_prev_next)

        config_layout.addWidget(label_time_range, 0, 0, 1, 1)   # button goes in down-right
        config_layout.addWidget(self.box_show_from, 0, 1, 1, 1)   # button goes in down-right
        config_layout.addWidget(self.box_show_until, 0, 2, 1, 1)   # button goes in down-right

        config_layout.addWidget(label_vertical_scale, 0, 3, 1, 1)   # button goes in down-right
        config_layout.addWidget(self.combo_chan_scale, 0, 4, 1, 1)   # button goes in down-right
        config_layout.addWidget(self.box_vmin, 0, 5, 1, 1)   # button goes in down-right
        config_layout.addWidget(self.box_vmax, 0, 6, 1, 1)   # button goes in down-right
        config_layout.addWidget(self.btn_reset_scale, 0, 7, 1, 1)   # button goes in down-right

        config_layout.addWidget(label_prev_next, 1, 0, 1, 1)   # button goes in down-right
        config_layout.addWidget(self.combo_prev_next, 1, 1, 1, 1)   # button goes in down-right

        config_widget.setLayout(config_layout)
        tab_widget.addTab(config_widget,'Config.')

        self.listw = QtGui.QListWidget()
        
        self.win = QtGui.QSplitter(QtCore.Qt.Vertical)
#        for group in self.group_chan:
#            self.win.addWidget(pg.GraphicsLayoutWidget)
        pg.setConfigOptions(antialias=True)
        
        ## Create a grid layout to manage the widgets size and position
        layout = QtGui.QHBoxLayout()
        v_splitter = QtGui.QSplitter(QtCore.Qt.Vertical)
        
        ## Add widgets to the layout in their proper positions
#        layout.addWidget(self.win,0,0,1,4)  # plot goes on right side, spanning 4 rows
#        layout.addWidget(tabWidget,1,0,4,4)   # button goes in down-left
        v_splitter.addWidget(self.win)  # plot goes on right side, spanning 4 rows
        v_splitter.addWidget(tab_widget)   # button goes in down-left

        layout.addWidget(v_splitter)
#        self.w.setLayout(layout)
        self.setLayout(layout)




class VizApplication(QApplication):
    
    def __init__(self,args):
        app = QtCore.QCoreApplication.instance()
        if app is None:
#        app = QApplication(sys.argv) 
            QApplication.__init__(self,args)
        self.window = VizUi()
        
    def load_data(self, data_epochs, times, ep_evts, channels='all', code_movable_1='onset', code_movable_2='offset', sync_chan=[[0,1],[0,1],[2,3],[2,3]], group_chan=True, list_trials=None, random_order=False):
        """Loads data and sets viz window view.
    
        Parameters:
        -----------
        data_epochs : 3D array
            Data array, dimensions should be trials x channels x time .
        times : 1D array 
            Times vector for one trial (length must correspond to data_epochs 
            3rd dimension).
        ep_evts : ep_evts object
            ep_evts corresponding to data_epochs (number of trials must
            correspond to data_epochs 1st dimension).
        channels : list of int
            List of channel indices to represent, if equal to 'all', all channels
            of data_epochs are used (default is 'all').
        code_movable_1 : int | str | float
            Code of events with which the user can interact. All events whose code
            is equal to 'code_movable_1' are displayed in dark blue and can be 
            moved (left click + drag), deleted (right click), or inserted 
            (Ctrl + left click) (default is 'onset').
        code_movable_2 : int | str | float
            Code of events with which the user can interact. All events whose code
            is equal to 'code_movable_2' are displayed in light blue and can be 
            moved (left click + drag), deleted (right click), or inserted 
            (Ctrl + right click) (default is 'offset').
        sync_chan : list
            List of channels to synchronise vertical scales. The length of sync_chan
            should be equal to the number of displayed channels, and specify, for
            each channel, the list of channels to synchronize. For example, the
            default list [[0,1],[0,1],[2,3],[2,3]] synchronizes channels 0 and 1,
            and channels 2 and 3. Set sync_chan to [[0],[1],[2],[3]] for no
            synchronization of vertical scales (default is [[0,1],[0,1],[2,3],[2,3]]).
        group_chan : bool
            if True, group channels plots of synchronized channels. Can only apply
            to contiguous channels, if non contiguous channels are synchronized,
            group_chan is set to False (default is True).
        list_trials : list
            List of trials to display, if None, display all trials (defauls is
            None).
        random_order: bool
            If True, display trials in random order (note that epoch number in
            viz window does not correspond to true epoch number in that case). 
        """

        self.window.load_data(data_epochs, times, ep_evts, channels=channels, code_movable_1=code_movable_1 , code_movable_2=code_movable_2 , sync_chan=sync_chan, group_chan=group_chan, list_trials=list_trials, random_order=random_order)
        
    def show(self):
#        self.window.w.show()
        self.window.show()
#        sys.exit(self.exec_())
        self.exec_() # For Spyder Users
        
    def get_events(self):
        return self.window.events

    
def main():

#    app = QApplication(sys.argv)
    ex = VizApplication(sys.argv)
    ex.show()

if __name__ == '__main__':
    main()