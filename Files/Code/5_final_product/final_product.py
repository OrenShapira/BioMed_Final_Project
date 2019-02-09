#%% imports and defines

# imports
from tkinter import *
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import tkinter as tk
import os
import glob
import matplotlib
import ctypes
import matplotlib.pyplot as plt
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import ast
from imutils import face_utils
import math
import imutils
import time
import dlib
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from imutils.video import FPS
from statistics import mean
from joblib import load

# define color list
c_red = (255,0,0)
c_lime = (0,255,0)
c_blue = (0,0,255)
c_black = (0,0,0)
c_magenta = (255,0,255)

# define text position lists
text_pos_r = []
text_pos_l = []
text_pos_gap = 40
line_styles = ['-', '--', '-.', ':']

#%% GUI class

# noinspection PyUnusedLocal
class LabelTool:
    def __init__(self, master):
        
        # set up the main frame
        self.parent = master
        self.parent.title("LabelTool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=True, height=True)

        # initialize global state
        self.imageList = []
        self.cur = 0
        self.total = 0
        self.tkImg = None
        self.palsy_eye_list = ['none','left','right']
        self.performance_list = ['Offline','Real-Time']
        self.dir_is_active = 0
        self.currentLabelIndex = 0
        self.precision_digits = 2
        
        for i in range(0,1):
            text_pos_r.append((10,50+i*text_pos_gap))
            text_pos_l.append((700,50+i*text_pos_gap))
        
        # prep for zoom
        self.zoom = 1
        
        # directories paths
        self.filesDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'files')
        self.databaseDir = os.path.join(os.path.dirname(os.path.abspath(os.path.join(__file__,"../.."))),'Database')
        
        # ----------------- GUI stuff ---------------------
        
        # upper panel for path and configurations (dir, media, palsy eye, performance)
        self.upperPanel = Frame(self.frame)
        self.upperPanel.grid(row=0, column=1, sticky=W + E)
        
        self.label = Label(self.frame, text="Dir Path:", width=7)
        self.label.grid(row=0, column=0, sticky=W + E)
        self.dirEntry = Entry(self.upperPanel, width = 100)
        self.dirEntry.insert(END, self.filesDir)
        self.dirEntry.config(state='disabled')
        self.dirEntry.pack(side=LEFT, padx=5, pady=3)
        
        self.label = Label(self.upperPanel, text="Media:", width=10)
        self.label.pack(side=LEFT, padx=5, pady=3)
        self.media_list_files = [f.name for f in os.scandir(self.filesDir) if f.is_dir()]
        self.media_list_database = [f[:-4] for f in os.listdir(self.databaseDir) if f.endswith(".mp4")]
        self.media_list = self.media_list_database + list(set(self.media_list_files) - set(self.media_list_database))
        self.media_list.insert(0,'webcam')
        self.mediaName = StringVar()
        self.mediaName.set(self.media_list[0])
        self.mediaList = OptionMenu(self.upperPanel, self.mediaName, *self.media_list, command=self.setMedia)
        self.currentMedia = self.mediaName.get()  # init
        self.mediaList.pack(side=LEFT, padx=5, pady=3)
        self.mediaList.config(width=25)
        
        self.label = Label(self.frame, text="Palsy Eye:", width=10)
        self.label.grid(row=0, column=2, sticky=W + E)
        self.palsyEye = StringVar()
        self.palsyEye.set(self.palsy_eye_list[0])
        self.palsyEyeList = OptionMenu(self.frame, self.palsyEye, *self.palsy_eye_list, command=self.setPalsyEye)
        self.currentPalsyEye = self.palsyEye.get()  # init
        self.palsyEyeList.grid(row=0, column=3)
        self.palsyEyeList.config(width=10)
        
        self.label = Label(self.frame, text="Performance:", width=10)
        self.label.grid(row=0, column=4, sticky=W + E)
        self.performanceName = StringVar()
        self.performanceName.set(self.performance_list[0])
        self.performanceList = OptionMenu(self.frame, self.performanceName, *self.performance_list, command=self.setPerformance)
        self.currentPerformance = self.performanceName.get()  # init
        self.performanceList.grid(row=0, column=5)
        self.performanceList.config(width=10)
        
        self.ldBtn = Button(self.frame, text="Run Media", width=10, background="green1", command=self.runMedia)
        self.ldBtn.grid(row=1, column=2, rowspan=1, columnspan=2, sticky=W + E)
        self.btnClass = Button(self.frame, text='Load Offline', width=10, background="yellow", command=self.loadOffline)
        self.btnClass.grid(row=1, column=4, rowspan=1, columnspan=2, sticky=W + E)        
        self.btnClass = Button(self.frame, text='Show Plots', width=10, background="cyan1", command=self.showPlots)
        self.btnClass.grid(row=2, column=2, rowspan=1, columnspan=2, sticky=W + E)
        self.ldBtn = Button(self.frame, text="Clear All", width=10, background="red", command=self.clearAll)
        self.ldBtn.grid(row=2, column=4, rowspan=1, columnspan=2, sticky=W + E)

        # center left panel for showing frames and graphs
        self.mainPanel = Canvas(self.frame, background="white", width=900, height=600)
        self.mainPanel.grid(row=1, column=1, rowspan=3, columnspan=1, sticky=W + N)
        
        # center right panel for log and graphs
        self.listbox = Listbox(self.frame, background="white", width=60, height=34)
        self.listbox.grid(row=3, column=2, rowspan=1, columnspan=4, sticky=W + N)
        
        # configuration panel - line1
        self.ctrPanel1 = Frame(self.frame)
        self.ctrPanel1.grid(row=7, column=1, rowspan=1, columnspan=1, sticky=W + E)
        
        # configuration panel - line1 (checkboxes)
        self.cbFeaturePalsyVar = IntVar(value=1)
        self.cbFeatureEarVar = IntVar(value=1)
        self.cbFeatureR2Var = IntVar(value=0)
        self.cbFeatureEllipseVar = IntVar(value=0)       
        self.cbFeaturePolyVar = IntVar(value=1)
        self.cbEstimateLabelVar = IntVar(value=1)

        self.cbShowFeaturesVar = IntVar(value=0)
        self.cbShowFeatures = Checkbutton(self.ctrPanel1,text="Show Features",variable=self.cbShowFeaturesVar)
        self.cbShowFeatures.pack(side=LEFT, padx=5, pady=3)
        self.cbShowFaceBboxVar = IntVar(value=0)
        self.cbShowFaceBbox = Checkbutton(self.ctrPanel1,text="Show Bbox",variable=self.cbShowFaceBboxVar)
        self.cbShowFaceBbox.pack(side=LEFT, padx=5, pady=3)
        self.cbShowScoreGraphVar = IntVar(value=1)
        self.cbShowScoreGraph = Checkbutton(self.ctrPanel1,text="Show Score Graph",variable=self.cbShowScoreGraphVar)
        self.cbShowScoreGraph.pack(side=LEFT, padx=5, pady=3)
        self.cbShowFeaturesGraphVar = IntVar(value=0)
        self.cbShowFeaturesGraph = Checkbutton(self.ctrPanel1,text="Show Features Graph",variable=self.cbShowFeaturesGraphVar)
        self.cbShowFeaturesGraph.pack(side=LEFT, padx=5, pady=3)
        
        self.cbSaveTimingVar = IntVar(value=0)
        self.cbSaveTiming = Checkbutton(self.ctrPanel1,text="Save Timing",variable=self.cbSaveTimingVar)
        self.cbSaveTiming.pack(side=LEFT, padx=5, pady=3)
        self.cbSaveRawWebcamVar = IntVar(value=0)
        self.cbSaveRawWebcam = Checkbutton(self.ctrPanel1,text="Save Webcam",variable=self.cbSaveRawWebcamVar)
        self.cbSaveRawWebcam.pack(side=LEFT, padx=5, pady=3)
        self.cbSaveVideoFeaturesVar = IntVar(value=1)
        self.cbSaveVideoFeatures = Checkbutton(self.ctrPanel1,text="Save Video",variable=self.cbSaveVideoFeaturesVar)
        self.cbSaveVideoFeatures.pack(side=LEFT, padx=5, pady=3)
        self.cbSaveFrameFeaturesVar = IntVar(value=1)
        self.cbSaveFrameFeatures = Checkbutton(self.ctrPanel1,text="Save Frames",variable=self.cbSaveFrameFeaturesVar)
        self.cbSaveFrameFeatures.pack(side=LEFT, padx=5, pady=3)
        
        # configuration panel - line2
        self.ctrPanel2 = Frame(self.frame)
        self.ctrPanel2.grid(row=8, column=1, rowspan=1, columnspan=1, sticky=W + E)
        
        # # configuration panel - line2 (enteries)
        self.goBtn = Button(self.ctrPanel2, text='Go', background="magenta", command=self.gotoImage)
        self.goBtn.pack(side=RIGHT)
        self.idxEntry = Entry(self.ctrPanel2, width=5)
        self.idxEntry.pack(side=RIGHT, padx=5)
        self.tmpLabel = Label(self.ctrPanel2, text="Go to")
        self.tmpLabel.pack(side=RIGHT, padx=5)
        self.progLabel = Label(self.ctrPanel2, text="Progress:     /    ")
        self.progLabel.pack(side=RIGHT, padx=5)
        
        self.WinLenLabel = Label(self.ctrPanel2, text="Win Len")
        self.WinLenLabel.pack(side=LEFT, padx=5)
        self.WinLenEntry = Entry(self.ctrPanel2, width=4)
        self.WinLenEntry.insert(END,'2')
        self.WinLenEntry.pack(side=LEFT, padx=4)
        self.FaceSearchLabel = Label(self.ctrPanel2, text="Face Search")
        self.FaceSearchLabel.pack(side=LEFT, padx=5)
        self.FaceSearchEntry = Entry(self.ctrPanel2, width=4)
        self.FaceSearchEntry.insert(END,'1')
        self.FaceSearchEntry.pack(side=LEFT, padx=4)
        self.WebcamTimeLabel = Label(self.ctrPanel2, text="Webcam Time")
        self.WebcamTimeLabel.pack(side=LEFT, padx=5)
        self.WebcamTimeEntry = Entry(self.ctrPanel2, width=4)
        self.WebcamTimeEntry.insert(END,'10')
        self.WebcamTimeEntry.pack(side=LEFT, padx=4)
        self.BlinkThLabel = Label(self.ctrPanel2, text="Blink Th")
        self.BlinkThLabel.pack(side=LEFT, padx=5)
        self.BlinkThEntry = Entry(self.ctrPanel2, width=4)
        self.BlinkThEntry.insert(END,'0.25')
        self.BlinkThEntry.pack(side=LEFT, padx=4)
        self.BlinkConsecLabel = Label(self.ctrPanel2, text="Blink Consec")
        self.BlinkConsecLabel.pack(side=LEFT, padx=5)
        self.BlinkConsecEntry = Entry(self.ctrPanel2, width=4)
        self.BlinkConsecEntry.insert(END,'2')
        self.BlinkConsecEntry.pack(side=LEFT, padx=4)
        self.BlinkBufferLabel = Label(self.ctrPanel2, text="Blink Buffer")
        self.BlinkBufferLabel.pack(side=LEFT, padx=5)
        self.BlinkBufferEntry = Entry(self.ctrPanel2, width=4)
        self.BlinkBufferEntry.insert(END,'4')
        self.BlinkBufferEntry.pack(side=LEFT, padx=4)
        
        # right low panel for key menu (60 characters per line)
        self.keyMenuText = "                                ** Key Menu **                       \r"\
        "     -Left (←): Go backforward                    -Right (→): Go forward   \r"\
        "  -BackSpace: Zoom out                         -Space: Zoom in          \r"
        
        self.label = Label(self.frame, text=self.keyMenuText)
        self.label.grid(row=7, column=2, rowspan=2, columnspan=4, sticky=W)
        
        # key binding
        self.parent.bind("<Left>", self.prevImage)         # press left arrow to go backforward
        self.parent.bind("<Right>", self.nextImage)        # press right arrow to go forward
        self.parent.bind("<BackSpace>", self.zoomOut)      # press backspace to zoom out
        self.parent.bind("<space>", self.zoomIn)           # press space to zoom in
    
    #%% define function
    
    def loadOffline(self):
        "load video frames and graph for offline analysis"
        
        # get working directory
        s = self.dirEntry.get()
        self.workingDir = os.path.join(self.filesDir,self.currentMedia)

        # get image list
        self.imageDir = os.path.join(self.workingDir,'frames_features')
        self.imageList = [f for f_ in [glob.glob(os.path.join(self.imageDir, e), recursive=True) for e in
                                       ('.\**\*.JPG', '.\**\*.PNG')] for f in f_]
        
        # cleaning
        self.clearAll()
        self.destroyAll()        
        self.plotCurrEyePanel = Canvas(self.frame, background="white", width=360, height=550)
        self.plotCurrEyePanel.grid(row=3, column=2, rowspan=1, columnspan=4, sticky=W + N)
        
        # check if path includes images
        if len(self.imageList) == 0:
            print('Dir load failed! No images found in the specified dir!')
            ctypes.windll.user32.MessageBoxW(0,'Dir load failed! No images found in the specified dir!',"Message",0)
            return
        
        # load blink bounds
        self.logBlinksFilePath = self.workingDir+'\\'+self.currentMedia+"_log_blinks.txt"
        if os.path.exists(self.logBlinksFilePath):
            # load from file
            logBlinksFile = open(self.logBlinksFilePath,'r')
            temp_data = logBlinksFile.readlines()                
            logBlinksFile.close()
            
            self.blink_details = []
            for line in temp_data:
                self.blink_details.append(list(ast.literal_eval(line[1:-2])))
        else:
            print('Show blink bound failed! No files found in the specified dir!')
            ctypes.windll.user32.MessageBoxW(0,'Show blink bound failed! No files found in the specified dir!',"Message",0)
            return
            
        # default to the 1st image in the collection
        self.cur = 0
        self.total = len(self.imageList)
        self.dir_is_active = 1
        self.video_fps = int(self.currentMedia[-5:-3])        

        # load frame
        self.loadImage()
        ctypes.windll.user32.MessageBoxW(0,'Dir load completed! %d images loaded from %s!' % (self.total, self.currentMedia),"Message",0)
    
    #%% define function
    def loadImage(self):
        "load frame and update marker point on the graph, the function is call from loadOffline function and used for offline analysis"
        
        # get working directory
        s = self.dirEntry.get()
        self.workingDir = os.path.join(self.filesDir,self.currentMedia)
        self.labelDir = self.workingDir
        
        # load image from image list
        imagePath = self.imageList[self.cur]
        self.img = Image.open(imagePath)
        self.img = self.img.resize([int(self.zoom * s) for s in self.img.size])
        self.mainPanel.config(width=900, height=600)
        self.progLabel.config(text="Progress:  %04d/%04d" % (self.cur, self.total-1))
        self.tkImg = ImageTk.PhotoImage(self.img)
        self.mainPanel.create_image(0, 0, image=self.tkImg, anchor=NW)

        # load dataframe
        self.load_xlsx = pd.ExcelFile(self.labelDir+'\\'+self.currentMedia+"_scores_n.xlsx")
        self.load_sheet_left = pd.read_excel(self.load_xlsx,'left_eye')
        self.load_sheet_right = pd.read_excel(self.load_xlsx,'right_eye')        
        
        self.video_fps = int(self.currentMedia[-5:-3])
        
        # create column of frame number and elapsed time and append it to data frame
        self.load_sheet_left['frame_number'] = self.load_sheet_left.index
        self.load_sheet_right['frame_number'] = self.load_sheet_right.index
        self.load_sheet_left['elapsed_time'] =  self.load_sheet_left['frame_number'].apply(lambda x: round(float(x/self.video_fps),self.precision_digits))
        self.load_sheet_right['elapsed_time'] =  self.load_sheet_right['frame_number'].apply(lambda x: round(float(x/self.video_fps),self.precision_digits))
        
        df_frame_number = self.load_sheet_left['frame_number']
        df_frame_elapsed_time = self.load_sheet_left['elapsed_time']
        
        # close all figures before creating new one
        matplotlib.pyplot.close('all')
        self.figure = plt.figure(figsize=(4.5,9),dpi=70)
        
        if (self.currentPalsyEye == 'left'):
            c_right_plot = c_blue
            c_left_plot = c_blue
        elif (self.currentPalsyEye == 'right'):
            c_right_plot = c_blue
            c_left_plot = c_blue
        else:
            c_right_plot = c_blue
            c_left_plot = c_blue
        
        # plot graph for right eye
        ax_right = plt.subplot(1,2,1)
        load_sheet_right_ax = self.load_sheet_right.plot(kind='line',style=line_styles[0],x='estimate_label',y='frame_number',
                       color=tuple(np.divide(c_right_plot,255)),ax=ax_right,label='score')
        plt.scatter(self.load_sheet_right['estimate_label'][self.cur], self.cur, c=tuple(np.divide(c_black,255)),zorder=3)
        
        # edit titles and axis
        plt.title(self.currentMedia+': score vs frame_number',loc='left')
        plt.xlabel("score_right")
        plt.ylabel('frame_number')
        ax_right.set_xlim([0,1])
        ax_right.set_ylim([df_frame_number.iloc[0],df_frame_number.iloc[-1]])
        ax_right.legend(loc='upper left')
        
        # plot graph for left eye
        ax_left = plt.subplot(1,2,2)
        load_sheet_left_ax = self.load_sheet_left.plot(kind='line',style=line_styles[0],x='estimate_label',y='frame_number',
                       color=tuple(np.divide(c_left_plot,255)),ax=ax_left,label='score')
        plt.scatter(self.load_sheet_left['estimate_label'][self.cur], self.cur, c=tuple(np.divide(c_black,255)),zorder=3) 
        
        # plot blinks bounds
        if self.currentPalsyEye == 'left':
            for i in range(len(self.blink_details)):
                load_sheet_left_ax.axhspan(self.blink_details[i][1],self.blink_details[i][2], facecolor='y',zorder=1)
                load_sheet_left_ax.axhspan(self.blink_details[i][1],self.blink_details[i][2],self.blink_details[i][3],self.blink_details[i][4],facecolor='r',zorder=2)
                load_sheet_right_ax.axhspan(self.blink_details[i][1],self.blink_details[i][2], facecolor=tuple(np.divide(c_lime,255)),zorder=1)
        elif self.currentPalsyEye == 'right':
            for i in range(len(self.blink_details)):
                load_sheet_left_ax.axhspan(self.blink_details[i][1],self.blink_details[i][2], facecolor=tuple(np.divide(c_lime,255)),zorder=1)
                load_sheet_right_ax.axhspan(self.blink_details[i][1],self.blink_details[i][2], facecolor='y',zorder=1)
                load_sheet_right_ax.axhspan(self.blink_details[i][1],self.blink_details[i][2],self.blink_details[i][3],self.blink_details[i][4],facecolor='r',zorder=2)
        else:
            for i in range(len(self.blink_details)):
                load_sheet_left_ax.axhspan(self.blink_details[i][1],self.blink_details[i][2], facecolor=tuple(np.divide(c_lime,255)),zorder=1)
                load_sheet_right_ax.axhspan(self.blink_details[i][1],self.blink_details[i][2], facecolor=tuple(np.divide(c_lime,255)),zorder=1)
        
        # edit titles and axis
        plt.xlabel('score_left')
        ax_left.set_xlim([0,1])
        ax_left.set_ylim([df_frame_number.iloc[0],df_frame_number.iloc[-1]])
        ax_left.legend(loc='upper left')
        ax_left.get_yaxis().set_visible(False)
        
        # set second y-axis (elapsed_time)
        ay2 = ax_left.twinx()
        # set the ticklabel position in the second x-axis, then convert them to the position in the first x-axis
        ay2_ticks_num = 6
        newlabel = [round((y*df_frame_elapsed_time.iloc[-1]/(ay2_ticks_num-1)),self.precision_digits) for y in range(0, ay2_ticks_num)]
        newpos = [int(np.ceil(y*self.video_fps)) for y in newlabel]
        
        # set the second y-axis
        ay2.set_yticks(newpos)
        ay2.set_yticklabels(newlabel)
        ay2.yaxis.set_ticks_position('right') 
        ay2.yaxis.set_label_position('right')
        ay2.spines['right'].set_position(('outward', 0))
        ay2.set_ylabel('elapsed_time [Sec]')
        
        plt.gcf().subplots_adjust(right=0.87)
        
        # load image
        self.plotCurrEyePanel.config(width=360, height=550)
        figure_canvas_agg = FigureCanvasAgg(self.figure)
        figure_canvas_agg.draw()
        figure_x, figure_y, figure_w, figure_h = self.figure.bbox.bounds
        figure_w, figure_h = int(figure_w+100), int(figure_h+100)
        self.figurePhoto = tk.PhotoImage(master=self.plotCurrEyePanel, width=figure_w, height=figure_h)
        self.plotCurrEyePanel.create_image(figure_w/2 + 25,figure_h/2 - 50,image=self.figurePhoto)
        tkagg.blit(self.figurePhoto, figure_canvas_agg.get_renderer()._renderer, colormode=2)
    
    #%% define function
    def levelToPercentage(self, level):
        "convert level to precentage"
        
        precentageFloat = level*100
        precentageString = "%.0f%s" % (precentageFloat,'%')
        return precentageString
    
    #%% define function
    def clearAll(self):
        "clear main panel and canvases"
        
        self.mainPanel.delete("all")
        try:
            self.plotCurrEyePanel.delete("all")
        except:
            self.listbox.delete(0, 'end')
    
    #%% define function
    def destroyAll(self):
        "destroy canvas object"

        try:
            self.plotCurrEyePanel.destroy()
        except:
            self.listbox.destroy()
        
    #%% define function
    def prevImage(self, event=None):
        "move to previous frame - used in offline analysis"
        
        if self.cur == 0:
            self.cur = self.total-1
        else:
            self.cur -= 1
        
        try:
            self.loadImage()
        except:
            print('Operation failed! Please load valid frames dir!')
            ctypes.windll.user32.MessageBoxW(0,'Operation failed! Please load valid frames dir!',"Message",0)
    
    #%% define function
    def nextImage(self, event=None):
        "move to next frame - used in offline analysis"
        
        if self.cur == self.total-1:
            self.cur = 0
            print('Last frame!')
            ctypes.windll.user32.MessageBoxW(0,'Last frame!',"Message",0)
            
        else:
            self.cur += 1    
        
        try:
            self.loadImage()
        except:
            print('Operation failed! Please load valid frames dir!')
            ctypes.windll.user32.MessageBoxW(0,'Operation failed! Please load valid frames dir!',"Message",0)
    
    #%% define function
    def gotoImage(self):
        "move to specific frame - used in offline analysis"
        
        idx = int(self.idxEntry.get())
        if 0 <= idx <= self.total-1:
            self.cur = idx
            try:
                self.loadImage()
            except:
                print('Operation failed! Please load valid frames dir!')
                ctypes.windll.user32.MessageBoxW(0,'Operation failed! Please load valid frames dir!',"Message",0)
    
    #%% define function
    def setPerformance(self, event=None):
        "set performance"
        
        self.currentPerformance = self.performanceName.get()
        print('set current performance to :', self.currentPerformance)
    
    #%% define function
    def setPalsyEye(self, event=None):
        "set palsy eye"
        
        self.currentPalsyEye = self.palsyEye.get()
        print('set current palsy eye to :', self.currentPalsyEye)
    
    #%% define function
    def setMedia(self, event=None):
        "set media to work with"
        
        self.currentMedia = self.mediaName.get()
        print('set current media to :', self.currentMedia)
        if (self.currentMedia == self.media_list[0]):   # webcam
            self.palsyEyeList.config(state='normal')
            self.palsyEye.set(self.palsy_eye_list[0])
        else:                                           # video
            self.palsyEyeList.config(state='disabled')
            if (self.currentMedia[-7] == 'l'):
                self.palsyEye.set(self.palsy_eye_list[1])
            elif (self.currentMedia[-7] == 'r'):
                self.palsyEye.set(self.palsy_eye_list[2])
            else:
                self.palsyEye.set(self.palsy_eye_list[0])
        
        self.currentPalsyEye = self.palsyEye.get()
        print('set current palsy eye to :', self.currentPalsyEye)
    
    #%% define function  
    def showPlots(self, event=None):
        "show graph and log"
              
        # get working directory
        s = self.dirEntry.get()
        self.workingDir = os.path.join(self.filesDir,self.currentMedia)
        self.labelDir = self.workingDir
        
        # cleaning
        self.clearAll()
        self.destroyAll()
        self.listbox = Listbox(self.frame, background="white", width=60, height=34)
        self.listbox.grid(row=3, column=2, rowspan=1, columnspan=4, sticky=W + N)
        
        # show graph plot
        self.scores_graphs_path = self.labelDir+'\\'+self.currentMedia+"_scores_graphs.png"
        if os.path.exists(self.scores_graphs_path):
            self.img = Image.open(self.scores_graphs_path)            
            resize_ratio = 900/self.img.size[0]
            self.img = self.img.resize([int(resize_ratio * s) for s in self.img.size])
            self.tkImg = ImageTk.PhotoImage(self.img)
            self.mainPanel.create_image(0, 0, image=self.tkImg, anchor=NW)
        else:
            print('Show plots failed! No files found in the specified dir!')
            ctypes.windll.user32.MessageBoxW(0,'Show plots failed! No files found in the specified dir!',"Message",0)
            return
        
        # show log
        self.logFilePath = self.labelDir+'\\'+self.currentMedia+"_log.txt"
        if os.path.exists(self.logFilePath):
            # load from file
            logFile = open(self.logFilePath,'r')
            data = logFile.readlines()                
            for i in range(len(data)):
                self.listbox.insert(END, data[i])
            logFile.close()
        else:
            print('Show log failed! No files found in the specified dir!')
            ctypes.windll.user32.MessageBoxW(0,'Show log failed! No files found in the specified dir!',"Message",0)
            return
    
    #%% define function
    def zoomIn(self, event=None):
        "zooming frame in - used in offline analysis"
        
        self.zoom *= 1.2
        try:
            self.loadImage()
        except:
            print('Operation failed! Please load valid frames dir!')
            ctypes.windll.user32.MessageBoxW(0,'Operation failed! Please load valid frames dir!',"Message",0)
    
    #%% define function
    def zoomOut(self, event=None):
        "zooming frame in - used in offline analysis"
        
        self.zoom /= 1.2
        try:
            self.loadImage()
        except:
            print('Operation failed! Please load valid frames dir!')
            ctypes.windll.user32.MessageBoxW(0,'Operation failed! Please load valid frames dir!',"Message",0)
    
    #%% define function
    def runMedia(self):
        "run media and analyze it"
        
        #%% imports and defines
        
        # imports
        from io import StringIO
        from csv import writer
        
        # cleaning
        self.clearAll()
        self.destroyAll()        
        self.plotCurrEyePanel = Canvas(self.frame, background="white", width=360, height=550)
        self.plotCurrEyePanel.grid(row=3, column=2, rowspan=1, columnspan=4, sticky=W + N)
        
        # Define constants
        PRED_PATH = '../Utils/Predictors/shape_predictor_68_face_landmarks.dat'
        PREC_DIGITS = 2
        
        # Define color list
        C_RED     = (0,0,255)
        C_CYAN    = (255,255,0)
        C_MAGENTA = (255,0,255)
        C_SILVER  = (192,192,192)
        C_LIME    = (0,255,0)
        C_PURPLE  = (128,0,128)
        C_YELLOW  = (0,255,255)
        C_WHITE   = (255,255,255)
        C_BLACK   = (0,0,0)
        C_BLUE    = (255,0,0)
        C_ORANGE  = (0,140,255)
        colors = [C_BLUE,C_CYAN,C_PURPLE,C_LIME,C_SILVER,C_YELLOW,C_MAGENTA,C_WHITE,C_BLACK,C_ORANGE,C_RED]
        
        #%% define function
        def update_structs(action, frame_number, window_len_frame, curr_h_score, curr_p_score,
                                   h_scores, p_scores, min_h_score, max_h_score, min_p_score, max_p_score):
            "Update structs after extraction"            
            
            # Append the score of the current frame
            h_scores.append(curr_h_score)
            p_scores.append(curr_p_score)
            
            # Delete the last score, if necessary
            if action == 0 and frame_number > window_len_frame:
                del h_scores[0]
                del p_scores[0]
            
            # Shrink the window of scores and recalculate extremum values, if necessary
            if action < 0:
                h_scores    = h_scores[-window_len_frame:]
                p_scores    = p_scores[-window_len_frame:]
                min_h_score = np.nanmin(h_scores)
                max_h_score = np.nanmax(h_scores)
                min_p_score = np.nanmin(p_scores)
                max_p_score = np.nanmax(p_scores)
            else:        
                if frame_number == window_len_frame:
                    min_h_score = np.nanmin(h_scores)
                    max_h_score = np.nanmax(h_scores)
                    min_p_score = np.nanmin(p_scores)
                    max_p_score = np.nanmax(p_scores)
                    
                elif frame_number > window_len_frame:
                    min_h_score = min(min_h_score, curr_h_score)
                    max_h_score = max(max_h_score, curr_h_score)
                    min_p_score = min(min_p_score, curr_p_score)
                    max_p_score = max(max_p_score, curr_h_score)
                    
                else:
                    min_h_score = 0
                    max_h_score = 0
                    min_p_score = 0
                    max_p_score = 0
                
            return [h_scores, p_scores, min_h_score, max_h_score, min_p_score, max_p_score]
        
        #%% define function
        def eye_aspect_ratio(eye):
            "Calculate eye aspect ratio (EAR)"
            
            # compute the euclidean distances between the two sets of vertical eye landmarks (x,y) coordinates
            A = math.sqrt(sum([(a-b)**2 for a,b in zip(eye[1],eye[5])]))
            B = math.sqrt(sum([(a-b)**2 for a,b in zip(eye[2],eye[4])]))
            # compute the euclidean distance between the horizontal eye landmark (x,y) coordinate
            C = math.sqrt(sum([(a-b)**2 for a,b in zip(eye[0],eye[3])]))
            # compute the eye aspect ratio
            ear = (A + B) / (2.0 * C)
            # return the eye aspect ratio
            return ear
        
        #%% define function
        def calc_r2(eye):
            "calculate R^2 score for P1, P2, P3 and P4"
            
            x = eye[0:4,0]
            y = eye[0:4,1]
            X = x[:, np.newaxis]
            linreg = LinearRegression()
            linreg.fit(X,y)
            y_pred = linreg.predict(X)
            return [X, y_pred, 1 - r2_score(y, y_pred)]
        
        #%% define function
        def calc_ellipse_area(eye):
            "Calculate the matched elipse area"
            
            (xe, ye), (MA, ma), angle = cv2.fitEllipse(eye)
            A = np.pi * MA * ma
            return A
        
        #%% define function
        def poly_area(eye):
            "Calculate the area of a polygon defined by (x,y) coordinates using Shoelace formula"
            
            x = eye[:,0]
            y = eye[:,1]
            
            correction = x[-1] * y[0] - y[-1]* x[0]
            main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
            return 0.5*np.abs(main_area + correction)
        
        #%% define function
        def define_window_size (curr_feature, curr_score, min_score, max_score, last_score):
            last_score_n = normalize_score(last_score, min_score, max_score, PREC_DIGITS)
            curr_score_n = normalize_score(curr_score, min_score, max_score, PREC_DIGITS)  
            "Normalize feature score for the current frame, Decide when to increase or shrink the window size"    
            
            if ((last_score_n < 0.2) and (curr_score_n < 0.2)) or ((last_score_n > 0.8) and (curr_score_n > 0.8)):
                # Shrink the window to the original size
                return -1    
            if (last_score_n < 0.1) or (last_score_n > 0.9):
                # Increase the window size - save the last score
                return 1    
            # If reached there, behave as usual (remove the last score)
            return 0
        
        #%% define function
        def normalize_feature_score(video_palsy_eye, curr_feature, curr_h_score, min_h_feature, max_h_feature,
                                    curr_p_score, min_p_feature, max_p_feature, PREC_DIGITS):        
            "Normalize feature score for current frame based on the latest scores"
            
            # Normalize according to the healthy eye, if there is a palsy eye
            curr_h_score_n = normalize_score(curr_h_score, min_h_feature, max_h_feature, PREC_DIGITS)    
            if video_palsy_eye != 'n':
                curr_p_score_n = normalize_score(curr_p_score, min_h_feature, max_p_feature, PREC_DIGITS)
            else:
                curr_p_score_n = normalize_score(curr_p_score, min_p_feature, max_p_feature, PREC_DIGITS)    
            
            return [curr_h_score_n, curr_p_score_n]
        
        #%% define function
        def normalize_score(curr_score, min_score, max_score, PREC_DIGITS):
            "Normalize score based on min and max values"
            
            normalized_score = (curr_score - min_score) / (max_score - min_score)
            if normalized_score > 1.0:
                normalized_score = 1.0
            elif normalized_score < 0.0:
                normalized_score = 0.0
            
            return round(float(normalized_score) ,PREC_DIGITS)
        
        #%% define function
        def blink_fast_estimator(df_features_row,prev_consec):
            "Blink fast estimator - Detect blinks events based on current features"
            
            # df_features_row[0] -> 'palsy_eye'
            if df_features_row[0] == 1:
                start_index = 0     # add palsy_eye feature to calculation for palsy eye
            else:
                start_index = 1
            
            # calculate average
            est_label_average = (sum(df_features_row[start_index:]))/(len(df_features_row)-start_index)
            if (est_label_average > 0.4):
                est_label = est_label_average + (1-est_label_average)*0.4
            else:
                est_label = est_label_average
                
            return (est_label)
        
        #%% define function
        def blink_slow_estimator(df_blink, clf, blink_consec, zero_cut=2, levels=5, num_frames=6):
            "Blink slow estimator - Estimate blinks quality based on feature buffer"
            
            # === Fill frames of NaN values ===
            df_blink.fillna(1,inplace=True)
            
            # === Cut the first blink_consec frames ===
            df_blink = df_blink.drop(df_blink.index[:blink_consec])
            
            # === Reduce data: save only strides of four samples ===
            zero_start = zero_cut
            zero_end = len(df_blink.index) - zero_cut
            
            if (zero_end - zero_start > 4):
                df_blink_offset = df_blink.drop(df_blink.index[zero_start + 4 : zero_end + 1]).reset_index(drop=True)
            else:
                df_blink_offset = df_blink.reset_index(drop=True)
                
            # === Add features from neighboring frames ===
            features = ['ear','poly']
            new_features = []
            
            for feature in features:
                new_features.append(feature)        
                # Add a column for each selected feature
                for offset in range(num_frames*(-1), num_frames+1):            
                    # Do something only if the offset isn't zero
                    if(offset == 0):
                        continue
                    new_column = feature + '_' + str(offset)
                    new_features.append(new_column)
                    df_blink_offset[new_column] = df_blink_offset[feature].shift(-1 * offset).fillna(1)
                
            new_features.append('palsy')
            
            # === Create X test vector ===
            Xtest = df_blink_offset[new_features]
            
            # === Classify the data using the given classifier ===
            y_pred = clf.predict(Xtest)
            df_blink_offset.loc[:,'prediction'] = y_pred / (levels - 1)
            
            # === Find the minimum predicted label and consider it as 'blink score' ===
            min_pred  = min(df_blink_offset.iloc[1:-1,:]['prediction'])
            min_pred_idx = df_blink_offset.loc[df_blink_offset['prediction'] == min_pred].index
            min_pred_mid = min_pred_idx[min(len(min_pred_idx) - 1, round(len(min_pred_idx)/2))]
            min_locs = df_blink_offset.loc[min_pred_mid-1:min_pred_mid+1,:]['prediction'].reset_index(drop=True)
            
            if len(min_locs) > 2:
                    if (min_pred == 0.5 and min_locs[0] == 0.75 and min_locs[2] == 0.75):
                        min_pred_final = min_pred
                    elif (min_pred == 0 and min_locs[0] == 0.75 and min_locs[2] == 1):
                        min_pred_final = min_pred
                    elif (min_pred == 0 and min_locs[0] == 0.25 and min_locs[2] == 0.5):
                        min_pred_final = min_pred
                    elif (min_pred == 0.25 and min_locs[0] == 1 and min_locs[2] == 0.75):
                        min_pred_final = min_pred
                    else:
                        min_pred_final = np.median(min_locs)
            
            elif (min_pred == 0.5 and min_locs[0] == 0.5 and df_blink_offset.iloc[-1,:]['prediction'] == 0.5):
                        min_pred_final = 0.75
            else:
                min_pred_final = max(min_locs)
            
            return min_pred_final
        
        #%% define arguments
               
        # define parameters to store args values
        if (self.currentMedia == self.media_list[0]):               # webcam
            source_is_webcam = 1
            if (self.currentPalsyEye == self.palsy_eye_list[1]):    # left
                arg_input = self.currentMedia+'_l'
            elif (self.currentPalsyEye == self.palsy_eye_list[2]):  # right
                arg_input = self.currentMedia+'_r'
            else:                                                   # none
                arg_input = self.currentMedia+'_n'
        else:                                                       # video
            arg_input = self.currentMedia+'.mp4'
            source_is_webcam = 0
        
        # args from gui
        arg_wlen = int(self.WinLenEntry.get())		
        arg_face_frame = int(self.FaceSearchEntry.get())
        arg_webcam_running_time = int(self.WebcamTimeEntry.get())
        arg_eye_blink_th = float(self.BlinkThEntry.get())
        arg_eye_blink_consec_frames = int(self.BlinkConsecEntry.get())        
        
        arg_palsy = self.cbFeaturePalsyVar.get()
        arg_ear = self.cbFeatureEarVar.get()
        arg_r2 = self.cbFeatureR2Var.get()
        arg_ellipse = self.cbFeatureEllipseVar.get()
        arg_poly = self.cbFeaturePolyVar.get()
        arg_estimate_label = self.cbEstimateLabelVar.get()
        
        arg_save_raw_webcam_to_db = self.cbSaveRawWebcamVar.get()
        arg_save_video_features = self.cbSaveVideoFeaturesVar.get()
        arg_save_frame_features = self.cbSaveFrameFeaturesVar.get()
        
        arg_show_score_graphs = self.cbShowScoreGraphVar.get()
        arg_show_score_graphs_feature = self.cbShowFeaturesGraphVar.get()
        arg_show_timing = self.cbSaveTimingVar.get()
        arg_show_face_bbox = self.cbShowFaceBboxVar.get()
        arg_show_features = self.cbShowFeaturesVar.get()
        
        #%% Initialization for time calculations
        timing = {}
        
        timing['initialization'] = []
        timing['grab_frame'] = []
        timing['detect_face'] = []
        timing['save_raw_video'] = []
        timing['faciel_landmarks'] = []
        timing['ear_feature'] = []
        timing['r2_feature'] = []
        timing['ellipse_feature'] = []
        timing['poly_feature'] = []
        timing['palsy_feature'] = []
        timing['estimate_label'] = []
        timing['normalize_feature'] = []
        timing['plot_features'] = []
        timing['blink_detection_full'] = []
        timing['blink_detection_slow'] = []
        timing['save_to_df'] = []
        timing['display_frame'] = []
        timing['frame_process'] = []
        timing['save_after_processing'] = []
        timing['plot_after_processing'] = []
        timing['timing_after_processing'] = []
        timing['save_to_offline_after_processing'] = []
        
        #%% Initialization
        time_start = time.clock()
        
        # cleaning
        self.clearAll()
        self.destroyAll()
        self.listbox = Listbox(self.frame, background="white", width=60, height=34)
        self.listbox.grid(row=3, column=2, rowspan=1, columnspan=4, sticky=W + N)
        
        # video source is webcam
        if (source_is_webcam):
        	# initialize the video stream and allow the camera sensor to warmup
        	print("[INFO] camera sensor warming up...")
        	stream = cv2.VideoCapture(0) # Default camera 
        	time.sleep(1.0)
        	
        	# Get FPS of the camera and update the length of the sliding window
        	print("[INFO] calibrating FPS...")
        	num_frames = 100
        	video_fps_start = time.clock()
        	for i in range(0, num_frames):
        	    ret, frame = stream.read()
        	video_fps_end = time.clock()
        	video_fps  = int(num_frames / (video_fps_end - video_fps_start))
        	print("[INFO] Estimated FPS is: ",video_fps)
        	
        	#video_fps = stream.get(cv2.CAP_PROP_FPS)
        	window_len_frame = math.ceil(video_fps * arg_wlen)
        	webcam_running_len = math.ceil(video_fps * arg_webcam_running_time)
          
        	# Get the location of the palsy eye
        	video_palsy_eye = arg_input[7]
        	palsy_prefix = "webcam_"
        		
        	# Search for the current webcam index
        	video_index = 1
        	while True:
        	    partial_path = "files/" + palsy_prefix + str(video_index).zfill(2)
        	    if len(glob.glob(partial_path + "*")) == 0:
        	        break
        	    video_index = video_index + 1
        	
        	# Set output path
        	output_path = partial_path + '_' + video_palsy_eye + '_' + str(video_fps).zfill(2) + "fps"
        	output_path_raw = partial_path + '_' + video_palsy_eye + '_' + str(video_fps).zfill(2) + "fps"
        
        # video source is file
        else:
        	# Calcultate parameters from arguments
        	video_path = '../../Database/' + arg_input	
        	
        	video_index = arg_input[-14:-12]
        	video_fps = int(arg_input[-9:-7])
        	video_palsy_eye = arg_input[-11]
        	window_len_frame = math.ceil(video_fps * arg_wlen)
        	
        	# Set output path
        	output_path = "files/" + arg_input[:-4]
        	
        	# Start the video stream thread
        	if os.path.exists(video_path):
        	    print("[INFO] starting video stream thread...")
        	    stream = cv2.VideoCapture(video_path)
        	    fileStream = True
        	    time.sleep(1.0)
        	else:
        	    print('Run Media failed! No video found in the specified dir!')
        	    ctypes.windll.user32.MessageBoxW(0,'Run Media failed! No video found in the specified dir!',"Message",0)
        	    return
        	
        # Create subfolders for outputs
        media_name = output_path[6:]
        output_path_full = output_path + "/" + output_path[6:]
        output_path_frames_features = output_path + "/frames_features"
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print("[INFO] Successfully created subfolders for output files: ", output_path)
        
        if arg_save_frame_features:
            frame_feature_list = []
            frame_feature_path_list = []
            if not os.path.exists(output_path_frames_features):
                os.makedirs(output_path_frames_features)
                print("[INFO] Successfully created subfolders for output files: ", output_path_frames_features)
        
        # Initialize structs
        df_column = []          # list of active features and label
        h_scores = {}           # Dictionary of lists, each represents a sliding window of a specific feature
        p_scores = {}           # Dictionary of lists, each represents a sliding window of a specific feature
        min_h_score = {}        # Dictionary of extremum values, each represents a specific feature
        max_h_score = {}        # Dictionary of extremum values, each represents a specific feature
        min_p_score = {}        # Dictionary of extremum values, each represents a specific feature
        max_p_score = {}        # Dictionary of extremum values, each represents a specific feature
         
        # Initialize program's structs
        feature_list = ['palsy','ear','r2','ellipse','poly']
        feature_dict = {'palsy':arg_palsy, 'ear':arg_ear, 'r2':arg_r2, 'ellipse':arg_ellipse, 'poly':arg_poly}
        for feature in feature_list:
            if feature_dict[feature]:
                df_column.append(feature)
                h_scores[feature]     = []
                p_scores[feature]     = []
                min_h_score[feature]  = 0
                max_h_score[feature]  = 0
                min_p_score[feature]  = 0
                max_p_score[feature]  = 0
        
        # estimate label should be the last object in df_column
        if arg_estimate_label:
            df_column.append('estimate_label')
                    
        # Define text position lists
        text_pos_r = []
        text_pos_m = []
        text_pos_l = []
        text_pos_gap = 15
        
        # Define text locations
        for i in range(0,10):
            text_pos_r.append((10,30+i*text_pos_gap))
            text_pos_m.append((250,15+i*text_pos_gap))
            text_pos_l.append((700,30+i*text_pos_gap))
        
        # Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(PRED_PATH)
        
        # Grab the indexes of the facial landmarks for the healthy and palsy eye, respectively
        if video_palsy_eye == 'l':
            (pStart, pEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (hStart, hEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]            
        else:
            (pStart, pEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            (hStart, hEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            
        # Set color to mark eye
        c_h_plot = C_RED
        c_p_plot = C_RED
        
        # initialize values
        prev_h_estimate_labels = [float('NaN')]
        prev_p_estimate_labels = [float('NaN')]
        
        rects = []
        action = 0
        frame_number = 0
        blink_counter = 0
        blink_details = [] # [counter, start_frame, stop_frame, h_label, p_label]
        blink_counter_temp = 0
        blink_region_frames_temp = []
        blink_h_values_temp = []
        blink_p_values_temp = []
        blink_status = 0
        dataframe_row_pointer_h = []
        dataframe_row_pointer_p = []
        post_blink_counter = 0
        post_blink_counter_buffering = 0
        blink_estimation_active = 0
        
        # string structures for dataframe
        output_h = StringIO()
        output_p = StringIO()
        csv_writer_h = writer(output_h)
        csv_writer_p = writer(output_p)
        
        # Load the slow classifier file
        clf = load('../4_training_classifiers/model.joblib')
        
        fps = FPS().start()
        
        time_end = time.clock()
        timing['initialization'].append(time_end - time_start)
        
        #%% Loop over frames from the video stream
        while True:
            # measure loop timing
            frame_time_start = time.clock()
            
            #%% grab frame from video stream
            time_start = time.clock()
        
            # Grab the frame from the threaded video file stream
            (grabbed, frame) = stream.read()
            if not grabbed:
                break
            
            # Resize the frame
            frame = imutils.resize(frame, width=900)
            if (frame_number == 0):
                # Get frame dimentions
                height, width = frame.shape[:2]
            
            time_end = time.clock()
            timing['grab_frame'].append(time_end - time_start)
                
            #%% Detect faces in the grayscale frame
            time_start = time.clock()
            
            if frame_number % arg_face_frame == 0 or len(rects) != 1:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame_number == 0 or len(rects) != 1:
                    # take entire frame
                    bbox_left = 0
                    bbox_top = 0
                    bbox_right = 0
                    bbox_bottom = 0
                    gray_crop = gray
                else:
                    # crop frame
                    bbox_left = prev_rect.left() - int((prev_rect.left())*0.2)
                    bbox_top = prev_rect.top() - int((prev_rect.top())*0.2)
                    bbox_right = prev_rect.right() + int((width - prev_rect.right())*0.2)
                    bbox_bottom = prev_rect.bottom() + int((height - prev_rect.bottom())*0.2)
                    gray_crop = gray[bbox_top:bbox_bottom, bbox_left:bbox_right]
                
                rects = detector(gray_crop, 0)
                
                # calculate rect relative to the entire frame
                if len(rects) == 1:
                    prev_rect_left = rects[0].left()+bbox_left
                    prev_rect_right = rects[0].right()+bbox_left
                    prev_rect_top = rects[0].top()+bbox_top
                    prev_rect_bottom = rects[0].bottom()+bbox_top
                    prev_rect = dlib.rectangle(left=prev_rect_left,top=prev_rect_top,right=prev_rect_right,bottom=prev_rect_bottom)
            
            time_end = time.clock()
            timing['detect_face'].append(time_end - time_start)
            
            #%% save raw video if reauired, add frame titles
            time_start = time.clock()
            
            if (source_is_webcam and arg_save_raw_webcam_to_db):
                if (frame_number == 0):
                    # Get frame dimentions
                    height, width = frame.shape[:2]
                    # Define the codec and create VideoWriter object
                    video_out_raw = cv2.VideoWriter(output_path_raw+".mp4",cv2.VideoWriter_fourcc('m','p','4','v'), video_fps, (width,height))		
        
                # Save resulting frame to output video
                video_out_raw.write(frame)
            
            # frame name
            frame_name = output_path[6:]+'_frame'+"{:04d}".format(frame_number)
            
            # show frame name and elapsed time
            elapsed_time = frame_number / video_fps
            cv2.putText(frame, frame_name+", T = {:.2f} Sec".format(elapsed_time), text_pos_m[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[-1], 2)
            
            time_end = time.clock()
            timing['save_raw_video'].append(time_end - time_start)
            
            #%% operate according to number of faces in frame
            if (len(rects) != 1):
                # notice about faces number on current frame
                cv2.putText(frame, "{:d} faces detected!".format(len(rects)), text_pos_m[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[-1], 2)
                
                # prepare NaN raw for current frame
                df_nan_row = [float('NaN')]*(len(df_column))
                
                # write a row of NaNs
                csv_writer_h.writerow(df_nan_row)
                csv_writer_p.writerow(df_nan_row)
                
            else:
                #%% draw face bounding box and landmarks and extract eye coordinates
                time_start = time.clock()
                
                # draw face bounding box if required
                if(arg_show_face_bbox):
                    (x, y, w, h) = face_utils.rect_to_bb(prev_rect)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # determine the facial landmarks for the face region, then
                # Convert the facial landmark (x,y) coordinates to a NumPy array
                shape = predictor(frame, prev_rect)
                shape = face_utils.shape_to_np(shape)
                
                # Extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                h_eye_shape = shape[hStart:hEnd]
                p_eye_shape = shape[pStart:pEnd]
                
                # Initialize rows for DataFrames
                df_h_row_n = []
                df_p_row_n = []
                
                # Feature index
                feature_index = 0                
                
                time_end = time.clock()
                timing['faciel_landmarks'].append(time_end - time_start)
                
                #%% Calculate relevant features
                for feature in df_column:
                    
                    if feature   == 'ear':
                        time_start = time.clock()
                        
                        curr_h_score = eye_aspect_ratio(h_eye_shape)
                        curr_p_score = eye_aspect_ratio(p_eye_shape)
                        
                        time_end = time.clock()
                        timing['ear_feature'].append(time_end - time_start)
                        
                    elif feature == 'r2':
                        time_start = time.clock()
                        
                        [x_h, y_pred_h, curr_h_score] = calc_r2(h_eye_shape)
                        [x_p, y_pred_p, curr_p_score] = calc_r2(p_eye_shape)
                        
                        time_end = time.clock()
                        timing['feature'].append(time_end - time_start)
                        
                    elif feature == 'ellipse':
                        time_start = time.clock()
                        
                        curr_h_score = calc_ellipse_area(h_eye_shape)
                        curr_p_score = calc_ellipse_area(p_eye_shape)
                        
                        time_end = time.clock()
                        timing['ellipse_feature'].append(time_end - time_start)
                        
                    elif feature == 'poly':
                        time_start = time.clock()
                        
                        curr_h_score = poly_area(h_eye_shape)
                        curr_p_score = poly_area(p_eye_shape)
                        
                        time_end = time.clock()
                        timing['poly_feature'].append(time_end - time_start)
                        
                    elif feature == 'palsy':
                        time_start = time.clock()
                        
                        # if feature is palsy, update it directly
                        curr_h_score_n = 0
                        if video_palsy_eye == 'n':
                            curr_p_score_n = 0
                        else:
                            curr_p_score_n = 1
                        
                        time_end = time.clock()
                        timing['palsy_feature'].append(time_end - time_start)
                        
                    elif feature == 'estimate_label':
                        time_start = time.clock()
                        
                        # take mean of last consc_frames
                        try:
                            prev_h_consc = mean(prev_h_estimate_labels[-arg_eye_blink_consec_frames:])
                            prev_p_consc = mean(prev_p_estimate_labels[-arg_eye_blink_consec_frames:])
                        except:
                            prev_h_consc = float('NaN')
                            prev_p_consc = float('NaN')
                        
                        # if object is estimate label, update it directly
                        curr_h_score_n = blink_fast_estimator(df_h_row_n,prev_h_consc)
                        curr_p_score_n = blink_fast_estimator(df_p_row_n,prev_p_consc)
                        
                        # save for estimation for next frame
                        prev_h_estimate_labels.append(curr_h_score_n)
                        prev_p_estimate_labels.append(curr_p_score_n)
                        
                        time_end = time.clock()
                        timing['estimate_label'].append(time_end - time_start)
                    
                    #%% Calculate normalized values
                    time_start = time.clock()
                    
                    if (feature != 'palsy' and feature !='estimate_label'):
                        if (frame_number <= window_len_frame):
                            curr_h_score_n = float('NaN')
                            curr_p_score_n = float('NaN')
                        else:
                            [curr_h_score_n, curr_p_score_n] = normalize_feature_score(video_palsy_eye, df_column[feature_index],
                                                                                       curr_h_score,min_h_score[feature],max_h_score[feature],
                                                                                       curr_p_score,min_p_score[feature],max_p_score[feature], 
                                                                                       PREC_DIGITS)
                            # Determine the window size for normalization
                            action = define_window_size(df_column[feature_index], curr_h_score, min_h_score[feature], max_h_score[feature], h_scores[feature][0])
                        
                        # Updates windows and extremum values
                        [h_scores[feature], p_scores[feature], min_h_score[feature], max_h_score[feature], min_p_score[feature], max_p_score[feature]] = update_structs(
                                action,frame_number, window_len_frame, curr_h_score, curr_p_score, h_scores[feature], p_scores[feature], 
                                min_h_score[feature], max_h_score[feature],min_p_score[feature], max_p_score[feature])
        
                    # Append to row for normalized data frame
                    df_h_row_n.append(curr_h_score_n)
                    df_p_row_n.append(curr_p_score_n)
                    
                    time_end = time.clock()
                    timing['normalize_feature'].append(time_end - time_start)
                    
                    #%% Visualize scores and features
                    time_start = time.clock()
                    
                    # Visualize scores
                    if video_palsy_eye == 'l':
                        curr_right_score_n = curr_h_score_n
                        curr_left_score_n  = curr_p_score_n
                    else:
                        curr_right_score_n = curr_p_score_n
                        curr_left_score_n  = curr_h_score_n
                            
                    cv2.putText(frame, "R_"+feature+": {:.2f}".format(curr_right_score_n), 
                                        text_pos_r[feature_index], cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[feature_index], 2)
                    cv2.putText(frame, "L_"+feature+": {:.2f}".format(curr_left_score_n), 
                                        text_pos_l[feature_index], cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[feature_index], 2)
                    
                    # Mark eyes                
                    for (x,y) in h_eye_shape:
                        cv2.circle(frame, (x,y), 1, c_h_plot, -1)
                    for (x,y) in p_eye_shape:
                        cv2.circle(frame, (x,y), 1, c_p_plot, -1)
                    
                    # Visualize features (if necessary)
                    if (arg_show_features):
                        if feature == 'r2':
                            cv2.line(frame,(int(x_h[0]),int(y_pred_h[0])),(int(x_h[-1]),int(y_pred_h[-1])),colors[feature_index],1)
                            cv2.line(frame,(int(x_p[0]),int(y_pred_p[0])),(int(x_p[-1]),int(y_pred_p[-1])),colors[feature_index],1)
                        elif feature == 'ellipse':
                            cv2.ellipse(frame,cv2.fitEllipse(h_eye_shape), colors[feature_index])
                            cv2.ellipse(frame,cv2.fitEllipse(p_eye_shape), colors[feature_index])
                        elif feature == 'poly':
                            cv2.drawContours(frame,[cv2.convexHull(h_eye_shape)],-1,colors[feature_index],1)
                            cv2.drawContours(frame,[cv2.convexHull(p_eye_shape)],-1,colors[feature_index],1)
                    
                    feature_index = feature_index + 1
                    
                    time_end = time.clock()
                    timing['plot_features'].append(time_end - time_start)
            
            #%% blink detection (fast and slow estimator)
            time_start = time.clock()
            
            # Blink detection (palsy: of healty eye, healty: of left eye)
            if (frame_number <= window_len_frame) or (len(rects) != 1):
                blink_indication_color = C_SILVER
            # blink status if not active
            elif blink_status == 0:
                # if the estimate label is below the blink label threshold, increment the blink frame counter
                if prev_h_estimate_labels[-1] <= arg_eye_blink_th:
                    blink_counter_temp += 1
                    blink_region_frames_temp.append(frame_number)
                    blink_h_values_temp.append(prev_h_estimate_labels[-1])
                    blink_p_values_temp.append(prev_p_estimate_labels[-1])
                    if blink_counter_temp == 1:
                        # hold dataframe with some frames before blink started
                        blink_buffer = int(self.BlinkBufferEntry.get())
                        output_h_temp = dataframe_row_pointer_h[-blink_buffer]
                        output_p_temp = dataframe_row_pointer_p[-blink_buffer]
                    # if the frame counter is above the blink consec frame threshold, blink status become active
                    if blink_counter_temp >= arg_eye_blink_consec_frames:
                        blink_indication_color = C_LIME
                        blink_counter += 1
                        blink_status = 1
                        blink_counter_temp = 0
                    else:
                        blink_indication_color = C_YELLOW
                # otherwise (the estimate label is above the blink label threshold), initialize blink frame counter
                else:
                    # reset the eye frame counter
                    blink_indication_color = C_RED
                    blink_counter_temp = 0
                    blink_region_frames_temp = []
                    blink_h_values_temp = []
                    blink_p_values_temp = []                    
            # otherwise (blink status is active)
            else:               
                # if the estimate label is above the blink label threshold, increment the blink frame counter
                if prev_h_estimate_labels[-1] > arg_eye_blink_th:
                    blink_counter_temp += 1
                    blink_indication_color = C_YELLOW
                    # if the frame counter is above the blink consec frame threshold, blink status become idle and blink estimator
                    if blink_counter_temp > arg_eye_blink_consec_frames:                      
                        # prepare for post blink rows buffering
                        # if next blink end is reached while buffering, it will be ignored
                        if post_blink_counter_buffering == 0:
                            post_blink_counter = blink_buffer - arg_eye_blink_consec_frames
                            output_h_temp_buffering = output_h_temp
                            output_p_temp_buffering = output_p_temp
                            blink_counter_buffering = blink_counter
                            blink_region_frames_temp_buffering_start = blink_region_frames_temp[0]
                            blink_region_frames_temp_buffering_end = blink_region_frames_temp[-1]
                            if post_blink_counter > 0:
                                post_blink_counter_buffering = post_blink_counter + 1   # send with buffering
                            else:
                                post_blink_counter_buffering = 1                        # send without buffering

                        # reset the eye frame counter
                        blink_indication_color = C_RED
                        blink_status = 0
                        blink_counter_temp = 0
                        blink_region_frames_temp = []
                        blink_h_values_temp = []
                        blink_p_values_temp = []
                    else:
                        blink_region_frames_temp.append(frame_number)
                        blink_h_values_temp.append(prev_h_estimate_labels[-1])
                        blink_p_values_temp.append(prev_p_estimate_labels[-1]) 
                else:
                    blink_indication_color = C_LIME
                    blink_region_frames_temp.append(frame_number)
                    blink_h_values_temp.append(prev_h_estimate_labels[-1])
                    blink_p_values_temp.append(prev_p_estimate_labels[-1]) 

            # concern post blink buffering
            if post_blink_counter_buffering > 0:
                post_blink_counter_buffering = post_blink_counter_buffering - 1
                blink_estimation_active = 0
                if post_blink_counter_buffering == 0:
                    # pointer for send relevant dataframe to estimator function
                    output_h.seek(output_h_temp_buffering)
                    output_p.seek(output_p_temp_buffering)
                    # estimate blink quality
                    time_start_slow_estimator = time.clock()
                    
                    pd.read_csv(output_h, names=df_column)
                    blink_h_value = 0
                    if video_palsy_eye == 'n':
                        pd.read_csv(output_p, names=df_column)
                        blink_p_value = 0
                    else:
                        blink_p_value = blink_slow_estimator(pd.read_csv(output_p, names=df_column), clf, arg_eye_blink_consec_frames)
                    trigger = blink_p_value - blink_h_value
                    
                    time_end_slow_estimator = time.clock()
                    timing['blink_detection_slow'].append(time_end_slow_estimator - time_start_slow_estimator)
                    
                    blink_details.append([blink_counter_buffering,blink_region_frames_temp_buffering_start,blink_region_frames_temp_buffering_end,
                                            blink_h_value, blink_p_value])
                    # update log
                    if video_palsy_eye == 'l': 
                        self.listbox.insert(END, 'Blink {}, Frame [{},{}]: r = {:.2f}, l = {:.2f}, Trig = +{:.2f}'
                                            .format(blink_counter_buffering,blink_region_frames_temp_buffering_start,blink_region_frames_temp_buffering_end,
                                            blink_h_value, blink_p_value, trigger))
                    else:
                        self.listbox.insert(END, 'Blink {}, Frame [{},{}]: r = {:.2f}, l = {:.2f}, Trig = +{:.2f}'
                                            .format(blink_counter_buffering,blink_region_frames_temp_buffering_start,blink_region_frames_temp_buffering_end,
                                            blink_p_value, blink_h_value, trigger))
                    blink_estimation_active = 1
                else:
                    # if next blink end is reached while buffering, it will be ignored
                    blink_indication_color = C_MAGENTA
            
            # update blink details
            if video_palsy_eye == 'l':                
                # update blink indication
                cv2.circle(frame, (250,45), 15, blink_indication_color, -1)
                cv2.putText(frame, "{}".format(blink_counter), (248, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_BLACK, 2)
                # update recent blinks label
                if (blink_estimation_active) and (blink_status == 0):
                    cv2.putText(frame, "({:.2f},{:.2f})".format(blink_h_value,blink_p_value), (205, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blink_indication_color, 2)
                    cv2.putText(frame, "(Trig +{:.2f})".format(trigger), (205, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blink_indication_color, 2)
            else:
                # update blink indication
                cv2.circle(frame, (640,45), 15, blink_indication_color, -1)
                cv2.putText(frame, "{}".format(blink_counter), (638, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_BLACK, 2)
                # update recent blinks label
                if (blink_estimation_active) and (blink_status == 0):
                    cv2.putText(frame, "({:.2f},{:.2f})".format(blink_p_value,blink_h_value), (595, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blink_indication_color, 2)                
                    cv2.putText(frame, "(Trig +{:.2f})".format(trigger), (595, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blink_indication_color, 2)
                    
            time_end = time.clock()
            timing['blink_detection_full'].append(time_end - time_start)
                    
            #%% Concatenate row to exist DataFrame
            time_start = time.clock()
            
            if (len(rects) == 1):
                dataframe_row_pointer_h.append(output_h.tell())
                dataframe_row_pointer_p.append(output_p.tell())
                csv_writer_h.writerow(df_h_row_n)
                csv_writer_p.writerow(df_p_row_n)
                   
            time_end = time.clock()
            timing['save_to_df'].append(time_end - time_start)
            
            #%% display and save the resulting frame
            time_start = time.clock()

            # display resulting frame
            if self.currentPerformance == self.performance_list[0]: #'Offline'
                frame_rgb = frame[...,[2,1,0]]
                self.photo = ImageTk.PhotoImage(image = Image.fromarray(frame_rgb))
                self.mainPanel.create_image(0, 0, image = self.photo, anchor = NW)
                self.mainPanel.update_idletasks()
            elif self.currentPerformance == self.performance_list[1]: #'Real-Time'
                self.mainPanel.update_idletasks()
                cv2.imshow('Frame', frame)
			
            # Save resulting frame to images
            if arg_save_frame_features:
                frame_feature_list.append(frame)
                frame_feature_path_list.append(output_path_frames_features+"/"+frame_name+".jpg")
            
            # Save resulting frame to video
            if arg_save_video_features:
                if (frame_number == 0):
                    # Get frame dimentions
                    height, width = frame.shape[:2]
                    # Define the codec and create VideoWriter object
                    video_out = cv2.VideoWriter(output_path_full+"_out.mp4",cv2.VideoWriter_fourcc('m','p','4','v'), video_fps, (width,height))
                
                # Save resulting frame to output video
                video_out.write(frame)
                
            # Press ESC on keyboard to break
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
                        
            # to break video from webcam after selected time
            if (source_is_webcam):
                if (frame_number >= webcam_running_len):
                    break
            
            # Promote frame
            frame_number = frame_number + 1
            fps.update()
            
            time_end = time.clock()
            timing['display_frame'].append(time_end - time_start)
            timing['frame_process'].append(time_end - frame_time_start)
        
        #%% save after finish processing
        time_start = time.clock()
        
        # read to dataframe
        output_h.seek(0) # we need to get back to the start of the BytesIO
        df_h_scores_n = pd.read_csv(output_h, names=df_column)
        output_p.seek(0) # we need to get back to the start of the BytesIO
        df_p_scores_n = pd.read_csv(output_p, names=df_column)
        
        # Save normalized data frame as excel files
        writer = pd.ExcelWriter(output_path_full+"_scores_n.xlsx")
        if video_palsy_eye == 'l':
            df_p_scores_n.to_excel(writer,sheet_name='left_eye')
            df_h_scores_n.to_excel(writer,sheet_name='right_eye')    
        else:
            df_h_scores_n.to_excel(writer,sheet_name='left_eye')
            df_p_scores_n.to_excel(writer,sheet_name='right_eye')
        
        writer.save()
        writer.close()
        
        # Do a bit of cleanup
        if (source_is_webcam and arg_save_raw_webcam_to_db):
            video_out_raw.release()
        if arg_save_video_features:
            video_out.release()
        
        # Save log to text file
        self.logFilePath = output_path_full+"_log.txt"
        logFile = open(self.logFilePath,'w')
        logFile.write('\n'.join(self.listbox.get(0,END)))
        logFile.close()
        
        # Save log blinks to text file
        self.logBlinksFilePath = output_path_full+"_log_blinks.txt"
        logFileBlinks = open(self.logBlinksFilePath,'w')
        for line in blink_details:    
            logFileBlinks.write("%s\n" % line)
        logFileBlinks.close()
        
        # vs.stop()
        stream.release()
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        cv2.destroyAllWindows()
        
        time_end = time.clock()
        timing['save_after_processing'].append(time_end - time_start)
        
        #%% show graphs
        time_start = time.clock()
        
        if arg_show_score_graphs:
            print("[INFO] Plot score graphs")
            
            # load data frames from excel files
            df_scores_n_xlsx = pd.ExcelFile(output_path_full+"_scores_n.xlsx")
            df_scores_l_n = pd.read_excel(df_scores_n_xlsx,'left_eye')
            df_scores_r_n = pd.read_excel(df_scores_n_xlsx,'right_eye')
            
            # Create column fo frame number and elapsed time
            df_frame_number = pd.Series(df_scores_l_n.index)
            df_frame_elapsed_time = df_frame_number.apply(lambda x: round(float(x/video_fps),PREC_DIGITS))
            
            # Append it to data frame
            df_scores_l_n['frame_number'] = df_frame_number
            df_scores_l_n['elapsed_time'] = df_frame_elapsed_time
            df_scores_r_n['frame_number'] = df_frame_number
            df_scores_r_n['elapsed_time'] = df_frame_elapsed_time
            
            plt.figure(figsize=(10,5))
            
            # plot graph for left eye
            ax_left = plt.subplot(2,1,1)
            # plot features
            feature_index = 0
            
            if arg_show_score_graphs_feature:
                for feature in df_column:
                    df_scores_l_n_ax = df_scores_l_n.plot(kind='line',x='frame_number',y=feature,color=tuple(reversed(np.divide(colors[feature_index],255))),ax=ax_left)
                    feature_index = feature_index + 1
            else:
                df_scores_l_n_ax = df_scores_l_n.plot(kind='line',x='frame_number',y='estimate_label',color=tuple(reversed(np.divide(colors[feature_index],255))),ax=ax_left)
                feature_index = feature_index + 1
            
            # edit titles and axis
            plt.title(media_name+': features_scores_n vs frame_number & elapsed_time')
            plt.xlabel("")
            plt.ylabel('scores_left_n')
            ax_left.set_xlim([df_frame_number.iloc[0],df_frame_number.iloc[-1]])
            ax_left.set_ylim([0,1])
            
            # plot graph for right eye
            ax_right = plt.subplot(2,1,2)
            # plot features
            feature_index = 0
            
            if arg_show_score_graphs_feature:
                for feature in df_column:
                    df_scores_r_n_ax = df_scores_r_n.plot(kind='line',x='frame_number',y=feature,color=tuple(reversed(np.divide(colors[feature_index],255))),ax=ax_right)
                    feature_index = feature_index + 1
            else:
                df_scores_r_n_ax = df_scores_r_n.plot(kind='line',x='frame_number',y='estimate_label',color=tuple(reversed(np.divide(colors[feature_index],255))),ax=ax_right)
                feature_index = feature_index + 1
            
            # plot blinks bounds
            if video_palsy_eye == 'l':
                for i in range(len(blink_details)):
                    df_scores_l_n_ax.axvspan(blink_details[i][1],blink_details[i][2], facecolor='y')
                    df_scores_l_n_ax.axvspan(blink_details[i][1],blink_details[i][2],blink_details[i][3],blink_details[i][4], facecolor='r')
                    df_scores_r_n_ax.axvspan(blink_details[i][1],blink_details[i][2], facecolor=tuple(np.divide(c_lime,255)))
                    
            elif video_palsy_eye == 'r':
                for i in range(len(blink_details)):
                    df_scores_l_n_ax.axvspan(blink_details[i][1],blink_details[i][2], facecolor=tuple(np.divide(c_lime,255)))
                    df_scores_r_n_ax.axvspan(blink_details[i][1],blink_details[i][2], facecolor='y')
                    df_scores_r_n_ax.axvspan(blink_details[i][1],blink_details[i][2],blink_details[i][3],blink_details[i][4], facecolor='r')
            else:
                for i in range(len(blink_details)):
                    df_scores_l_n_ax.axvspan(blink_details[i][1],blink_details[i][2], facecolor=tuple(np.divide(c_lime,255)))
                    df_scores_r_n_ax.axvspan(blink_details[i][1],blink_details[i][2], facecolor=tuple(np.divide(c_lime,255)))
            
            # edit titles and axis
            plt.ylabel('scores_right_n')
            ax_right.set_xlim([df_frame_number.iloc[0],df_frame_number.iloc[-1]])
            ax_right.set_ylim([0,1])
            
            # set second x-axis (elapsed_time)
            ax2 = ax_right.twiny()
            # set the ticklabel position in the second x-axis, then convert them to the position in the first x-axis
            ax2_ticks_num = 6
            newlabel = [round((x*df_frame_elapsed_time.iloc[-1]/(ax2_ticks_num-1)),PREC_DIGITS) for x in range(0, ax2_ticks_num)]
            newpos = [int(np.ceil(x*video_fps)) for x in newlabel]
            # set the second x-axis
            ax2.set_xticks(newpos)
            ax2.set_xticklabels(newlabel)
            ax2.xaxis.set_ticks_position('bottom') 
            ax2.xaxis.set_label_position('bottom')
            ax2.spines['bottom'].set_position(('outward', 36))
            ax2.set_xlabel('elapsed_time [Sec]')
            
            # save plot to file
            plt.savefig(output_path_full+"_scores_graphs.png",bbox_inches='tight',dpi=300)
            # show plot
            self.img = Image.open(output_path_full+"_scores_graphs.png")            
            resize_ratio = 900/self.img.size[0]
            self.img = self.img.resize([int(resize_ratio * s) for s in self.img.size])
            self.tkImg = ImageTk.PhotoImage(self.img)
            self.mainPanel.create_image(0, 0, image=self.tkImg, anchor=NW)
            ##plt.show()
        
        time_end = time.clock()
        timing['plot_after_processing'].append(time_end - time_start)
            
        #%% show timings
        time_start = time.clock()
        
        if arg_show_timing:    
            # Show timing histograms
            print("[INFO] Plot timing graphs")
            f1 = plt.figure('Histograms',figsize=(18,12),dpi=70)
            
            plt.subplot(3,4,1)
            plt.hist(list(map(lambda x:1000*x,timing['grab_frame'])), color = 'red')
            plt.ylabel('grab_frame')
            plt.xlabel('Processing time [ms]')
            
            plt.subplot(3,4,2)
            plt.hist(list(map(lambda x:1000*x,timing['detect_face'])), color='orange')
            plt.ylabel('detect_face')
            plt.xlabel('Processing time [ms]')
            
            plt.subplot(3,4,3)
            plt.hist(list(map(lambda x:1000*x,timing['faciel_landmarks'])), color='cyan')
            plt.ylabel('faciel_landmarks')
            plt.xlabel('Processing time [ms]')
            
            plt.subplot(3,4,4)
            plt.hist(list(map(lambda x:1000*x,timing['ear_feature'])), color='purple')
            plt.ylabel('ear_feature')
            plt.xlabel('Processing time [ms]')
            
            plt.subplot(3,4,5)
            plt.hist(list(map(lambda x:1000*x,timing['poly_feature'])), color='magenta')
            plt.ylabel('poly_feature')
            plt.xlabel('Processing time [ms]')
            
            plt.subplot(3,4,6)
            plt.hist(list(map(lambda x:1000*x,timing['estimate_label'])), color='green')
            plt.ylabel('estimate_label')
            plt.xlabel('Processing time [ms]')
            
            plt.subplot(3,4,7)
            plt.hist(list(map(lambda x:1000*x,timing['normalize_feature'])), color='gray')
            plt.ylabel('normalize_feature')
            plt.xlabel('Processing time [ms]')
            
            plt.subplot(3,4,8)
            plt.hist(list(map(lambda x:1000*x,timing['plot_features'])), color='brown')
            plt.ylabel('plot_features')
            plt.xlabel('Processing time [ms]')
              
            plt.subplot(3,4,9)
            plt.hist(list(map(lambda x:1000*x,timing['blink_detection_full'])), color='yellow')
            plt.ylabel('blink_detection_full')
            plt.xlabel('Processing time [ms]')
            
            plt.subplot(3,4,10)
            plt.hist(list(map(lambda x:1000*x,timing['save_to_df'])), color='blue')
            plt.ylabel('save_to_df')
            plt.xlabel('Processing time [ms]')
            
            plt.subplot(3,4,11)
            plt.hist(list(map(lambda x:1000*x,timing['display_frame'])), color='pink')
            plt.ylabel('display_frame')
            plt.xlabel('Processing time [ms]')
            
            plt.subplot(3,4,12)
            plt.hist(list(map(lambda x:1000*x,timing['frame_process'])), color='black')
            plt.ylabel('single frame_process')
            plt.xlabel('Processing time [ms]')
            
            # save plot to file
            plt.savefig(output_path_full+"_timing_histograms.png",bbox_inches='tight',dpi=300)
            # show plot
            plt.show()
        
        time_end = time.clock()
        timing['timing_after_processing'].append(time_end - time_start)
        
        #%% save frames features
        time_start = time.clock()
        
        if arg_save_frame_features:
            print("[INFO] Save frame features to offline")
            # Save resulting frame to frames_features
            for i in range(0,len(frame_feature_list)):
                cv2.imwrite(frame_feature_path_list[i], frame_feature_list[i])
        
        print("[INFO] Done! :-)")
                        
        if (source_is_webcam):
            # webcam become video
            source_is_webcam = 0
            # update media name
            self.media_list.append(media_name)
            self.mediaName.set(media_name)
            self.mediaList.destroy()
            self.mediaList = OptionMenu(self.upperPanel, self.mediaName, *self.media_list, command=self.setMedia)
            self.currentMedia = self.mediaName.get()
            self.mediaList.pack(side=LEFT, padx=5, pady=3)
            self.mediaList.config(width=25)                        
            print('set current media to :', self.currentMedia)            
            # update palsy eye 
            self.palsyEyeList.config(state='disabled')
            if (self.currentMedia[-7] == 'l'):
                self.palsyEye.set(self.palsy_eye_list[1])
            elif (self.currentMedia[-7] == 'r'):
                self.palsyEye.set(self.palsy_eye_list[2])
            else:
                self.palsyEye.set(self.palsy_eye_list[0])
            self.currentPalsyEye = self.palsyEye.get()
            print('set current palsy eye to :', self.currentPalsyEye)
            
        time_end = time.clock()
        timing['save_to_offline_after_processing'].append(time_end - time_start)

        #%% print running times
        for key in timing:
            if len(timing[key]) > 0:
                print ('Mean time of {} = {:.2f}ms'.format(key, 1000*mean(timing[key])))
            else:
                print ('Mean time of {} = {:.2f}ms'.format(key, 0))
                
#%% main
if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    root.resizable(width=True, height=True)
    root.state('zoomed')
    root.mainloop()
