

# imports
from tkinter import *
from PIL import Image, ImageDraw, ImageFont, ImageTk
import pandas as pd
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
import os
import glob
import random
import matplotlib
import ctypes
import matplotlib.pyplot as plt
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg

# define tagging labels
number_of_levels = 5
eye_open_levels = [i/(number_of_levels-1) for i in range(0,number_of_levels)]
eye_open_levels.insert(0,np.nan)

# define color list
c_red = (255,0,0)
c_lime = (0,255,0)
c_blue = (0,0,255)
c_magenta = (255,0,255)

# define text position lists
text_pos_r = []
text_pos_l = []
text_pos_gap = 40

line_styles = ['-', '--', '-.', ':']

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
        self.eyes_list = ['left','right']
        self.users_list = ['oren','tom']
        self.dir_is_active = 0
        self.currentLabelIndex = 0
        self.precision_digits = 2
        
        for i in range(0,len(self.users_list)):
            text_pos_r.append((10,50+i*text_pos_gap))
            text_pos_l.append((700,50+i*text_pos_gap))
        
        # prep for zoom
        self.zoom = 1
        # ----------------- GUI stuff ---------------------
        
        # upper panel (dir, eye, user, load)
        self.upperPanel = Frame(self.frame)
        self.upperPanel.grid(row=0, column=1, sticky=W + E)
        
        self.filesDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'files')
        
        self.label = Label(self.frame, text="Dir Path:", width=7)
        self.label.grid(row=0, column=0, sticky=W + E)
        self.dirEntry = Entry(self.upperPanel, width = 100)
        self.dirEntry.insert(END, self.filesDir)
        self.dirEntry.config(state='disabled')
        self.dirEntry.pack(side=LEFT, padx=5, pady=3)
        
        self.label = Label(self.upperPanel, text="Tag Media:", width=10)
        self.label.pack(side=LEFT, padx=5, pady=3)
        self.media_list = [f.name for f in os.scandir(self.filesDir) if f.is_dir()]
        self.mediaName = StringVar()
        self.mediaName.set(self.media_list[0])
        self.mediaList = OptionMenu(self.upperPanel, self.mediaName, *self.media_list, command=self.setMedia)
        self.currentMedia = self.mediaName.get()  # init
        self.mediaList.pack(side=LEFT, padx=5, pady=3)
        self.mediaList.config(width=25)
        
        self.label = Label(self.frame, text="Eye:", width=10)
        self.label.grid(row=0, column=2, sticky=W + E)
        self.eyeName = StringVar()
        self.eyeName.set(self.eyes_list[0])
        self.eyeList = OptionMenu(self.frame, self.eyeName, *self.eyes_list, command=self.setEye)
        self.currentEye = self.eyeName.get()  # init
        self.eyeList.grid(row=0, column=3)
        self.eyeList.config(width=10)
        
        self.label = Label(self.frame, text="User:", width=10)
        self.label.grid(row=0, column=4, sticky=W + E)
        self.userName = StringVar()
        self.userName.set(self.users_list[0])
        self.userList = OptionMenu(self.frame, self.userName, *self.users_list, command=self.setUser)
        self.currentUser = self.userName.get()  # init
        self.userList.grid(row=0, column=5)
        self.userList.config(width=10)

        # center panel for labeling
        self.mainPanel = Canvas(self.frame, background="white", width=900, height=600)
        self.mainPanel.grid(row=1, column=1, rowspan=3, columnspan=1, sticky=W + N)
        
        # right panel
        self.ldBtn = Button(self.frame, text="Load Dir", width=20, background="yellow1", command=self.loadDir)
        self.ldBtn.grid(row=1, column=2, rowspan=1, columnspan=4, sticky=W + E)
        
        self.btnClass = Button(self.frame, text='Save & Export Labels', width=10, background="green1", command=self.saveAndExport)
        self.btnClass.grid(row=2, column=2, rowspan=1, columnspan=2, sticky=W + E)
        
        self.btnClass = Button(self.frame, text='Reset Labels', width=10, background="red1", command=self.resetLabels)
        self.btnClass.grid(row=2, column=4, rowspan=1, columnspan=2, sticky=W + E)
        
        self.plotCurrEyePanel = Canvas(self.frame, background="white", width=360, height=550)
        self.plotCurrEyePanel.grid(row=3, column=2, rowspan=1, columnspan=4, sticky=W + N)

        # control panel for image navigation
        self.ctrPanel1 = Frame(self.frame)
        self.ctrPanel1.grid(row=7, column=1, rowspan=1, columnspan=1, sticky=W + E)
        self.prevBtn = Button(self.ctrPanel1, text='<< Prev', width=10, background="orange", command=self.prevImage)
        self.prevBtn.pack(side=LEFT, padx=5, pady=3)
        self.increaseBtn = Button(self.ctrPanel1, text='↑ Increase ↑', width=10, background="cyan", command=self.increaseLabel)
        self.increaseBtn.pack(side=LEFT, padx=5, pady=3)
        self.decreaseBtn = Button(self.ctrPanel1, text='↓ Decrease ↓', width=10, background="cyan", command=self.decreaseLabel)
        self.decreaseBtn.pack(side=LEFT, padx=5, pady=3)
        self.nextBtn = Button(self.ctrPanel1, text='Next >>', width=10, background="orange", command=self.nextImage)
        self.nextBtn.pack(side=LEFT, padx=5, pady=3)
        
        self.currentLabelValue = Label(self.ctrPanel1, text="")
        self.currentLabelValue.pack(side=RIGHT, padx=5)
        self.currentLabelTitle = Label(self.ctrPanel1, text="Current Label %s:" % str(eye_open_levels))
        self.currentLabelTitle.pack(side=RIGHT, padx=5)
        
        self.ctrPanel2 = Frame(self.frame)
        self.ctrPanel2.grid(row=8, column=1, rowspan=1, columnspan=1, sticky=W + E)
        self.nextBtn = Button(self.ctrPanel2, text='Zoom In', width=10, background="magenta", command=self.zoomIn)
        self.nextBtn.pack(side=LEFT, padx=5, pady=3)
        self.nextBtn = Button(self.ctrPanel2, text='Zoom Out', width=10, background="magenta", command=self.zoomOut)
        self.nextBtn.pack(side=LEFT, padx=5, pady=3)
        
        self.goBtn = Button(self.ctrPanel2, text='Go', background="magenta", command=self.gotoImage)
        self.goBtn.pack(side=RIGHT)
        self.idxEntry = Entry(self.ctrPanel2, width=5)
        self.idxEntry.pack(side=RIGHT, padx=5)
        self.tmpLabel = Label(self.ctrPanel2, text="Go to Image No.")
        self.tmpLabel.pack(side=RIGHT, padx=5)
        self.progLabel = Label(self.ctrPanel2, text="Progress:     /    ")
        self.progLabel.pack(side=RIGHT, padx=5)
        
        # right low panel for key menu (60 characters per line)
        self.keyMenuText = "                                ** Key Menu **                       \r"\
        "     -Left (←): Go backforward                    -Right (→): Go forward   \r"\
        "       -Up (↑): Increase label                           -Down (↓): Decrease label\r"\
        "  -BackSpace: Zoom out                         -Space: Zoom in          \r"
        
        self.label = Label(self.frame, text=self.keyMenuText)
        self.label.grid(row=7, column=2, rowspan=2, columnspan=4, sticky=W)
        
        # key binding
        self.parent.bind("<Left>", self.prevImage)         # press left arrow to go backforward
        self.parent.bind("<Right>", self.nextImage)        # press right arrow to go forward
        self.parent.bind("<Down>", self.decreaseLabel) # press down arrow to adjust label (decrease)
        self.parent.bind("<Up>", self.increaseLabel)      # press up arrow to adjust label (increase)
        self.parent.bind("<BackSpace>", self.zoomOut)     # press backspace to zoom out
        self.parent.bind("<space>", self.zoomIn)          # press space to zoom in
        
    # define functions
    def loadDir(self):
        s = self.dirEntry.get()
        self.workingDir = os.path.join(self.filesDir,self.currentMedia)

        # get image list
        self.imageDir = os.path.join(self.workingDir,'frames_for_tag')
        self.imageList = [f for f_ in [glob.glob(os.path.join(self.imageDir, e), recursive=True) for e in
                                       ('.\**\*.JPG', '.\**\*.PNG')] for f in f_]
                                       
        # check if path includes images
        if len(self.imageList) == 0:
            print('Dir load failed! No images found in the specified dir!')
            ctypes.windll.user32.MessageBoxW(0,'Dir load failed! No images found in the specified dir!',"Message",0)
            return
            
        # get label dir
        self.labelDir = self.workingDir
        
        self.frame_names = [self.currentMedia+'_frame'+"{:04d}".format(frame_number) for frame_number in range(0,len(self.imageList))]
        self.df_labels = pd.DataFrame()
        self.df_labels['frame_name'] = self.frame_names
                
        for user in self.users_list:
            for eye in self.eyes_list:        
                try:
                    load_xlsx = pd.ExcelFile(self.labelDir+'\\'+self.currentMedia+"_labels.xlsx")
                    load_sheet = pd.read_excel(load_xlsx,'labels')
                    label_column = load_sheet[user+'_'+eye].values
                except:
                    label_column = pd.Series([np.nan]*len(self.imageList))
        
                self.df_labels[user+'_'+eye] = pd.to_numeric(label_column, downcast='float')
            
        
        # default to the 1st image in the collection
        self.cur = 0
        self.total = len(self.imageList)
        self.dir_is_active = 1
        self.currentLabel = self.currentUser+'_'+self.currentEye
        self.video_fps = int(self.currentMedia[-5:-3])
        self.loadImage()
        ctypes.windll.user32.MessageBoxW(0,'Dir load completed! %d images loaded from %s!' % (self.total, self.currentMedia),"Message",0)

    def loadImage(self):
        # load image
        imagePath = self.imageList[self.cur]
        self.img = Image.open(imagePath)
        self.img = self.img.resize([int(self.zoom * s) for s in self.img.size])
        self.mainPanel.config(width=900, height=600)
        self.progLabel.config(text="%04d/%04d" % (self.cur, self.total-1))
        
        # initialise the drawing context with  the image object as background
        self.draw = ImageDraw.Draw(self.img)
        # create font object with the font file and specify desired size         
        font = ImageFont.truetype("arial.ttf", 20)
                
         # show current label
        if (self.df_labels[self.currentLabel].isnull()[self.cur]):
            self.currentLabelIndex = 0
        else:
            self.currentLabelIndex = eye_open_levels.index(self.df_labels[self.currentLabel][self.cur])
        
        self.currentLabelValue.config(text="%.2f (%s)" % (self.df_labels[self.currentLabel][self.cur],self.levelToPercentage(self.df_labels[self.currentLabel][self.cur])))
        
        # print labels
        i = 0
        for user in self.users_list:
            if (self.currentUser == user):
                if (self.currentEye == 'right'):
                    color_text_r = c_lime
                    color_text_l = c_red
                else:
                    color_text_r = c_red
                    color_text_l = c_lime
            else:
                    color_text_r = c_red
                    color_text_l = c_red
             
            # draw the message on the background
            self.draw.text(text_pos_r[i], "R_"+user+"_label: {:.2f}".format(self.df_labels[user+'_right'][self.cur]),
                      fill=color_text_r, font=font)
            self.draw.text(text_pos_l[i], "L_"+user+"_label: {:.2f}".format(self.df_labels[user+'_left'][self.cur]),
                      fill=color_text_l, font=font)
            
            i = i + 1
        
        self.tkImg = ImageTk.PhotoImage(self.img)
        self.mainPanel.create_image(0, 0, image=self.tkImg, anchor=NW)

        #### print graphs
        matplotlib.pyplot.close('all') # close all figs before creating new one
        self.figure = plt.figure(figsize=(4.5,9),dpi=70)
        
        # create column fo frame number and append it to data frame
        df_frame_number = self.df_labels['frame_name'].str[-4:].astype(int)
        df_frame_elapsed_time = df_frame_number.apply(lambda x: round(float(x/self.video_fps),self.precision_digits))
        self.df_labels['frame_number'] = df_frame_number
        self.df_labels['elapsed_time'] = df_frame_elapsed_time
        
        # plot graph for right eye
        ax_right = plt.subplot(1,2,1)
        
        # plot users labels
        i = 0
        for user_eye in self.df_labels.columns:
            if (any(substring in user_eye for substring in self.users_list) and any(substring in user_eye for substring in ['right'])):
                if (user_eye == self.currentLabel):
                    self.df_labels.plot(kind='line',style=line_styles[i%len(line_styles)],x=user_eye,y='frame_number',
                                   color=tuple(np.divide(c_lime,255)),ax=ax_right,label=user_eye)
                    plt.scatter(self.df_labels[self.currentLabel][self.cur], self.cur, c=tuple(np.divide(c_blue,255)))  
                else:
                    self.df_labels.plot(kind='line',style=line_styles[i%len(line_styles)],x=user_eye,y='frame_number',
                                   color=tuple(np.divide(c_red,255)),ax=ax_right,label=user_eye)
                i = i + 1
                
        # edit titles and axis
        plt.title(self.currentMedia+': labels vs frame_number',loc='left')
        plt.xlabel("labels_right")
        plt.ylabel('frame_number')
        ax_right.set_xlim([min(eye_open_levels[1:]),max(eye_open_levels[1:])])
        ax_right.set_ylim([df_frame_number.iloc[0],df_frame_number.iloc[-1]])
        ax_right.legend(loc='upper left')
        
        # plot graph for left eye
        ax_left = plt.subplot(1,2,2)
        
        # plot features
        i = 0
        for user_eye in self.df_labels.columns:
            if (any(substring in user_eye for substring in self.users_list) and any(substring in user_eye for substring in ['left'])):
                if (user_eye == self.currentLabel):
                    self.df_labels.plot(kind='line',style=line_styles[i%len(line_styles)],x=user_eye,y='frame_number',
                                   color=tuple(np.divide(c_lime,255)),ax=ax_left,label=user_eye)
                    plt.scatter(self.df_labels[self.currentLabel][self.cur], self.cur, c=tuple(np.divide(c_blue,255)))  
                else:
                    self.df_labels.plot(kind='line',style=line_styles[i%len(line_styles)],x=user_eye,y='frame_number',
                                   color=tuple(np.divide(c_red,255)),ax=ax_left,label=user_eye)
                i = i + 1
        
        # edit titles and axis
        plt.xlabel('labels_left')
        ax_left.set_xlim([min(eye_open_levels[1:]),max(eye_open_levels[1:])])
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
    
    def levelToPercentage(self, level):
        precentageFloat = level*100
        precentageString = "%.0f%s" % (precentageFloat,'%')
        return precentageString
    
    def prevImage(self, event=None):
        if self.cur == 0:
            self.cur = self.total-1
        else:
            self.cur -= 1
        
        self.loadImage()

    def nextImage(self, event=None):
        
        # determine current label (if the current label is 'NaN', then it is determined relatively to previous label)
        self.cur_prev = self.cur
        self.previous_label = self.df_labels[self.currentLabel][self.cur_prev]
        
        if self.cur == self.total-1:
            self.cur = 0
            print('Finish tagging current eye!')
            ctypes.windll.user32.MessageBoxW(0,'Finish tagging current eye!',"Message",0)
            
        else:
            self.cur += 1
           
        if ((self.df_labels[self.currentLabel].isnull()[self.cur]) and (self.previous_label is not np.nan)):
            self.df_labels.at[self.cur,self.currentLabel] = self.previous_label        
        
        self.loadImage()

    def decreaseLabel(self, event=None):
        self.decrease_label = eye_open_levels[self.currentLabelIndex-1]
        self.df_labels.at[self.cur,self.currentLabel] = self.decrease_label
        self.loadImage()
            
    def increaseLabel(self, event=None):
        if (self.currentLabelIndex == number_of_levels):
            self.increase_label = eye_open_levels[0]
        else:
            self.increase_label = eye_open_levels[self.currentLabelIndex+1]
        
        self.df_labels.at[self.cur,self.currentLabel] = self.increase_label
        self.loadImage()
            
    def gotoImage(self):
        idx = int(self.idxEntry.get())
        if 0 <= idx <= self.total-1:
            self.cur = idx
            self.loadImage()
    
    def setEye(self, event=None):
        self.currentEye = self.eyeName.get()
        print('set current eye to :', self.currentEye)

    def setUser(self, event=None):
        self.currentUser = self.userName.get()
        print('set current user to :', self.currentUser)
        
    def setMedia(self, event=None):
        self.currentMedia = self.mediaName.get()
        print('set current media to :', self.currentMedia)
    
    def saveAndExport(self, event=None):
        if (self.dir_is_active == 1):
            self.df_labels.to_excel(self.labelDir+'\\'+self.currentMedia+"_labels.xlsx",sheet_name='labels')
            print('Save and Export completed!')
            ctypes.windll.user32.MessageBoxW(0,'Save and Export completed!',"Message",0)
            # save plot to file
            plt.savefig(self.labelDir+'\\'+self.currentMedia+"_labels_graphs.png",bbox_inches='tight',dpi=300)
        else:
            print('Save and Export faild! dir is not active!')
            ctypes.windll.user32.MessageBoxW(0,'Save and Export faild! dir is not active!',"Message",0)
    
    def resetLabels(self, event=None):
        if (self.dir_is_active == 1):
            self.df_labels[self.currentLabel] = pd.Series([np.nan]*len(self.imageList))
            print('Reset labels completed! (%s column set to NaN)!' % (self.currentLabel))
            ctypes.windll.user32.MessageBoxW(0,'Reset labels completed! (%s column set to NaN)!' % (self.currentLabel),"Message",0)
            self.loadImage()
        else:
            print('Reset labels faild! dir is not active!')
            ctypes.windll.user32.MessageBoxW(0,'Reset labels faild! dir is not active!',"Message",0)
    
    def zoomIn(self, event=None):
        self.zoom *= 1.2
        self.loadImage()

    def zoomOut(self, event=None):
        self.zoom /= 1.2
        self.loadImage()

if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    root.resizable(width=True, height=True)
    root.state('zoomed')
    root.mainloop()
