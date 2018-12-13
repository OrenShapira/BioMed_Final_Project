# =============================== USAGE: ===============================
# python feature_extraction_video.py --input palsy/normal_xx_y_zzfps.mp4 
#(xx = video index, y = palsy eye (l/r/n), zz = fps)
# 
# Optionals:
# (--ear, --r2, --ellipse, --poly) : Features to be considered (default: 1)
# --pictures                       : Save pictures containing each frame (default: 1)
# --graphs                         : Save graph of the scores given to each frame (default: 1)
# --times                          : Plot histograms of performance analysis (default: 1)
# --wlen                           : Length of sliding window in seconds (default: 2)
# --face_frame                     : Detect faces every x frames (default: 1)
# ======================================================================

# Import necessary packages
import os
import shutil
import argparse
import cv2
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====================================================
# CONSTRUCT THE ARGUMENT PARSE AND PARSE THE ARGUMENTS
# ====================================================
# Define argument names
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, help="path to input video file")
ap.add_argument("--palsy", default=1, help="whether to use the palsy_eye feature")
ap.add_argument("--ear", type=int, default=1, help="whether to use EAR feature")
ap.add_argument("--r2", type=int, default=0, help="whether to use 1-r^2 feature")
ap.add_argument("--ellipse", type=int, default=1, help="whether to use ellipse area feature")
ap.add_argument("--poly", type=int, default=1, help="whether to use polygon area feature")
ap.add_argument("--pictures", type=int, default=0, help="whether to save frame pictures")
ap.add_argument("--tag", type=int, default=0, help="whether to save frames for tag")
ap.add_argument("--graphs", type=int, default=0, help="whether to show graphs of scores")
ap.add_argument("--times", type=int, default=0, help="show timing histograms")
ap.add_argument("--wlen", type=int, default=2, help="length of sliding window in seconds")
ap.add_argument("--face_frame", type=int, default=1, help="number of frames to search for faces")
args = vars(ap.parse_args())

# Calcultate parameters from arguments
video_path = '../../Database/' + args["input"]
video_path_r_output = '../../Database/r_' + args["input"]

video_index = args["input"][-14:-12]
video_fps = int(args["input"][-9:-7])
video_palsy_eye = args["input"][-11]

window_len_frame = math.ceil(video_fps * args["wlen"])

# Create subfolders for outputs
output_path = "files/" + args["input"][:-4]  
output_path_full = output_path + "/" + args["input"][:-4]
output_r_path = "files/r_" + args["input"][:-4]  
output_r_path_full = output_r_path + "/r_" + args["input"][:-4]

output_r_path_step3 = "../3_database_tagging/files/r_" + args["input"][:-4]  

# Define constants
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
colors = [C_ORANGE,C_CYAN,C_PURPLE,C_MAGENTA,C_MAGENTA,C_PURPLE,C_CYAN,C_SILVER,C_LIME,C_YELLOW,C_WHITE,C_BLACK,C_BLUE,C_RED]

# Define text position lists
text_pos_r = []
text_pos_m = []
text_pos_l = []
text_pos_gap = 15

# Define text locations
for i in range(0,10):
    text_pos_r.append((10,30+i*text_pos_gap))
    text_pos_m.append((10,15+i*text_pos_gap))
    text_pos_l.append((300,30+i*text_pos_gap))
	
#### step 1: create reverse video
print("step 1: create reverse video")
# videoCapture method of cv2 return video object 
# Pass absolute address of video file 
cap = cv2.VideoCapture(video_path) 

# counter variable for counting frames 
counter = 0
frame_list = [] 

# Initialize the value of check variable 
check = True

# If reached the end of the video then we got False value of check. 
# keep looping untill we got False value of check. 
while(check == True): 
	# read method of video object will return a tuple with 1st element denotes whether 
	# the frame was read successfully or not, 2nd element is the actual frame. 
	# Grab the current frame.  
	check , vid = cap.read() 
	# Add each frame in the list by using append method of the List 
	frame_list.append(vid) 
	# increment the counter by 1 
	counter += 1

# last value in the frame_list is None because when video reaches to the end 
# then false value store in check variable and None value is store in vide variable. 
# removing the last value from the frame_list by using pop method of List 
frame_list.pop() 

# reverse the order of the element present in the list by using 
# reverse method of the List.
frame_list.reverse() 
counter = len(frame_list)-1

# Press ESC on keyboard to break
for frame in frame_list: 
	
	# show frame name and elapsed time
    frame_name = args["input"][:-4]+'_frame'+"{:04d}".format(counter)
    elapsed_time = counter / video_fps
	
	# imwrite method of cv2 saves the image to the specified format.
	#cv2.putText(frame, frame_name+", T = {:.2f} Sec".format(elapsed_time), text_pos_m[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[-1], 2)
    #cv2.imwrite("frame%d.jpg" %counter , frame)
	
    if (counter == len(frame_list)-1):
        # Get frame dimentions
        height, width = frame.shape[:2]
        # Define the codec and create VideoWriter object
        video_out = cv2.VideoWriter(video_path_r_output,cv2.VideoWriter_fourcc('m','p','4','v'), video_fps, (width,height))
    
    # Display the resulting frame
    cv2.imshow('Frame', frame)
	# Save resulting frame to output video
    video_out.write(frame)
	
    counter -= 1
	
    key = cv2.waitKey(int(1000/video_fps)) & 0xFF
    if key == 27:
        break

cap.release()
video_out.release()
cv2.destroyAllWindows() 

#### step 2: run feature extraction on reversed video
print("step 2: run feature extraction on reversed video")
os.system("python feature_extraction_video.py --input r_"+str(args["input"])+" --tag 0 --face_frame "+str(args["face_frame"])+" --pictures 0 --graphs 0 --r2 "+str(args["r2"])+" --times 0")

#### step 3: run feature extraction on original video
print("step 3: run feature extraction on original video")
os.system("python feature_extraction_video.py --input "+str(args["input"])+" --tag "+str(args["tag"])+" --face_frame "+str(args["face_frame"])+" --pictures "+str(args["pictures"])+" --graphs 0 --r2 "+str(args["r2"])+" --times "+str(args["times"]))

#### step 4: fill missing cell in original excel from reverse excel
print("step 4: fill missing cell in original excel from reverse excel")

# load data frames from excel files
df_scores_n_xlsx = pd.ExcelFile(output_path_full+"_scores_n.xlsx")
df_scores_l_n = pd.read_excel(df_scores_n_xlsx,'left_eye')
df_scores_r_n = pd.read_excel(df_scores_n_xlsx,'right_eye')

df_r_scores_n_xlsx = pd.ExcelFile(output_r_path_full+"_scores_n.xlsx")
df_r_scores_l_n = pd.read_excel(df_r_scores_n_xlsx,'left_eye')
df_r_scores_r_n = pd.read_excel(df_r_scores_n_xlsx,'right_eye')

# reverse df_r_scores columns
df_r_scores_l_n.index = df_r_scores_l_n.index[::-1]
df_r_scores_r_n.index = df_r_scores_r_n.index[::-1]

df_r_scores_l_n = df_r_scores_l_n.sort_index()
df_r_scores_r_n = df_r_scores_r_n.sort_index()

# replace window NaN values with values from reversed dataframe
df_scores_l_n.iloc[0:window_len_frame+1] = df_r_scores_l_n[0:window_len_frame+1]
df_scores_r_n.iloc[0:window_len_frame+1] = df_r_scores_r_n[0:window_len_frame+1]

# Save total data frames as excel files
writer = pd.ExcelWriter(output_path_full+"_scores_n.xlsx")
df_scores_l_n.to_excel(writer,sheet_name='left_eye')
df_scores_r_n.to_excel(writer,sheet_name='right_eye')

writer.save()
writer.close()

#### step 5: create graph for full features
print("step 5: create graph for full features")

# Show Graphs
if args["graphs"]: 
    # load data frames from excel files
    df_scores_n_xlsx = pd.ExcelFile(output_path_full+"_scores_n.xlsx")
    df_scores_l_n = pd.read_excel(df_scores_n_xlsx,'left_eye')
    df_scores_r_n = pd.read_excel(df_scores_n_xlsx,'right_eye')
    
    df_column = list(df_scores_l_n.columns)
	
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
    for feature in df_column:
        df_scores_l_n.plot(kind='line',x='frame_number',y=feature,color=tuple(reversed(np.divide(colors[feature_index],255))),ax=ax_left)
        feature_index = feature_index + 1
    # edit titles and axis
    plt.title(args["input"]+': features_scores_n vs frame_number & elapsed_time')
    plt.xlabel("")
    plt.ylabel('scores_left_n')
    ax_left.set_xlim([df_frame_number.iloc[0],df_frame_number.iloc[-1]])
    ax_left.set_ylim([0,1])
    
    # plot graph for right eye
    ax_right = plt.subplot(2,1,2)
    # plot features
    feature_index = 0
    for feature in df_column:
        df_scores_r_n.plot(kind='line',x='frame_number',y=feature,color=tuple(reversed(np.divide(colors[feature_index],255))),ax=ax_right)
        feature_index = feature_index + 1
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
    plt.show()

#### step 6: delete unnecessary files
print("step 6: delete unnecessary files")

# delete from database
if os.path.exists(video_path_r_output):	
	os.remove(video_path_r_output)

# delete from step 2 files folder
if os.path.exists(output_r_path):		
	shutil.rmtree(output_r_path)

# delete from step 3 files folder
if os.path.exists(output_r_path_step3):
	shutil.rmtree(output_r_path_step3)