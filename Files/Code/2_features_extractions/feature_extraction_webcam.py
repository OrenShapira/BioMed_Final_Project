# =============================== USAGE: ===============================
# python feature_extraction.py --user TOM/OREN --input palsy/normal_xx_y_zzfps.mp4 
#(xx = video index, y = palsy eye (l/r/n), zz = fps)
# 
# Optionals:
# (--ear, --r2, --ellipse, --poly) : Features to be considered (default: 1)
# --pictures                       : Save pictures containing each frame (default: 1)
# --graphs                         : Save graph of the scores given to each frame (default: 1)
# --wlen                           : Length of sliding window in seconds (default: 2)
# ======================================================================

# Import necessary packages
#from imutils.video import VideoStream
from imutils import face_utils
import pandas as pd
import numpy as np
import math
import argparse
#import imutils
import time
import dlib
import cv2
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from io import StringIO
from csv import writer
from imutils.video import FPS
import glob

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
colors = [C_ORANGE,C_PURPLE,C_MAGENTA,C_CYAN,C_SILVER,C_LIME,C_YELLOW,C_WHITE,C_BLACK,C_BLUE,C_RED]

# ====================================================
# DEFINE FUNCTIONS
# ====================================================            
# Update structs after extraction
def update_structs(action, frame_number, window_len_frame, curr_h_score, curr_p_score,
                           h_scores, p_scores, min_h_score, max_h_score, min_p_score, max_p_score):
    # Append the score of the current frame
    h_scores.append(curr_h_score)
    p_scores.append(curr_p_score)
    
    # 6. Delete the last score, if necessary
    if action == 0 and frame_number > window_len_frame:
        del h_scores[0]
        del p_scores[0]
    
    # 7. Shrink the window of scores and recalculate extremum values, if necessary
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

# ===== Define functions for feature extraction =====
# Calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of vertical eye landmarks (x,y) coordinates
    A = math.sqrt(sum([(a-b)**2 for a,b in zip(eye[1],eye[5])]))
    B = math.sqrt(sum([(a-b)**2 for a,b in zip(eye[2],eye[4])]))
    # compute the euclidean distance between the horizontal eye landmark (x,y) coordinate
    C = math.sqrt(sum([(a-b)**2 for a,b in zip(eye[0],eye[3])]))
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

# calculate R^2 score for P1, P2, P3 and P4
def calc_r2(eye):
    x = eye[0:4,0]
    y = eye[0:4,1]
    X = x[:, np.newaxis]
    linreg = LinearRegression()
    linreg.fit(X,y)
    y_pred = linreg.predict(X)
    return [X, y_pred, 1 - r2_score(y, y_pred)]

# Calculate the matched elipse area
def calc_ellipse_area(eye):
    (xe, ye), (MA, ma), angle = cv2.fitEllipse(eye)
    A = np.pi * MA * ma
    return A
    
# Calculate the area of a polygon defined by (x,y) coordinates using Shoelace formula
def poly_area(eye):
    x = eye[:,0]
    y = eye[:,1]
    
    correction = x[-1] * y[0] - y[-1]* x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5*np.abs(main_area + correction)
    #return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    
# ===== Normalize feature score for the current frame =====
# Decide when to increase or shrink the window size
def define_window_size (curr_feature, curr_score, min_score, max_score, last_score):
    last_score_n = normalize_score(last_score, min_score, max_score, PREC_DIGITS)
    curr_score_n = normalize_score(curr_score, min_score, max_score, PREC_DIGITS)    
    if ((last_score_n < 0.2) and (curr_score_n < 0.2)) or ((last_score_n > 0.8) and (curr_score_n > 0.8)):
        # Shrink the window to the original size
        return -1    
    if (last_score_n < 0.1) or (last_score_n > 0.9):
        # Increase the window size - save the last score
        return 1    
    # If reached there, behave as usual (remove the last score)
    return 0

# Normalize feature score for current frame based on the latest scores
def normalize_feature_score(video_palsy_eye, curr_feature, 
                            curr_h_score, min_h_feature, max_h_feature,
                            curr_p_score, min_p_feature, max_p_feature, 
                            PREC_DIGITS):        
    # Normalize according to the healthy eye, if there is a palsy eye
    curr_h_score_n = normalize_score(curr_h_score, min_h_feature, max_h_feature, PREC_DIGITS)    
    if video_palsy_eye != 'n':
        curr_p_score_n = normalize_score(curr_p_score, min_h_feature, max_p_feature, PREC_DIGITS)
    else:
        curr_p_score_n = normalize_score(curr_p_score, min_p_feature, max_p_feature, PREC_DIGITS)    
    # return values
    return [curr_h_score_n, curr_p_score_n]

# Normalize score based on min and max values
def normalize_score(curr_score, min_score, max_score, PREC_DIGITS):
    normalized_score = (curr_score - min_score) / (max_score - min_score)
    if normalized_score > 1.0:
        normalized_score = 1.0
    elif normalized_score < 0.0:
        normalized_score = 0.0
    
    return round(float(normalized_score) ,PREC_DIGITS)

# ====================================================
# CONSTRUCT THE ARGUMENT PARSE AND PARSE THE ARGUMENTS
# ====================================================
# Define argument names
ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True, help="user name")
ap.add_argument("--palsy_eye", required=True, help="location_of_the_palsy_eye")
ap.add_argument("--palsy", default=1, help="whether to use the palsy_eye feature")
ap.add_argument("--ear", type=int, default=1, help="whether to use EAR feature")
ap.add_argument("--r2", type=int, default=1, help="whether to use 1-r^2 feature")
ap.add_argument("--ellipse", type=int, default=1, help="whether to use ellipse area feature")
ap.add_argument("--poly", type=int, default=1, help="whether to use polygon area feature")
ap.add_argument("--pictures", type=int, default=1, help="whether to save frame pictures")
ap.add_argument("--graphs", type=int, default=1, help="whether to show graphs of scores")
ap.add_argument("--times", type=int, default=1, help="show timing histograms")
ap.add_argument("--wlen", type=int, default=2, help="length of sliding window in seconds")
args = vars(ap.parse_args())

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
window_len_frame = math.ceil(video_fps * args["wlen"]) 

# Get the location of the palsy eye
video_palsy_eye = args["palsy_eye"]
if video_palsy_eye == 'n':
    palsy_prefix = "normal_"
else:
    palsy_prefix = "palsy_"

# Search for the current webcam index
video_index = 1
while True:
    partial_path = "files/" + palsy_prefix + str(video_index).zfill(2)
    if len(glob.glob(partial_path + "*")) == 0:
        break
    video_index = video_index + 1

# Create subfolders for outputs
output_path = partial_path + '_' + video_palsy_eye + '_' + str(video_fps).zfill(2) + "fps"
output_path_frames_features = output_path + "/frames_features"
output_path_frames_for_tag  = output_path + "/frames_for_tag"

if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(output_path_frames_features) and args["pictures"]:
    os.makedirs(output_path_frames_features)
if not os.path.exists(output_path_frames_for_tag) and args["pictures"]:
    os.makedirs(output_path_frames_for_tag)
    
output_path_full = output_path + "/" + output_path[13:]

print("[INFO] Successfully created subfolders for output files: ", output_path)

# Initialize structs
df_column = []          # list of active features
h_scores = {}           # Dictionary of lists, each represents a sliding window of a specific feature
p_scores = {}           #
min_h_score = {}        # Dictionary of extremum values, each represents a specific feature
max_h_score = {}
min_p_score = {}
max_p_score = {}

# Initialize program's structs
feature_list = ['ear','r2','ellipse','poly']
for feature in feature_list:
    if args[feature]:
        df_column.append(feature)
        h_scores[feature]     = []
        p_scores[feature]     = []
        min_h_score[feature]  = 0
        max_h_score[feature]  = 0
        min_p_score[feature]  = 0
        max_p_score[feature]  = 0
            
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

# ====================================================
# FEATURE EXTRACTION FROM VIDEO
# ====================================================
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

frame_number = 0
rects_old = []
action = 0

land_time = []
ear_time = []
r2_time = []
ellipse_time = []
poly_time = []
norm_time = []
df_time = []
frame_time = []
total_time = []

output_h = StringIO()
output_p = StringIO()
csv_writer_h = writer(output_h)
csv_writer_p = writer(output_p)

fps = FPS().start()

# Loop over frames from the video stream
while True:
    total_time_frame = 0
    frame_time_start = time.clock()

    # Grab the frame from the threaded video file stream
    (grabbed, frame) = stream.read()
    
    if not grabbed:
        break

    frame_time_end = time.clock()
    frame_time.append(1000*(frame_time_end - frame_time_start))
    total_time_frame = total_time_frame + (frame_time_end - frame_time_start)
    

    # Detect faces in the grayscale frame
    if frame_number == 0:
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       rects = detector(gray, 0)
    
    # If no faces were detected, extract facial landmarks based on the bounding box from the previous frame
    if (len(rects) != 1):
        rects = rects_old
    rects_old = rects
    
    # frame name
    frame_name = output_path[14:]+'_frame'+"{:04d}".format(frame_number)
    
    if args["pictures"]:
        # show frame name and elapsed time
        elapsed_time = frame_number / video_fps
        cv2.putText(frame, frame_name+", T = {:.2f} Sec".format(elapsed_time), text_pos_m[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[-1], 2)
        
        # save frame to frames_for_tag
        cv2.imwrite(output_path_frames_for_tag+"/"+frame_name+".jpg", frame)
    
    if (len(rects) != 1):
        # notice about faces number on current frame
        if args["pictures"]:
            cv2.putText(frame, "{:d} faces detected!".format(len(rects)), text_pos_m[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[-1], 2)
        
        # prepare NaN raw for current frame
        df_nan_row = [float('NaN')]*(len(df_column)-1)
        
        # create new Dataframe for first frame, otherwise concatenate to exist DataFrame
        if (frame_number == 0):
            #df_h_scores = pd.DataFrame([df_nan_row],columns = df_column)
            #df_p_scores = pd.DataFrame([df_nan_row],columns = df_column)
            df_h_scores_n = pd.DataFrame([df_nan_row],columns = df_column)
            df_p_scores_n = pd.DataFrame([df_nan_row],columns = df_column)
        else:
            #df_h_scores = pd.concat([df_h_scores, pd.DataFrame([df_nan_row],columns = df_column)])
            #df_p_scores = pd.concat([df_p_scores, pd.DataFrame([df_nan_row],columns = df_column)])
            df_h_scores_n.loc[frame_number] = [df_nan_row]
            df_p_scores_n.loc[frame_number] = [df_nan_row]
        
    else:
        if(args["pictures"]):
            (x, y, w, h) = face_utils.rect_to_bb(rects[0])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # determine the facial landmarks for the face region, then
        # Convert the facial landmark (x,y) coordinates to a NumPy array
        land_time_start = time.clock()
        
        shape = predictor(frame, rects[0])
        shape = face_utils.shape_to_np(shape)
        
        land_time_end = time.clock()
        land_time.append(1000*(land_time_end - land_time_start))
        total_time_frame = total_time_frame + (land_time_end - land_time_start)
        
        # Extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        h_eye_shape = shape[hStart:hEnd]
        p_eye_shape = shape[pStart:pEnd]
        
        # Initialize rows for DataFrames
        #df_h_row = []
        #df_p_row = []
        df_h_row_n = []
        df_p_row_n = []
        
        # Feature index
        feature_index = 0
        
        norm_time_total = 0
        
        # ==== Calculate relevant features ====
        for feature in df_column:
            # Extract features
            if feature   == 'ear':
                ear_time_start = time.clock()
                
                curr_h_score = eye_aspect_ratio(h_eye_shape)
                curr_p_score = eye_aspect_ratio(p_eye_shape)
                
                ear_time_end = time.clock()
                ear_time.append(1000*(ear_time_end - ear_time_start))
                total_time_frame = total_time_frame + (ear_time_end - ear_time_start)
                
            elif feature == 'r2':
                r2_time_start = time.clock()
                
                [x_h, y_pred_h, curr_h_score] = calc_r2(h_eye_shape)
                [x_p, y_pred_p, curr_p_score] = calc_r2(p_eye_shape)
                
                r2_time_end = time.clock()
                r2_time.append(1000*(r2_time_end - r2_time_start))
                total_time_frame = total_time_frame + (r2_time_end - r2_time_start)
                
            elif feature == 'ellipse':
                ellipse_time_start = time.clock()
                
                curr_h_score = calc_ellipse_area(h_eye_shape)
                curr_p_score = calc_ellipse_area(p_eye_shape)
                
                ellipse_time_end = time.clock()
                ellipse_time.append(1000*(ellipse_time_end - ellipse_time_start))
                total_time_frame = total_time_frame + (ellipse_time_end - ellipse_time_start)
                
            elif feature == 'poly':
                poly_time_start = time.clock()
                
                curr_h_score = poly_area(h_eye_shape)
                curr_p_score = poly_area(p_eye_shape)
                
                poly_time_end = time.clock()
                poly_time.append(1000*(poly_time_end - poly_time_start))
                total_time_frame = total_time_frame + (poly_time_end - poly_time_start)
            
            # Append to row for data prame
            #df_h_row.append(curr_h_score)
            #df_p_row.append(curr_p_score)
            
            # Calculate normalized values
            norm_time_start = time.clock()
            
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
            
            # Append to row for normalized data frame
            df_h_row_n.append(curr_h_score_n)
            df_p_row_n.append(curr_p_score_n)
            
            # ===== Updates windows and extremum values =====
            [h_scores[feature], p_scores[feature], min_h_score[feature], max_h_score[feature], min_p_score[feature], max_p_score[feature]] = update_structs(
                    action,frame_number, window_len_frame, curr_h_score, curr_p_score, h_scores[feature], p_scores[feature], 
                    min_h_score[feature], max_h_score[feature],min_p_score[feature], max_p_score[feature])
            
            norm_time_end = time.clock()
            norm_time_total = norm_time_total + (norm_time_end - norm_time_start)
            
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
            
            if feature == 'ear':                   
                for (x,y) in h_eye_shape:
                    cv2.circle(frame, (x,y), 1, C_RED, -1)
                for (x,y) in p_eye_shape:
                    cv2.circle(frame, (x,y), 1, C_RED, -1)
                            
            
            # Visualize features (if necessary)
            if (args["pictures"]):                    
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
        
    norm_time.append(1000 * norm_time_total)
    total_time_frame = total_time_frame + norm_time_total
    
    # Create new Dataframe for first frame, otherwise concatenate to exist DataFrame
    df_time_start = time.clock()
    
    if (frame_number == 0):
        #df_h_scores = pd.DataFrame([df_h_row],columns = df_column)
        #df_p_scores = pd.DataFrame([df_p_row],columns = df_column)
        csv_writer_h.writerow(df_h_row_n)
        csv_writer_p.writerow(df_p_row_n)
        
        #df_h_scores_n = pd.DataFrame([df_h_row_n],columns = df_column)
        #df_p_scores_n = pd.DataFrame([df_p_row_n],columns = df_column)
        
        if args["pictures"]:
            # Get frame dimentions
            height, width = frame.shape[:2]
            # Define the codec and create VideoWriter object
            video_out = cv2.VideoWriter(output_path_full+"_out.mp4",cv2.VideoWriter_fourcc('m','p','4','v'), video_fps, (width,height))
    else: 
        #df_h_scores = pd.concat([df_h_scores, pd.DataFrame([df_h_row],columns = df_column)],ignore_index=True)
        #df_p_scores = pd.concat([df_p_scores, pd.DataFrame([df_p_row],columns = df_column)],ignore_index=True)
        #df_h_scores_n.loc[frame_number] = df_h_row_n
        #df_p_scores_n.loc[frame_number] = df_p_row_n
        csv_writer_h.writerow(df_h_row_n)
        csv_writer_p.writerow(df_p_row_n)
        
    df_time_end = time.clock()
    df_time.append(1000* (df_time_end - df_time_start))
    total_time_frame = total_time_frame + (df_time_end - df_time_start)
    
    # Display the resulting frame
    cv2.imshow('Frame', frame)
            
    if args["pictures"]:
        # Save resulting frame to frames_features
        cv2.imwrite(output_path_frames_features+"/"+frame_name+".jpg", frame)
        # Save resulting frame to output video
        video_out.write(frame)
        
    # Press ESC on keyboard to break
    key = cv2.waitKey(int(1000/video_fps)) & 0xFF
    if key == 27:
        break
        
    # Promote frame
    frame_number = frame_number + 1
    fps.update()
    
    total_time.append(total_time_frame*1000)

output_h.seek(0) # we need to get back to the start of the BytesIO
df_h_scores_n = pd.read_csv(output_h, names=df_column)
output_p.seek(0) # we need to get back to the start of the BytesIO
df_p_scores_n = pd.read_csv(output_p, names=df_column)

if args["palsy"]:
    df_h_scores_n['palsy_eye'] = 0
    if video_palsy_eye == 'n':
        df_p_scores_n['palsy_eye'] = 0
    else:
        df_p_scores_n['palsy_eye'] = 1

# Save normalized data frames as excel files
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
if args["pictures"]:
    video_out.release()

#vs.stop()
stream.release()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()

# ======== Show Graphs ========
if args["graphs"]: 
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
    
# ======== Show Timings ========
if args["times"]:
    # Show histogram of landmarks detection
    f1 = plt.figure('Histograms')
    plt.subplot(3,3,1)
    plt.hist(land_time, color = 'red')
    plt.ylabel('Facial Landmarks Dectection')
    
    # Show histogram of EAR
    plt.subplot(3,3,2)
    plt.hist(ear_time, color='orange')
    plt.ylabel('EAR Extraction')
    
    # Show histogram of r2
    plt.subplot(3,3,3)
    plt.hist(r2_time, color='cyan')
    plt.ylabel('1-r^2 Extraction')
    
    # Show histogram of ellipse
    plt.subplot(3,3,4)
    plt.hist(ellipse_time, color='purple')
    plt.ylabel('Ellipse Area Extraction')
    
    # Show histogram of ellipse
    plt.subplot(3,3,5)
    plt.hist(poly_time, color='magenta')
    plt.ylabel('Polygon Area Extraction')
    
    # Show histogram of normalization
    plt.subplot(3,3,6)
    plt.hist(norm_time, color='green')
    plt.ylabel('Normalization')
    
    # Show histogram of dataframe addition
    plt.subplot(3,3,7)
    plt.hist(df_time, color='gray')
    plt.ylabel('Adding to DataFrame')
    plt.xlabel('Processing time [ms]')
    
    # Show histogram of frame extraction
    plt.subplot(3,3,8)
    plt.hist(frame_time, color='brown')
    plt.ylabel('Occurances - Frame Extraction')
    plt.xlabel('Processing time [ms]')
     
    # Show histogram of frame extraction    
    plt.subplot(3,3,9)
    plt.ylabel('Total Processing')
    plt.hist(total_time, color='blue')
    plt.xlabel('Processing time [ms]')
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()  
    

print("[INFO] Done! :-)")

