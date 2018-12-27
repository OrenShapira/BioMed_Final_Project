# =============================== USAGE: ===============================
# python generate_database.py --user TOM/OREN 
# 
# Optionals:
# (--ear, --r2, --ellipse, --poly) : Features to be considered (default: 1)
# -- exclude                       : Videos to be discarded from learning (default: [])
#                                    Usage:
#                                     --exclude normal/palsy : don't take healthy patients into account
#                                     --exclude normal a b palsy c: ignore normal_a, normal_b and palsy_c
# -- weights                       : The importance of the tags of each user (default: [0.5 0.5])
#                                    Usage:
#                                     --weights w1 w2 ... wN : for N users (default: N=2). All weights must sum into 1
# -- levels                        : The number of levels used for tagging (default: 5)
# --graphs                         : Save graph of the scores given to each frame (default: 1)

                                             
# ======================================================================

# Import necessary packages
import argparse
import os
import numpy as np
import pandas as pd
from io import StringIO
from csv import writer
import matplotlib.pyplot as plt

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

PREC_DIGITS = 2

# Save a dictionary of genders for each one of the videos
gender_dict = {'normal_01_n_30fps':'M', 'palsy_01_r_30fps':'M', 'palsy_02_r_30fps':'M', 'palsy_03_r_30fps':'M',
               'palsy_04_l_25fps':'F', 'palsy_05_l_25fps':'F', 'palsy_06_l_25fps':'F', 'palsy_07_r_25fps':'M',
               'palsy_08_r_30fps':'M', 'palsy_09_r_30fps':'M', 'palsy_10_r_30fps':'M', 'palsy_11_l_30fps':'F',
               'palsy_12_l_30fps':'F', 'palsy_13_l_30fps':'F', 'palsy_14_r_30fps':'M', 'palsy_15_r_30fps':'M',
               'palsy_16_l_30fps':'M', 'palsy_17_r_30fps':'F', 'palsy_18_r_30fps':'F', 'palsy_19_r_25fps':'F',
               'palsy_20_r_30fps':'M', 'palsy_21_l_30fps':'M', 'palsy_22_l_30fps':'F', 'normal_02_n_30fps':'M',
               'normal_03_n_30fps':'F', 'normal_04_n_30fps':'M', 'normal_05_n_30fps':'F', 'normal_06_n_30fps':'F',
               'normal_07_n_30fps':'M', 'normal_08_n_30fps':'M', 'normal_09_n_30fps':'M', 'normal_10_n_30fps':'F',
               'normal_11_n_30fps':'F', 'normal_12_n_30fps':'M', 'normal_13_n_30fps':'F', 'normal_14_n_30fps':'F',
               'normal_15_n_30fps':'F', 'normal_16_n_30fps':'F'}

# ====================================================
# UNDERSTAND WHICH VIDEOS TO USE FOR LEARNING
# ====================================================
# Define argument names
ap = argparse.ArgumentParser()
ap.add_argument("--ear", type=int, default=1, help="whether to use EAR feature")
ap.add_argument("--r2", type=int, default=0, help="whether to use 1-r^2 feature")
ap.add_argument("--ellipse", type=int, default=1, help="whether to use ellipse area feature")
ap.add_argument("--poly", type=int, default=1, help="whether to use polygon area feature")
ap.add_argument("--exclude", nargs='*', default=[], help="videos to be discarded from learning")
ap.add_argument("--weights", type=float, nargs='*', default=[0.5,0.5], help="the importance of the tags of each user")
ap.add_argument("--levels", type=int, default=5, help="the number of levels used for tagging")
ap.add_argument("--graphs", type=int, default=1, help="whether to save graphs of scores")
args = vars(ap.parse_args())

# Find files that have extracted features
file_names_features = os.listdir("../2_features_extraction/files")

# Find files that have labels
file_names_labels = []

for folder in os.listdir("../3_database_tagging/files"):
    if os.path.exists("../3_database_tagging/files/" + folder + "/" + folder + "_labels.xlsx"):
        file_names_labels.append(folder)

# Intersect the generated lists to find the relevant files that can we learn from
file_names = list(set(file_names_features) & set(file_names_labels))

# Construct a list of file indices to exclude - normal and palsy
exclude_list = args["exclude"]
exclude_all_normal = False
exclude_all_palsy = False
i = 0
file_names_exclude = []

# Parse the arguments - understand which videos to exclude
while i < len(exclude_list):
    if exclude_list[i] == 'normal':
        if i == len(exclude_list) - 1 or exclude_list[i+1] == 'palsy':
            exclude_all_normal = True 
        flag = 'n'            
    elif exclude_list[i] == 'palsy':
        if i == len(exclude_list) - 1 or exclude_list[i+1] == 'normal':
            exclude_all_palsy = True
        flag = 'p'
    else:
        if flag == 'n':
            file_names_exclude.append('normal_' + exclude_list[i].zfill(2))
        elif flag == 'p':
            file_names_exclude.append('palsy_' + exclude_list[i].zfill(2))
    i = i + 1

# Omit the files specified by the user
file_names_final = sorted([file for file in file_names if not file.startswith(tuple(file_names_exclude))])

if exclude_all_normal:
    file_names_final = [file for file in file_names_final if not file.startswith('normal')]
if exclude_all_palsy:
    file_names_final = [file for file in file_names_final if not file.startswith('palsy')]
    
print('File names for database - after exclusion:')
for file in file_names_final:
    print(file)
    
# Extract the weights to be used
weights = args["weights"]

# Extract the amount of levels for tagging
levels = args["levels"]

# Define df_column
df_column = ['video_name', 'frame_number', 'elapsed_time', 'eye_location','gender']

# Add columns for features
if args["ear"]:
    df_column.append("ear")
if args["ellipse"]:
    df_column.append("ellipse")
if args["r2"]:
    df_column.append("r2")
if args["poly"]:
    df_column.append("poly")

df_column = df_column + ['palsy_eye', 'label']

# Set CSV variable and write to it
output = StringIO()
csv_writer = writer(output)
  
# ====================================================
# BUILD THE DATABASE
# ====================================================
print('Building database...')
for file in file_names_final:
    path_features = "../2_features_extraction/files/" + file + "/" + file + "_scores_n.xlsx"
    df_features_xlsx  = pd.ExcelFile(path_features)
    df_features_left  = pd.read_excel(df_features_xlsx,'left_eye')
    df_features_right = pd.read_excel(df_features_xlsx,'right_eye')
    
    path_labels = "../3_database_tagging/files/" + file + "/" + file + "_labels.xlsx"
    df_labels_xlsx = pd.ExcelFile(path_labels)
    df_labels = pd.read_excel(df_labels_xlsx,'labels')
    
    # Get all the columns that correspond to the left and right eyes
    tag_left_cols   = [col for col in df_labels.columns if col.endswith('_left')]
    tag_right_cols = [col for col in df_labels.columns if col.endswith('_right')]
    
    #  +++ Iterate on each frame - Start from LEFT eye +++
    for i in range(len(df_labels.index)):
        # +++ Start from LEFT eye +++
        df_row_left = [file]
        df_row_left.append(int(df_labels.loc[i,'frame_name'].split('_')[4][5:]))
        df_row_left.append(float(df_labels.loc[i,'elapsed_time']))
        df_row_left.append('left')
        df_row_left.append(gender_dict[file])
        # Continue only if there are non-null features AND there are non-null tags
        if not df_features_left.iloc[i,:-1].isnull().all() and not df_labels.loc[i,tag_left_cols].isnull().all():
            # Append normalized features
            df_row_left = df_row_left + df_features_left.iloc[i,:].values.tolist()
            
            # Determine the final tag based on the given weights
            sum_weights = 0
            sum_tags = 0
            k = 0
            # Use only labels which are non-null
            for tag_col in tag_left_cols:
                curr_label = df_labels.loc[i,tag_col]
                if not np.isnan(curr_label):
                    sum_weights = sum_weights + weights[k]
                    sum_tags = sum_tags + curr_label * weights[k]
                k = k + 1
            
            # Calculate the final label based on all tags
            left_label = np.round(sum_tags / sum_weights * (levels-1)) / (levels-1)
            df_row_left.append(left_label)
            
            # Write to CSV file
            csv_writer.writerow(df_row_left)
     
    # +++ Do the same for RIGHT eye +++
    for i in range(len(df_labels.index)):
        df_row_right = [file]
        df_row_right.append(int(df_labels.loc[i,'frame_name'].split('_')[4][5:]))
        df_row_right.append(float(df_labels.loc[i,'elapsed_time']))
        df_row_right.append('right')
        df_row_right.append(gender_dict[file])
        # Continue only if there are non-null features AND there are non-null tags
        if not df_features_right.iloc[i,:-1].isnull().all() and not df_labels.loc[i,tag_right_cols].isnull().all():
            # Append normalized features
            df_row_right = df_row_right + df_features_right.iloc[i,:].values.tolist()
            
            # Determine the final tag based on the given weights
            sum_weights = 0
            sum_tags = 0
            k = 0
            # Use only labels which are non-null
            for tag_col in tag_right_cols:
                curr_label = df_labels.loc[i,tag_col]
                if not np.isnan(curr_label):
                    sum_weights = sum_weights + weights[k]
                    sum_tags = sum_tags + curr_label * weights[k]
                k = k + 1
            
            # Calculate the final label based on all tags - do some averaging
            right_label = np.round(sum_tags / sum_weights * (levels-1)) / (levels-1)
            df_row_right.append(right_label)
            
            # Write to CSV file
            csv_writer.writerow(df_row_right)
            

output.seek(0) # we need to get back to the start of the BytesIO
df_db = pd.read_csv(output, names=df_column)
writer = pd.ExcelWriter("files/database.xlsx")
df_db.to_excel(writer,sheet_name='database')

writer.save()
writer.close()


# === SAVE GRAPHS OF FEATURES VS. LABELS, FOR EACH VIDEO IN THE DATABASE ===
if args["graphs"]:
    print('Saving graphs...')
    # Load data frames from excel files
    df_db_xlsx = pd.ExcelFile("files/database.xlsx")
    df_db = pd.read_excel(df_db_xlsx,'database')
    
    # Run for each video in the database
    for file in file_names_final:
        # Choose only frames which are relevant for the current video & eye 
        df_file_l = df_db.loc[(df_db['video_name'] == file) & (df_db['eye_location'] == 'left')]
        df_file_r = df_db.loc[(df_db['video_name'] == file) & (df_db['eye_location'] == 'right')]
        
        plt.figure(figsize=(14,7))
    
        # plot graph for left eye
        ax_left = plt.subplot(2,1,1)
        # plot features and labels
        feature_index = 0
        for feature in df_column[5:]:
            df_file_l.plot(kind='line',x='frame_number',y=feature,color=tuple(reversed(np.divide(colors[feature_index],255))),ax=ax_left)
            feature_index = feature_index + 1
        # edit titles and axis
        plt.title(file+': Normalized features & labels vs frame_number & elapsed_time')
        plt.xlabel("")
        plt.ylabel('scores & labels - left eye')
        ax_left.set_xlim([df_file_l.loc[:,'frame_number'].iloc[0],df_file_l.loc[:,'frame_number'].iloc[-1]])
        ax_left.set_ylim([0,1])
        
        # plot graph for right eye
        ax_right = plt.subplot(2,1,2)
        # plot features and labels
        feature_index = 0
        for feature in df_column[5:]:
            df_file_r.plot(kind='line',x='frame_number',y=feature,color=tuple(reversed(np.divide(colors[feature_index],255))),ax=ax_right)
            feature_index = feature_index + 1
        # edit titles and axis
        plt.title(file+': Normalized features & labels vs frame_number & elapsed_time')
        plt.xlabel("")
        plt.ylabel('scores & labels - right eye')
        ax_right.set_xlim([df_file_r.loc[:,'frame_number'].iloc[0],df_file_r.loc[:,'frame_number'].iloc[-1]])
        ax_right.set_ylim([0,1])
        
        # set second x-axis (elapsed_time)
        ax2 = ax_right.twiny()
        # set the ticklabel position in the second x-axis, then convert them to the position in the first x-axis
        ax2_ticks_num = 6
        newlabel = [round((x*df_file_r.loc[:,'elapsed_time'].iloc[-1]/(ax2_ticks_num-1)),PREC_DIGITS) for x in range(0, ax2_ticks_num)]
        try:
            video_fps = int(file[11:13])
        except:
            video_fps = int(file[12:14])
        newpos = [int(np.ceil(x*video_fps)) for x in newlabel]
        # set the second x-axis
        ax2.set_xticks(newpos)
        ax2.set_xticklabels(newlabel)
        ax2.xaxis.set_ticks_position('bottom') 
        ax2.xaxis.set_label_position('bottom')
        ax2.spines['bottom'].set_position(('outward', 36))
        ax2.set_xlabel('elapsed_time [Sec]')
        
        # save plot to file
        plt.savefig("files/" + file +"_scores_labels_graphs.png",bbox_inches='tight',dpi=300)

# === BUILDING REDUCED DATABASE - EXTRACTING ONLY THE BLINKS ===
print('Extracting blinks from data...')
global_counter = [0] * (levels - 1)

NUM_REP = 4
blink_frames = []
num_rows = len(df_db.index)

blink_number = 0

for file in file_names_final:
    # initiate a counter of levels
    file_counter = [0] * (levels - 1)
    
    #  +++ find the indices of the detected blinks - LEFT EYE +++
    blink_frames_video = df_db.loc[(df_db['label'] < 1) & (df_db['video_name'] == file) & (df_db['eye_location'] == 'left')].index
    
    # Find the start and the end of the current file
    video_frames = df_db.loc[(df_db['video_name'] == file) & (df_db['eye_location'] == 'left')].index
    first_frame = video_frames[0]
    last_frame = video_frames[-1]
    
    i = 0
    blink_start = 0
    blink_end = 0
    max_counter = 0
    
    for b in blink_frames_video:
        # generate the label in terms of [0, levels - 1]
        label = int(df_db.loc[b,'label'] * (levels - 1))
        
        # check if the previous frame was the last frame describing the current blink
        if ((i != 0) and (b != blink_frames_video[i-1] + 1)) or i == len(blink_frames_video) - 1:
            
            # Update blink_end
            blink_end = blink_frames_video[i-1]
            
            # Update blink_start
            if blink_start == 0:
                blink_start = blink_frames_video[0]

            # find the maximum counter and add frames of label 1, before and after
            if max_counter/2 > (blink_start - first_frame): 
                blink_frames = blink_frames + list(range(first_frame,blink_start))
                df_db.loc[first_frame:blink_start, 'blink_number'] = blink_number                
                blink_frames = blink_frames + list(range(blink_end + 1, blink_end + 1 + (max_counter - (blink_start - first_frame))))
                df_db.loc[blink_end + 1:blink_end + 1 + (max_counter - (blink_start - first_frame)), 'blink_number'] = blink_number
                
            elif blink_end + max_counter/2 > last_frame:
                blink_frames = blink_frames + list(range(blink_end + 1, last_frame + 1))
                df_db.loc[blink_end + 1 : last_frame + 1, 'blink_number'] = blink_number
                blink_frames = blink_frames + list(range(blink_start - (max_counter - (last_frame - blink_end)), blink_start))
                df_db.loc[blink_start - (max_counter - (last_frame - blink_end)) : blink_start, 'blink_number'] = blink_number
            
            else:        
                blink_frames = blink_frames + list(range(blink_start - int(np.ceil(max_counter/2)), blink_start))
                df_db.loc[blink_start - int(np.ceil(max_counter/2)) : blink_start, 'blink_number'] = blink_number
                blink_frames = blink_frames + list(range(blink_end + 1, blink_end + int(np.ceil(max_counter/2)) + 1))
                df_db.loc[blink_end + 1 : blink_end + int(np.ceil(max_counter/2)) + 1, 'blink_number'] = blink_number
            
            # Initialize max_counter
            max_counter = 1
            
            # Add to blink number column
            df_db.loc[b,'blink_number'] = blink_number
            
            # Increase blink number
            if i != len(blink_frames_video) - 1:
                blink_number = blink_number + 1
            
            # Update blink_start
            blink_start = b
            
            # Add this frame to the reduced database
            blink_frames.append(b)
            
            # Set counters to zero and increase the appropriate counter
            file_counter = [0] * (levels - 1)
            file_counter[label] = file_counter[label] + 1
            global_counter[label] = global_counter[label] + 1
            
        
        else:
            if i != 0:
                prev_label = int(df_db.loc[blink_frames_video[i-1],'label'] * (levels - 1))
                # Check if we need to initialize the counter of the previous label
                if (prev_label != label):
                    file_counter[prev_label] = 0
            
            # Check if we need to add this frame to the database
            if file_counter[label] != NUM_REP:
                # Add this frame to the reduced database
                blink_frames.append(b)
                # Add to blink number column
                df_db.loc[b,'blink_number'] = blink_number
                
                # Update counters
                file_counter[label] = file_counter[label] + 1
                global_counter[label] = global_counter[label] + 1
                
                # Calculate max_counter
                if max_counter < file_counter[label]:
                    max_counter = file_counter[label]
                
   
            # If we reached the end of the video, add frames of label 1
            if b == num_rows - 1:
                max_counter = max(file_counter)
                blink_frames = blink_frames + list(range(b - max_counter, b))
        
        i = i + 1
    
    blink_number = blink_number + 1
    
    #  +++ find the indices of the detected blinks - RIGHT EYE +++
    blink_frames_video = df_db.loc[(df_db['label'] < 1) & (df_db['video_name'] == file) & (df_db['eye_location'] == 'right')].index
    
    # Find the start and the end of the current file
    video_frames = df_db.loc[(df_db['video_name'] == file) & (df_db['eye_location'] == 'right')].index
    first_frame = video_frames[0]
    last_frame = video_frames[-1]
    
    i = 0
    blink_start = 0
    blink_end = 0
    max_counter = 0
    
    for b in blink_frames_video:
        # generate the label in terms of [0, levels - 1]
        label = int(df_db.loc[b,'label'] * (levels - 1))
        
        # check if the previous frame was the last frame describing the current blink
        if ((i != 0) and (b != blink_frames_video[i-1] + 1)) or i == len(blink_frames_video) - 1:
            # Update blink_end
            blink_end = blink_frames_video[i-1]
            
            # Update blink_start
            if blink_start == 0:
                blink_start = blink_frames_video[0]
            
            # find the maximum counter and add frames of label 1, before and after
            if max_counter/2 > (blink_start - first_frame): 
                blink_frames = blink_frames + list(range(first_frame,blink_start))
                df_db.loc[first_frame:blink_start, 'blink_number'] = blink_number                
                blink_frames = blink_frames + list(range(blink_end + 1, blink_end + 1 + (max_counter - (blink_start - first_frame))))
                df_db.loc[blink_end + 1:blink_end + 1 + (max_counter - (blink_start - first_frame)), 'blink_number'] = blink_number
                
            elif blink_end + max_counter/2 > last_frame:
                blink_frames = blink_frames + list(range(blink_end + 1, last_frame + 1))
                df_db.loc[blink_end + 1 : last_frame + 1, 'blink_number'] = blink_number
                blink_frames = blink_frames + list(range(blink_start - (max_counter - (last_frame - blink_end)), blink_start))
                df_db.loc[blink_start - (max_counter - (last_frame - blink_end)) : blink_start, 'blink_number'] = blink_number
            
            else:        
                blink_frames = blink_frames + list(range(blink_start - int(np.ceil(max_counter/2)), blink_start))
                df_db.loc[blink_start - int(np.ceil(max_counter/2)) : blink_start, 'blink_number'] = blink_number
                blink_frames = blink_frames + list(range(blink_end + 1, blink_end + int(np.ceil(max_counter/2)) + 1))
                df_db.loc[blink_end + 1 : blink_end + int(np.ceil(max_counter/2)) + 1, 'blink_number'] = blink_number
            
            
            # Initialize max_counter
            max_counter = 1
            
            # Add to blink number column
            df_db.loc[b,'blink_number'] = blink_number
            
            # Increase blink number
            if i != len(blink_frames_video) - 1:
                blink_number = blink_number + 1
            
            # Update blink_start
            blink_start = b
            
            # Add this frame to the reduced database
            blink_frames.append(b)
            
            # Set counters to zero and increase the appropriate counter
            file_counter = [0] * (levels - 1)
            file_counter[label] = file_counter[label] + 1
            global_counter[label] = global_counter[label] + 1
        
        else:
            if i != 0:
                prev_label = int(df_db.loc[blink_frames_video[i-1],'label'] * (levels - 1))
                # Check if we need to initialize the counter of the previous label
                if (prev_label != label):
                    file_counter[prev_label] = 0
            
            # Check if we need to add this frame to the database
            if file_counter[label] != NUM_REP:
                # Add this frame to the reduced database
                blink_frames.append(b)
                # Add to blink number column
                df_db.loc[b,'blink_number'] = blink_number
                
                # Update counters
                file_counter[label] = file_counter[label] + 1
                global_counter[label] = global_counter[label] + 1
                
                # Calculate max_counter
                if max_counter < file_counter[label]:
                    max_counter = file_counter[label]
                
   
            # If we reached the end of the video, add frames of label 1
            if b == num_rows - 1:
                max_counter = max(file_counter)
                blink_frames = blink_frames + list(range(b - max_counter, b))
        
        i = i + 1
    
    blink_number = blink_number + 1
            
# Add to the reduced database
df_db_reduced = df_db.iloc[sorted(blink_frames),:]
# Remove duplicate values
df_db_reduced = df_db_reduced[~df_db_reduced.index.duplicated(keep='first')]
writer = pd.ExcelWriter("files/database_reduced.xlsx")
df_db_reduced.to_excel(writer,sheet_name='database')
writer.save()
writer.close()

# Append the number of 1-tagged frames into the counter
global_counter.append(len(df_db_reduced) - sum(global_counter))

# === REMOVE MORE SAMPLES TO END UP WITH A UNIFORMLY DISTRIBUTED DATABASE ===
print('Final improvements...')

# Reset df_db_reduced indices
df_db_final = df_db_reduced.reset_index()

final_counter = []
final_counter_normal = []
final_counter_palsy = []

# +++ START WITH HEALTHY PATIENTS
df_db_normal = df_db_reduced[df_db_reduced['video_name'].str.startswith('normal')]

# Mark the last blink that indicates a normal blink
last_normal_index = int(max(df_db_normal['blink_number']))

# Count the number of appearances for each level
for level in range(levels):
    final_counter_normal.append(len(df_db_normal[df_db_normal['label'] == level / (levels - 1)].index))
    
# Find the level that has minimum occurances
min_level = min(final_counter_normal)

for level in range(levels):
    
    flag = False
    
    # If the current level is the minimum one - do nothing
    if final_counter_normal[level] == min_level:
        continue
    
    while final_counter_normal[level] > min_level: 
        # Calculate how much samples to omit from each blink
        num_omit = max(1, (final_counter_normal[level] - min_level) // (last_normal_index + 1))
        omit_idx = [] 
 
        # Omit samples from each blink, if possible
        for blink in range(last_normal_index + 1):
            omit_blink_idx = []
            omit_blink_cnt = 0
            
            lvl_idx = df_db_final.loc[(df_db_final['blink_number'] == blink) & (df_db_final['label'] == level / (levels - 1))].index
            for i in range(1,len(lvl_idx)):
                if lvl_idx[i] == lvl_idx[i-1] + 1 or flag:
                    omit_blink_idx.append(lvl_idx[i])
                    omit_blink_cnt = omit_blink_cnt + 1
                if omit_blink_cnt == num_omit:
                    break

            omit_idx = omit_idx + omit_blink_idx
            final_counter_normal[level] = final_counter_normal[level] - len(omit_blink_idx)
            
            if final_counter_normal[level] <= min_level:
                break
    
        if len(omit_idx) == 0:
            flag = True

        # Drop the relevant inidices
        df_db_final = df_db_final.drop(omit_idx)
        
# +++ CONTINUE WITH PALSY PATIENTS
df_db_palsy = df_db_reduced[df_db_reduced['video_name'].str.startswith('palsy')]

# Count the number of appearances for each level
for level in range(levels):
    final_counter_palsy.append(len(df_db_palsy[df_db_palsy['label'] == level / (levels - 1)].index))
    
# Find the level that has minimum occurances
min_level = min(final_counter_palsy)

for level in range(levels):
    
    flag = False
    
    # If the current level is the minimum one - do nothing
    if final_counter_palsy[level] == min_level:
        continue
    
    while final_counter_palsy[level] > min_level: 
        # Calculate how much samples to omit from each blink
        num_omit = max(1, (final_counter_palsy[level] - min_level) // (blink_number - last_normal_index - 1))
        omit_idx = [] 
 
        # Omit samples from each blink, if possible
        for blink in range(last_normal_index + 1, blink_number):
            omit_blink_idx = []
            omit_blink_cnt = 0
            
            lvl_idx = df_db_final.loc[(df_db_final['blink_number'] == blink) & (df_db_final['label'] == level / (levels - 1))].index
            for i in range(1,len(lvl_idx)):
                if lvl_idx[i] == lvl_idx[i-1] + 1 or flag:
                    omit_blink_idx.append(lvl_idx[i])
                    omit_blink_cnt = omit_blink_cnt + 1
                if omit_blink_cnt == num_omit:
                    break

            omit_idx = omit_idx + omit_blink_idx
            final_counter_palsy[level] = final_counter_palsy[level] - len(omit_blink_idx)
            
            if final_counter_palsy[level] <= min_level:
                break
    
        if len(omit_idx) == 0:
            flag = True

        # Drop the relevant inidices
        df_db_final = df_db_final.drop(omit_idx)


# Calculate the final counter
final_counter = [x + y for x, y in zip(final_counter_normal, final_counter_palsy)]

final_counter_left = []
final_counter_right = []

# Count the number of appearances for each level - LEFT & RIGHT EYES
df_db_left = df_db_final[df_db_final['eye_location'] == 'left']
df_db_right = df_db_final[df_db_final['eye_location'] == 'right']

for level in range(levels):
    final_counter_left.append(len(df_db_left[df_db_left['label'] == level / (levels - 1)].index))
    final_counter_right.append(len(df_db_right[df_db_right['label'] == level / (levels - 1)].index))

# Save the final database
print('=== DATABASE STATISTICS: ===')
print('Total database size: ', len(df_db_final.index))
print('Number of levels for labeling: ', levels)
print('Label distribution is: ', final_counter)
print('For normal patients: ', final_counter_normal)
print('For palsy patients: ', final_counter_palsy)
print('For left eyes: ', final_counter_left)
print('For right eyes: ', final_counter_right)
print('Total number of blinks: ', df_db_final['blink_number'].nunique())

df_db_male = df_db_final.loc[df_db_final['gender'] == 'M']
df_db_female = df_db_final.loc[df_db_final['gender'] == 'F']

df_db_male_normal = df_db_male[df_db_male['video_name'].str.startswith('normal')].reset_index()
df_db_female_normal = df_db_female[df_db_female['video_name'].str.startswith('normal')].reset_index()
df_db_male_palsy = df_db_male[df_db_male['video_name'].str.startswith('palsy')].reset_index()
df_db_female_palsy = df_db_female[df_db_female['video_name'].str.startswith('palsy')].reset_index()

num_blinks_normal_male = df_db_male_normal['blink_number'].nunique()
num_blinks_normal_female = df_db_female_normal['blink_number'].nunique()
num_blinks_palsy_male = df_db_male_palsy['blink_number'].nunique()
num_blinks_palsy_female = df_db_female_palsy['blink_number'].nunique()

print('Normal: ' + str(num_blinks_normal_male) + ' for males, ' + str(num_blinks_normal_female) + ' for females')
print('Palsy: ' + str(num_blinks_palsy_male) + ' for males, ' + str(num_blinks_palsy_female) + ' for females')


print('============ DONE!============')

df_db_final = df_db_final.reset_index().drop(['level_0', 'index'],axis=1)
writer = pd.ExcelWriter("files/database_final.xlsx")
df_db_final.to_excel(writer,sheet_name='database')
writer.save()
writer.close()       