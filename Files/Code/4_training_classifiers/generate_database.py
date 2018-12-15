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

print(file_names)

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
file_names_final = [file for file in file_names if not file.startswith(tuple(file_names_exclude))]

if exclude_all_normal:
    file_names_final = [file for file in file_names_final if not file.startswith('normal')]
if exclude_all_palsy:
    file_names_final = [file for file in file_names_final if not file.startswith('palsy')]
    
print('After exclusion:')
for file in file_names_final:
    print(file)
    
# Extract the weights to be used
weights = args["weights"]

# Extract the amount of levels for tagging
levels = args["levels"]

# Define df_column
df_column = ['video_name', 'frame_number', 'elapsed_time', 'eye_location']

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
    
    # Iterate on each frame
    for i in range(len(df_labels.index)):
        # +++ Start from LEFT eye +++
        df_row_left = [file]
        df_row_left.append(int(df_labels.loc[i,'frame_name'].split('_')[4][5:]))
        df_row_left.append(float(df_labels.loc[i,'elapsed_time']))
        df_row_left.append('left')
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
        df_row_right = [file]
        df_row_right.append(int(df_labels.loc[i,'frame_name'].split('_')[4][5:]))
        df_row_right.append(float(df_labels.loc[i,'elapsed_time']))
        df_row_right.append('right')
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
df_database = pd.read_csv(output, names=df_column)
writer = pd.ExcelWriter("files/database.xlsx")
df_database.to_excel(writer,sheet_name='database')

writer.save()
writer.close()

# === SAVE GRAPHS OF FEATURES VS. LABELS, FOR EACH VIDEO IN THE DATABASE ===
if args["graphs"]:
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
        for feature in df_column[4:]:
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
        for feature in df_column[4:]:
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
        video_fps = int(file[11:13])
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
            
