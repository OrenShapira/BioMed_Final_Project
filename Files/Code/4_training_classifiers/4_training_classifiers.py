
# coding: utf-8

# # Training Classifiers (Section 4)

# ## Import directories

# In[1]:


# import OpenCV path
arg_user = 'OREN'
import sys
if (arg_user == 'OREN'):
    sys.path.append('D:\Softwares\Anaconda3\envs\opencv-envpython=3.6\Lib')
elif(arg_user == 'TOM'):
    sys.path.append('C:/Users/user/AppData/Local/conda/conda/envs/opencv-env/Lib/site-packages')


# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import cv2
#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load database

# In[3]:


df_db_xlsx = pd.ExcelFile("files/database_reduced.xlsx")
df_db = pd.read_excel(df_db_xlsx,'database')


# In[4]:


levels = 5
features = ['ear','poly']
NUM_FRAMES = 6


# In[5]:


# Extract number of blinks
num_blinks = df_db['blink_number'].nunique()


# ## Add scores from neighboring frames

# In[6]:


df_db_offset = df_db.copy()

for offset in range(NUM_FRAMES*(-1), NUM_FRAMES+1):
    # Do something only if the offset isn't zero
    if(offset == 0):
        continue
    
    # Run for each blink
    for blink_number in range(num_blinks):
        
        blink_idx = df_db.loc[df_db['blink_number'] == blink_number].index

        for i in range(len(blink_idx)):
            # Add a column for each selected feature
            for feature in features:
                new_column = feature + '_' + str(offset)
                if (i + offset < 0) or (i + offset > len(blink_idx) - 1):
                    df_db_offset.loc[blink_idx[i], new_column] = 1
                else:
                    df_db_offset.loc[blink_idx[i], new_column]  = df_db_offset.loc[blink_idx[i + offset], feature]


# In[7]:


df_db_offset = df_db_offset.reset_index(drop=True)


# ## Create X and y vectors

# In[8]:


# Add the new columns to the feature list
new_features = []

for feature in features:
    add_cols   = [col for col in df_db_offset.columns if col.startswith(feature)]
    new_features = new_features + add_cols
new_features.append('palsy_eye')
print(new_features)


# In[9]:


X = df_db_offset[new_features]
y = np.array([int(l*4) for l in df_db_offset['label']])


# ## Examine the use of PCA

# In[10]:


#Fitting the PCA algorithm with our Data
pca = PCA().fit(X)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Dataset Explained Variance')
plt.show()


# In[11]:


# Pefrorm PCA using a lower number support vectors
model = PCA(n_components=17)            
model.fit(X)                      
X_PCA = model.transform(X)         


# ## Define function for cross-validation - split using blink number

# In[12]:


def cv_blink(df_db_offset, X, y, model, num_cv=5, random_state=0, use_prob=False):
    
    # Define constants
    num_blinks = 436
    num_blinks_normal = 214
    num_blinks_palsy = num_blinks - num_blinks_normal
    
    # Assemble an array saving the blink numbers that are not selected yet
    normal_choice = range(num_blinks_normal)
    palsy_choice  = range(num_blinks_normal, num_blinks)
    
    # Start cross validation
    for i in range(num_cv):
        
        # Randomize blink numbers for the current test set (size around num_blinks / num_cv)
        # Consider the rest as train set
        test_mask = np.zeros(len(X), dtype=bool)
        np.random.seed(random_state); rnd_idx_normal = sorted(np.random.choice(normal_choice, min(len(normal_choice),int(np.round(num_blinks_normal/num_cv))), replace=False))
        np.random.seed(random_state); rnd_idx_palsy = sorted(np.random.choice(palsy_choice, min(len(palsy_choice),int(np.round(num_blinks_palsy/num_cv))), replace=False))
        test_mask = np.zeros(len(X), dtype=bool)
        test_mask[df_db_offset.loc[df_db_offset['blink_number'].isin(rnd_idx_normal)].index] = True
        test_mask[df_db_offset.loc[df_db_offset['blink_number'].isin(rnd_idx_palsy)].index] = True
        Xtest, ytest = X[test_mask], y[test_mask]
        Xtrain, ytrain = X[~test_mask], y[~test_mask]

        # Apply the classifier - train on the train set, then evaluate using the test set
        model.fit(Xtrain, ytrain)                  
        y_model = model.predict(Xtest)             
        
        df_db_test_cv = df_db_offset.iloc[Xtest.index,:][['blink_number','video_name','frame_number','eye_location','palsy_eye','label']]
        df_db_test_cv.loc[:,'prediction'] = y_model / (levels - 1)

        # Add probability estimation info
        if use_prob:
            pred_pro = model.predict_proba(Xtest)
            for j in range(levels):
                col_name = 'prob_' + str(j)
                df_db_test_cv.loc[:,col_name] = pred_pro[:,j] 

        # Update test dataframe
        if i==0:
            df_db_test = df_db_test_cv.copy()
        else:
            df_db_test = pd.concat([df_db_test, df_db_test_cv])

        # Remove the selected blink numbers from the normal and palsy arrays
        normal_choice = [num for num in normal_choice if not (num in rnd_idx_normal)]
        palsy_choice = [num for num in palsy_choice if not (num in rnd_idx_palsy)]

    return df_db_test


# ## Evaluate performance using only the MINIMUM values from each blink

# In[13]:


def eval_blinks(df_db_test):
    
    df_db_test = df_db_test.reset_index(drop=True)
    y_blink_test = []
    y_blink_pred = []

    df_db_blink_res = df_db_test.loc[df_db_test['palsy_eye'] == 1, ['blink_number', 'video_name', 'eye_location']].drop_duplicates().reset_index(drop=True)
    
    for blink in df_db_test['blink_number'].unique():
        palsy_eye = min(df_db_test.loc[df_db_test['blink_number'] == blink]['palsy_eye'])
        if palsy_eye == 1:
            min_label = min(df_db_test.loc[df_db_test['blink_number'] == blink].iloc[1:-1,:]['label'])
            df_db_blink_res.loc[df_db_blink_res['blink_number'] == blink, 'min_label'] = min_label
            #y_blink_test.append(int(min_label * (levels - 1)))
            
            min_pred  = min(df_db_test.loc[df_db_test['blink_number'] == blink].iloc[1:-1,:]['prediction'])
            min_pred_idx = df_db_test.loc[(df_db_test['blink_number'] == blink) & (df_db_test['prediction'] == min_pred)].index
            min_pred_mid = min_pred_idx[min(len(min_pred_idx) - 1, round(len(min_pred_idx)/2))]
            min_locs = df_db_test.loc[df_db_test['blink_number'] == blink].loc[min_pred_mid-1:min_pred_mid+1,:]['prediction'].reset_index(drop=True)
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
            else:
                min_pred_final = np.median(min_locs)

            df_db_blink_res.loc[df_db_blink_res['blink_number'] == blink, 'min_pred'] = min_pred_final
            #y_blink_pred.append(int(min_pred_final * (levels - 1))

    return df_db_blink_res


# ## Try Naive Bayes

# In[14]:


nb = GaussianNB()                       # 2. instantiate model
df_db_test = cv_blink(df_db_offset, X, y, nb, num_cv=5, random_state=0, use_prob=False)


# In[15]:


# Evaluate score per frame
y_test = np.array([int(l*(levels-1)) for l in df_db_test['label']])
y_pred = np.array([int(l*(levels-1)) for l in df_db_test['prediction']])

mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score using Naive Bayes: ',accuracy_score(y_test, y_pred))


# In[16]:


# Evaluate score per blink
df_db_blink_res = eval_blinks(df_db_test)

y_blink_test = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_label']])
y_blink_pred = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_pred']])

mat = confusion_matrix(y_blink_test, y_blink_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score per blink using Naive Bayes: ',accuracy_score(y_blink_test, y_blink_pred))


# ## Try Support Vector Machine

# ### Linear SVM

# In[ ]:


linsvm = SVC(kernel='linear', C=1000)
df_db_test = cv_blink(df_db_offset, X, y, linsvm, num_cv=5, random_state=0, use_prob=False)


# In[19]:


# Evaluate score per frame
y_test = np.array([int(l*(levels-1)) for l in df_db_test['label']])
y_pred = np.array([int(l*(levels-1)) for l in df_db_test['prediction']])

mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score using Linear SVM: ',accuracy_score(y_test, y_pred))


# In[20]:


# Evaluate score per blink
df_db_blink_res = eval_blinks(df_db_test)

y_blink_test = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_label']])
y_blink_pred = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_pred']])

mat = confusion_matrix(y_blink_test, y_blink_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score per blink using Linear SVM: ',accuracy_score(y_blink_test, y_blink_pred))


# ### RBF SVM

# In[21]:


rbfsvm = SVC(kernel='rbf', C=1000, gamma='scale', probability=False, random_state=0)
df_db_test = cv_blink(df_db_offset, X, y, rbfsvm, num_cv=5, random_state=0, use_prob=False)


# In[ ]:


# Evaluate score per frame
y_test = np.array([int(l*(levels-1)) for l in df_db_test['label']])
y_pred = np.array([int(l*(levels-1)) for l in df_db_test['prediction']])

mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score using RBF SVM: ',accuracy_score(y_test, y_pred))


# In[ ]:


# Evaluate score per blink
df_db_blink_res = eval_blinks(df_db_test)

y_blink_test = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_label']])
y_blink_pred = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_pred']])

mat = confusion_matrix(y_blink_test, y_blink_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score per blink using RBF SVM: ',accuracy_score(y_blink_test, y_blink_pred))


# ### Use PCA to examine the best classifier

# In[22]:


rbfsvm = SVC(kernel='rbf', C=1000, gamma='scale', probability=False, random_state=0)
df_db_test = cv_blink(df_db_offset, pd.DataFrame(data=X_PCA), y, rbfsvm, num_cv=5, random_state=0, use_prob=False)

# Evaluate score per frame
y_test = np.array([int(l*(levels-1)) for l in df_db_test['label']])
y_pred = np.array([int(l*(levels-1)) for l in df_db_test['prediction']])

mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score using RBF SVM: ',accuracy_score(y_test, y_pred))


# In[23]:


# Evaluate score per blink
df_db_blink_res = eval_blinks(df_db_test)

y_blink_test = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_label']])
y_blink_pred = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_pred']])

mat = confusion_matrix(y_blink_test, y_blink_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score per blink using RBF SVM: ',accuracy_score(y_blink_test, y_blink_pred))


# ### Evaluate performace using only 3 levels

# In[24]:


y_test_3l = y_test
y_pred_3l = y_pred

for i in range(len(y_test)):
    if (y_test[i] == 1) or (y_test[i] == 3):
        y_test_3l[i] = 2
    if (y_pred[i] == 1) or (y_pred[i] == 3):
        y_pred_3l[i] = 2

mat = confusion_matrix(y_test_3l, y_pred_3l)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score using RBF SVM for 3-level classification: ',accuracy_score(y_test_3l, y_pred_3l))


# In[25]:


# Evaluate score per blink using only 3 levels
df_db_test_3l = df_db_test.copy()
df_db_test_3l.loc[:,'label'] = y_test_3l / (levels-1)
df_db_test_3l.loc[:,'prediction'] = y_pred_3l / (levels-1)
df_db_blink_res_3l = eval_blinks(df_db_test_3l)

y_blink_test_3l = np.array([int(l*(levels-1)) for l in df_db_blink_res_3l['min_label']])
y_blink_pred_3l = np.array([int(l*(levels-1)) for l in df_db_blink_res_3l['min_pred']])

mat = confusion_matrix(y_blink_test_3l, y_blink_pred_3l)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score per blink using RBF SVM for 3-level classification: ',accuracy_score(y_blink_test_3l, y_blink_pred_3l))


# ## Save classifier to a variable

# In[26]:


from joblib import dump, load

rbfsvm = SVC(kernel='rbf', C=1000, gamma='scale', probability=False, random_state=0)
rbfsvm.fit(X, y)

dump(rbfsvm, 'model.joblib')


# ## Try K-Nearest Neighbors

# In[12]:


neigh = np.arange(start=1, stop=41, step=2)
scores = []

for n in neigh:
    knn = KNeighborsClassifier(n_neighbors=n, weights='distance')
    df_db_test = cv_blink(df_db_offset, X, y, knn, num_cv=5, random_state=0, use_prob=False)
    df_db_blink_res = eval_blinks(df_db_test)
    y_blink_test = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_label']])
    y_blink_pred = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_pred']])
    score = accuracy_score(y_blink_test, y_blink_pred)
    print('Accuracy score using ' + str(n) + 'neighbors: ' + str(score))
    scores.append(accuracy_score(y_blink_test, y_blink_pred))

plt.plot(neigh,scores)
plt.xlabel('Number of neighbors')
plt.ylabel('KNN Score')

max_value = max(scores)
max_index = scores.index(max_value)
print('Max score is: ', max_value)
print('Using ' + str(neigh[max_index]) + ' neighbors')


# In[14]:


# Display results using KNN with the optimal number of neighbors
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
df_db_test = cv_blink(df_db_offset, X, y, knn, num_cv=5, random_state=0, use_prob=False)
df_db_blink_res = eval_blinks(df_db_test)

y_test = np.array([int(l*(levels-1)) for l in df_db_test['label']])
y_pred = np.array([int(l*(levels-1)) for l in df_db_test['prediction']])

mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score using K-Nearest Neighbors: ',accuracy_score(y_test, y_pred))


# In[15]:


y_blink_test = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_label']])
y_blink_pred = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_pred']])

mat = confusion_matrix(y_blink_test, y_blink_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score per blink using K-Nearest Neighbors: ',accuracy_score(y_blink_test, y_blink_pred))


# ## Try Random Forest

# In[33]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

parameters = {'max_features':['sqrt', 'log2', 10],
              'max_depth':[5, 7, 9]}

rf = RandomForestClassifier(n_estimators=500, random_state=0)

df_db_test = cv_blink(df_db_offset, X, y, rf, num_cv=5, random_state=0, use_prob=False)


# In[34]:


# Evaluate score per frame
y_test = np.array([int(l*(levels-1)) for l in df_db_test['label']])
y_pred = np.array([int(l*(levels-1)) for l in df_db_test['prediction']])

mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score using Random Forest: ',accuracy_score(y_test, y_pred))


# In[35]:


# Evaluate score per blink
df_db_blink_res = eval_blinks(df_db_test)

y_blink_test = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_label']])
y_blink_pred = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_pred']])

mat = confusion_matrix(y_blink_test, y_blink_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score per blink using Random Forest: ',accuracy_score(y_blink_test, y_blink_pred))


# In[37]:


from joblib import dump, load

rf = RandomForestClassifier(n_estimators=500, random_state=0)
rf.fit(X, y)

dump(rf, 'forest.joblib')


# # Choose the optimal number of neighboring frames to take into account

# ## Show historgam of blink durations

# In[299]:


dur_list = []

for blink in df_db['blink_number'].unique():
    dur_list.append(len(df_db.loc[df_db['blink_number'] == blink].index))

plt.hist(dur_list)
plt.ylabel('Occurances')
plt.xlabel('Blink duration [frames]')
plt.title('Histogram of blink duration')


# In[300]:


max_frames = 10

df_db_ext = df_db.copy()

for offset in range(max_frames*(-1), max_frames+1):
    # Do something only if the offset isn't zero
    if(offset == 0):
        continue
    
    # Run for each blink
    for blink_number in range(num_blinks):
        
        blink_idx = df_db.loc[df_db['blink_number'] == blink_number].index

        for i in range(len(blink_idx)):
            # Add a column for each selected feature
            for feature in features:
                new_column = feature + '_' + str(offset)
                if (i + offset < 0) or (i + offset > len(blink_idx) - 1):
                    df_db_ext.loc[blink_idx[i], new_column] = 1
                else:
                    df_db_ext.loc[blink_idx[i], new_column]  = df_db_ext.loc[blink_idx[i + offset], feature]


# In[306]:


# Run for each offset, starting from zero
df_db_ext = df_db_ext.reset_index(drop=True)
offset_scores = []

for n in range(max_frames+1):
    
    # Construct a list of features to consider
    new_features = features
    for feature in features:
        add_cols = []
        for m in range(1,n+1):
            add_cols   = add_cols + [col for col in df_db_ext.columns if (col.startswith(feature) and col.endswith(str(m)))]
        new_features = new_features + add_cols
    new_features.append('palsy_eye')
    print(len(new_features))
    
    # Construct X and y vectors
    X = df_db_ext[new_features]
    y = np.array([int(l*4) for l in df_db_ext['label']])
    
    # Classify using RBF SVM
    rbfsvm = SVC(kernel='rbf', C=1000, gamma='scale', probability=False, random_state=0)
    df_db_test = cv_blink(df_db_ext, X, y, rbfsvm, num_cv=5, random_state=0, use_prob=False)
    
    # Evaluate score per blink
    df_db_blink_res = eval_blinks(df_db_test)

    y_blink_test = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_label']])
    y_blink_pred = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_pred']])
    
    # Append the accuracy score to the scores list
    offset_scores.append(accuracy_score(y_blink_test, y_blink_pred))


# In[307]:


# Show graphs
plt.plot(range(max_frames+1),offset_scores)
plt.xlabel('Number of neighboring frames')
plt.ylabel('Classification accuracy score')
plt.title('Model performance vs. number of neighboring frames')


# In[6]:


# Try assymetrical offset 
max_frames = 15

df_db_ext = df_db.copy()

for offset in range(max_frames*(-1), 0):
    # Do something only if the offset isn't zero
    if(offset == 0):
        continue
    
    # Run for each blink
    for blink_number in range(num_blinks):
        
        blink_idx = df_db.loc[df_db['blink_number'] == blink_number].index

        for i in range(len(blink_idx)):
            # Add a column for each selected feature
            for feature in features:
                new_column = feature + '_' + str(offset)
                if (i + offset < 0) or (i + offset > len(blink_idx) - 1):
                    df_db_ext.loc[blink_idx[i], new_column] = 1
                else:
                    df_db_ext.loc[blink_idx[i], new_column]  = df_db_ext.loc[blink_idx[i + offset], feature]


# In[10]:


# Run for each offset, starting from zero
df_db_ext = df_db_ext.reset_index(drop=True)
offset_scores = []

for n in range(max_frames+1):
    
    # Construct a list of features to consider
    new_features = features
    for feature in features:
        add_cols = []
        for m in range(1,n+1):
            add_cols   = add_cols + [col for col in df_db_ext.columns if (col.startswith(feature) and col.endswith('-'+str(m)))]
        new_features = new_features + add_cols
    new_features.append('palsy_eye')
    print((new_features))
    
    # Construct X and y vectors
    X = df_db_ext[new_features]
    y = np.array([int(l*4) for l in df_db_ext['label']])
    
    # Classify using RBF SVM
    rbfsvm = SVC(kernel='rbf', C=1000, gamma='scale', probability=False, random_state=0)
    df_db_test = cv_blink(df_db_ext, X, y, rbfsvm, num_cv=5, random_state=0, use_prob=False)
    
    # Evaluate score per blink
    df_db_blink_res = eval_blinks(df_db_test)

    y_blink_test = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_label']])
    y_blink_pred = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_pred']])
    
    # Append the accuracy score to the scores list
    offset_scores.append(accuracy_score(y_blink_test, y_blink_pred))


# In[11]:


# Show graphs
plt.plot(range(max_frames+1),offset_scores)
plt.xlabel('Number of neighboring frames')
plt.ylabel('Classification accuracy score')
plt.title('Model performance vs. number of neighboring frames')


# # Different approach: using textual features

# ## Extract HOG features

# In[6]:


# define HOG descriptor
IMG_WIDTH = 34
IMG_HEIGHT = 26
BLOCK_SIZE = 3

# change window size so that it will be divisable by the block size
if IMG_WIDTH % BLOCK_SIZE == 0:
    win_width = IMG_WIDTH
else:
    win_width = (IMG_WIDTH // BLOCK_SIZE) * BLOCK_SIZE
    
if IMG_HEIGHT % BLOCK_SIZE == 0:
    win_height = IMG_HEIGHT
else:
    win_height = (IMG_HEIGHT // BLOCK_SIZE) * BLOCK_SIZE

# Define HOG parameters
winSize = (win_width, win_height)
blockSize = (2*BLOCK_SIZE, 2*BLOCK_SIZE)
blockStride = (BLOCK_SIZE, BLOCK_SIZE)
cellSize = (BLOCK_SIZE, BLOCK_SIZE)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)


# In[7]:


# Extract features for each frame in the database
df_db_hog = df_db.copy().reset_index(drop=True)

for i in range(len(df_db.index)):
    # extract file name
    file_name  = df_db_hog.loc[i,'video_name']
    # extract eye location
    eye_loc    = df_db_hog.loc[i,'eye_location']
    # extract frame number
    frame_num  = df_db_hog.loc[i,'frame_number']
    # extract image directory
    img_dir = '../2_features_extraction/files/' + file_name + '/eye_pics/' + eye_loc + '/' + str(frame_num).zfill(3) + '.jpg'
    # read the appropriate image
    image = cv2.imread(img_dir)
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # calculate HOG descriptor
    hist = hog.compute(gray)
    
    if i == 0:
        feature_matrix = np.transpose(hist)
    else:
        feature_matrix = np.vstack([feature_matrix, np.transpose(hist)])


# ## Define X and y vectors

# In[8]:


feature_matrix.shape[1]


# In[9]:


#X = feature_matrix
X = pd.DataFrame(data=feature_matrix,                  
                 index=range(feature_matrix.shape[0]), 
                 columns=range(feature_matrix.shape[1]))     
y = np.array([int(l*4) for l in df_db_hog['label']])


# ## Examine the use of PCA

# In[16]:


#Fitting the PCA algorithm with our Data
pca = PCA().fit(X)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Dataset Explained Variance')
plt.show()


# In[17]:


# Pefrorm PCA using a lower number support vectors
model = PCA(n_components=1600)            
model.fit(X)                      
X_PCA = model.transform(X)         


# ## Try SVM

# ### Linear SVM

# In[29]:


linsvm = SVC(kernel='linear', C=1000)
df_db_test = cv_blink(df_db_hog, X, y, linsvm, num_cv=5, random_state=0, use_prob=False)


# In[30]:


# Evaluate score per frame
y_test = np.array([int(l*(levels-1)) for l in df_db_test['label']])
y_pred = np.array([int(l*(levels-1)) for l in df_db_test['prediction']])

mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score using HOG & Linear SVM: ',accuracy_score(y_test, y_pred))


# In[31]:


# Evaluate score per blink
df_db_blink_res = eval_blinks(df_db_test)

y_blink_test = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_label']])
y_blink_pred = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_pred']])

mat = confusion_matrix(y_blink_test, y_blink_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score per blink using HOG & Linear SVM: ',accuracy_score(y_blink_test, y_blink_pred))


# ### RBF SVM

# In[37]:


rbfsvm = SVC(kernel='rbf', C=1000, gamma='scale', probability=False, random_state=0)
df_db_test = cv_blink(df_db_hog, X, y, rbfsvm, num_cv=5, random_state=0, use_prob=False)


# In[38]:


# Evaluate score per frame
y_test = np.array([int(l*(levels-1)) for l in df_db_test['label']])
y_pred = np.array([int(l*(levels-1)) for l in df_db_test['prediction']])

mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score using HOG & RBF SVM: ',accuracy_score(y_test, y_pred))


# In[39]:


# Evaluate score per blink
df_db_blink_res = eval_blinks(df_db_test)

y_blink_test = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_label']])
y_blink_pred = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_pred']])

mat = confusion_matrix(y_blink_test, y_blink_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score per blink using HOG & RBF SVM: ',accuracy_score(y_blink_test, y_blink_pred))


# ## Try K-Nearest Neighbors

# In[20]:


neigh = np.arange(start=1, stop=41, step=2)
scores = []

for n in neigh:
    knn = KNeighborsClassifier(n_neighbors=n, weights='distance')
    df_db_test = cv_blink(df_db_hog, X, y, knn, num_cv=5, random_state=0, use_prob=False)
    df_db_blink_res = eval_blinks(df_db_test)
    y_blink_test = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_label']])
    y_blink_pred = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_pred']])
    score = accuracy_score(y_blink_test, y_blink_pred)
    print('Accuracy score using ' + str(n) + 'neighbors: ' + str(score))
    scores.append(score)

plt.plot(neigh,scores)
plt.xlabel('Number of neighbors')
plt.ylabel('KNN Score')

max_value = max(scores)
max_index = scores.index(max_value)
print('Max score is: ', max_value)
print('Using ' + str(neigh[max_index]) + ' neighbors')


# In[21]:


# Display results using KNN with the optimal number of neighbors
knn = KNeighborsClassifier(n_neighbors=11, weights='distance')
df_db_test = cv_blink(df_db_hog, X, y, knn, num_cv=5, random_state=0, use_prob=False)
df_db_blink_res = eval_blinks(df_db_test)

y_test = np.array([int(l*(levels-1)) for l in df_db_test['label']])
y_pred = np.array([int(l*(levels-1)) for l in df_db_test['prediction']])

mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score using K-Nearest Neighbors: ',accuracy_score(y_test, y_pred))


# In[22]:


y_blink_test = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_label']])
y_blink_pred = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_pred']])

mat = confusion_matrix(y_blink_test, y_blink_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score per blink using K-Nearest Neighbors: ',accuracy_score(y_blink_test, y_blink_pred))


# ### Use PCA to examine the best classifier

# In[20]:


# Display results using KNN with the optimal number of neighbors
knn = KNeighborsClassifier(n_neighbors=11, weights='distance')
df_db_test = cv_blink(df_db_hog, pd.DataFrame(data=X_PCA), y, knn, num_cv=5, random_state=0, use_prob=False)
df_db_blink_res = eval_blinks(df_db_test)

y_test = np.array([int(l*(levels-1)) for l in df_db_test['label']])
y_pred = np.array([int(l*(levels-1)) for l in df_db_test['prediction']])

mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score using K-Nearest Neighbors: ',accuracy_score(y_test, y_pred))


# In[21]:


y_blink_test = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_label']])
y_blink_pred = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_pred']])

mat = confusion_matrix(y_blink_test, y_blink_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score per blink using K-Nearest Neighbors: ',accuracy_score(y_blink_test, y_blink_pred))


# ## Try Naive Bayes

# In[23]:


nb = GaussianNB()                       # 2. instantiate model
df_db_test = cv_blink(df_db_hog, X, y, nb, num_cv=5, random_state=0, use_prob=False)


# In[24]:


# Evaluate score per frame
y_test = np.array([int(l*(levels-1)) for l in df_db_test['label']])
y_pred = np.array([int(l*(levels-1)) for l in df_db_test['prediction']])

mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score using Naive Bayes: ',accuracy_score(y_test, y_pred))


# In[25]:


# Evaluate score per blink
df_db_blink_res = eval_blinks(df_db_test)

y_blink_test = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_label']])
y_blink_pred = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_pred']])

mat = confusion_matrix(y_blink_test, y_blink_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score per blink using Naive Bayes: ',accuracy_score(y_blink_test, y_blink_pred))


# ## Try Random Forest

# In[13]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

parameters = {'max_features':['sqrt', 'log2', 10],
              'max_depth':[5, 7, 9]}

rf = RandomForestClassifier(n_estimators=500, random_state=0)

df_db_test = cv_blink(df_db_hog, X, y, rf, num_cv=5, random_state=0, use_prob=False)


# In[14]:


# Evaluate score per frame
y_test = np.array([int(l*(levels-1)) for l in df_db_test['label']])
y_pred = np.array([int(l*(levels-1)) for l in df_db_test['prediction']])

mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score using Random Forest: ',accuracy_score(y_test, y_pred))


# In[15]:


# Evaluate score per blink
df_db_blink_res = eval_blinks(df_db_test)

y_blink_test = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_label']])
y_blink_pred = np.array([int(l*(levels-1)) for l in df_db_blink_res['min_pred']])

mat = confusion_matrix(y_blink_test, y_blink_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

print('Accuracy score per blink using Random Forest: ',accuracy_score(y_blink_test, y_blink_pred))

