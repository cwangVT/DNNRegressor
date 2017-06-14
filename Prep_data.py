import os
import sys
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import itertools
import math
import scipy.optimize
import matplotlib
matplotlib.use("Agg")				# to work without X-window
import matplotlib.pyplot as plt

# ==============================================================================
# ====================== Set Environment Parameters ============================
# ==============================================================================
input_fold = "input_data/"
output_fold = "output_data/"
path_to_save = "tmp/my_model.ckpt"
pre_post_process = True

# set the verbosity mode to show detailed info during training
#tf.logging.set_verbosity(tf.logging.INFO)

# ============================================================================
# ====================== Load Data ===========================================
# ============================================================================

# Set the number of features
N_Features = 5 

# load the train dataset from csv file
# here assuming that the csv file has head line
train_set = pd.read_csv(input_fold+"train_set.txt", skipinitialspace=True)
if train_set.shape[1]!= N_Features+1:
	sys.exit("Error: # of features in training set (%d) is not equal to N_Features (%d)" % (train_set.shape[1]-1, N_Features))
COLUMNS = train_set.columns.values.tolist()

# load the validation data from csv file
valid_set = pd.read_csv(input_fold+"valid_set.txt", skipinitialspace=True)	
if valid_set.shape[1]!= N_Features+1:
	sys.exit("Error: # of features in validating set (%d) is not equal to N_Features (%d)" % (valid_set.shape[1]-1, N_Features))
if valid_set.columns.values.tolist()!=COLUMNS:
	sys.exit("Error: name of features in validating set is different")
# load the test dataset from csv file.
test_set = pd.read_csv(input_fold+"test_set.txt", skipinitialspace=True)
if test_set.shape[1]!= N_Features+1:
	sys.exit("Error: # of features in testing set (%d) is not equal to N_Features (%d)" % (test_set.shape[1]-1, N_Features))
if test_set.columns.values.tolist()!=COLUMNS:
        sys.exit("Error: name of features in testing set is different")

# load the predict dataset from csv file.
pred_set = pd.read_csv(input_fold+"predict_set.txt", skipinitialspace=True)
if pred_set.shape[1]!= N_Features+1:
	sys.exit("Error: # of features in predicting set (%d) is not equal to N_Features (%d)" % (pred_set.shape[1]-1, N_Features))
if pred_set.columns.values.tolist()!=COLUMNS:
        sys.exit("Error: name of features in predicting set is different")

FEATURES = COLUMNS[:-1]
LABEL = [COLUMNS[-1]]

# get the size of each dataset
n_sample = train_set.shape[0]
n_test = test_set.shape[0]
n_valid = valid_set.shape[0]
n_pred = pred_set.shape[0]

# =============================================================
# ================ perform preprocess =========================
# =============================================================
# after processing: x_norm = [x - mean(x)]/[max(x)-min(x)]

if pre_post_process:
	means = train_set.mean().to_dict()
	maxx = train_set.max().to_dict()
	minn = train_set.min().to_dict()
	maxmin = {}
	for jj in COLUMNS:
		maxmin[jj] = maxx[jj]-minn[jj]
		if maxmin[jj] == 0:
			print( "======= Feature "+jj+" never change")

def pre_process(xx,index):
	if pre_post_process:
		xx-=means[index]
		if maxmin[index] != 0:
			xx/=maxmin[index]
	return xx

train_set =pd.DataFrame([[pre_process(train_set[kk][ii],kk) for kk in COLUMNS] for ii in range(n_sample)],columns =COLUMNS)
test_set =pd.DataFrame([[pre_process(test_set[kk][ii],kk) for kk in COLUMNS] for ii in range(n_test)],columns =COLUMNS)
pred_set =pd.DataFrame([[pre_process(pred_set[kk][ii],kk) for kk in COLUMNS] for ii in range(n_pred)],columns =COLUMNS)
valid_set =pd.DataFrame([[pre_process(valid_set[kk][ii],kk) for kk in COLUMNS] for ii in range(n_valid)],columns =COLUMNS)

# saving processing parameters
f_proc = open(output_fold+"preprocess.txt","w")
f_proc.write("feature,mean,maxmin\n")
for ii in COLUMNS:
	f_proc.write(ii+","+str(means[ii])+","+str(maxmin[ii])+"\n")
f_proc.close()

train_set.to_csv(input_fold+"train_set_preped.txt", index=False)
test_set.to_csv(input_fold+"test_set_preped.txt", index=False)
pred_set.to_csv(input_fold+"predict_set_preped.txt", index=False)
valid_set.to_csv(input_fold+"valid_set_preped.txt", index = False)

