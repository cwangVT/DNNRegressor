from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
# set the verbosity mode to do not show warning info during training
tf.logging.set_verbosity(tf.logging.ERROR)

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

train_set.loc[:,:] =[[pre_process(train_set[kk][ii],kk) for kk in COLUMNS] for ii in range(n_sample)]
test_set.loc[:,:] =[[pre_process(test_set[kk][ii],kk) for kk in COLUMNS] for ii in range(n_test)]
pred_set.loc[:,:] =[[pre_process(pred_set[kk][ii],kk) for kk in COLUMNS] for ii in range(n_pred)]
valid_set.loc[:,:] =[[pre_process(valid_set[kk][ii],kk) for kk in COLUMNS] for ii in range(n_valid)]

# Separate feature X and label Y
X_train = train_set.loc[:,FEATURES]
Y_train = train_set.loc[:,LABEL]
X_test = test_set.loc[:,FEATURES]
Y_test = test_set.loc[:,LABEL]
X_valid = valid_set.loc[:,FEATURES]
Y_valid = valid_set.loc[:,LABEL]
X_pred = pred_set.loc[:,FEATURES]
Y_pred = pred_set.loc[:,LABEL]
# saving processing parameters
f_proc = open(output_fold+"preprocess.txt","w")
f_proc.write("feature,mean,maxmin\n")
for ii in COLUMNS:
	f_proc.write(ii+","+str(means[ii])+","+str(maxmin[ii])+"\n")
f_proc.close()

# =============================================================
# ============== Set Parameters for NN ========================
# =============================================================


# ============================================================================
# ====================== Create NN ===========================================
# ============================================================================

feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

# function to generate the feature data and label data from loaded dataset
def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
  labels = tf.constant(data_set[LABEL[0]].values)
  return feature_cols, labels

print(LABEL)
print(LABEL[0])


# ========================================================================
# =============== Run the DNN ============================================
# ========================================================================
config = tf.ConfigProto()
# Do not use GPU
config = tf.ConfigProto(device_count = {'GPU':0 })


# create validation_monitor
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
       eval_steps = 1,
       input_fn=lambda: input_fn(train_set),
       every_n_steps=500)

# create the DNN regressor with the feature columns above
# agian, only using features, no data yet
regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_cols,
        hidden_units=[50,50,50,50,50],
        optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.00001,
        l1_regularization_strength=0.0),
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1),
        model_dir="tmp/")

# train the DNN using trainset

current_step = 0
n_batch = 500

n_sample = train_set.shape[0]
list_sample = [i for i in range(n_sample)]
random.shuffle(list_sample)
batch_size = math.floor(n_sample/n_batch)
batch_step = 20000

regressor.fit(input_fn=lambda: input_fn(train_set,) ,
                monitors=[validation_monitor],
                steps= batch_step)

#======= predicting ======
predicted_result=regressor.predict(input_fn=lambda: input_fn(test_set))
predictions = list(itertools.islice(predicted_result, 0,None))
test_labels = list(test_set[LABEL[0]].values)
if len(test_labels)!=len(predictions):
        print(len(test_labels))
        print(len(test_labels))

cost = 0
nn = len(test_labels)
for ii in range(nn):
        tmp = (predictions[ii]-test_labels[ii])
        cost+= tmp*tmp
cost=cost/nn

#        cost = tf.reduce_mean(tf.square(predictions-test_labels))
print("==== cost ===", cost)
                            
print(predictions[0],test_labels[0])


# ==================================================================
# ====================== end of NN =================================
# ==================================================================
'''
# post-process function
def post_process(xx,index):
	if pre_post_process:
		return xx*maxmin[index]+means[index]
	else:
		return xx 
# print post processed data, check post-process function
#print([post_process(train_set_values[0][i],i) for i in range(N_Features+1)])

# save the testing data
Y_test = [[post_process(Y_test[ii],N_Features)] for ii in range(len(Y_test))]
test_vals = [[post_process(test_vals[ii],N_Features)] for ii in range(len(Y_test))]

ftmp= open(output_fold+"test_result.txt","w")
for ii in range(len(test_vals)):
	ftmp.write(str(test_vals[ii])+"\t"+str(Y_test[ii])+"\n")
ftmp.close()

# plotting predicted vs labeled for testing dataset
plt.figure(0)
plt.plot(Y_test[0:100], 'rs--', test_vals[0:100], 'b^-.')
plt.xlabel('samples')
plt.ylabel('value')
plt.title('predicted(blue) vs label(red)')
plt.grid(True)
plt.savefig(output_fold+"test_result.png")

# saving pred_result
Y_pred = [[post_process(Y_pred[ii],N_Features)] for ii in range(len(Y_pred))]
pred_vals = [[post_process(pred_vals[ii],N_Features)] for ii in range(len(pred_vals))]
plot_x = [[post_process(X_pred[ii],0)] for ii in range(len(X_pred))]

ftmp= open(output_fold+"pred_result.txt","w")
for ii in range(len(pred_vals)):
        ftmp.write(str(pred_vals[ii])+"\t"+str(Y_pred[ii])+"\n")
ftmp.close()
# plotting predicted vs labeled for testing dataset

plt.figure(1)
plt.plot(plot_x,Y_pred, 'r--', plot_x,pred_vals, 'b-.')
plt.xlabel('samples')
plt.ylabel('value')
plt.title('predicted(blue) vs label(red)')
plt.grid(True)
plt.savefig(output_fold+"pred_result.png")

'''
