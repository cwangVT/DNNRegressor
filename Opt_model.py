import os
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

# Set the number of features
N_Features = 16 

# Network Structure Parameters
n_hidden_1 = 50
n_hidden_2 = 50
n_hidden_3 = 50
n_hidden_4 = 50
n_hidden_5 = 50
n_input = N_Features		# number of input features
n_classes = 1			# for regressin, only one output class


# ============================================================================
# ====================== Load Data ===========================================
# ============================================================================

# loading processing parameters
f_proc = open(output_fold+"preprocess.txt","r")
f_proc.readline()	# skip the title line
COLUMNS = []
means = []
maxmin = []
for ff in f_proc:
	tt = ff.strip().split(",")
	COLUMNS.append(tt[0])
	means.append(float(tt[1]))
	maxmin.append(float(tt[2]))
f_proc.close()

# loading NN parameters
# Set variables: Weight and Bias
weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1],0, 0.1,dtype=tf.float64)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1,dtype=tf.float64)),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1,dtype=tf.float64)),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.1,dtype=tf.float64)),
        'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5], 0, 0.1,dtype=tf.float64)),
        'out': tf.Variable(tf.random_normal([n_hidden_5, n_classes], 0, 0.1,dtype=tf.float64))
}
biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1,dtype=tf.float64)),
        'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1,dtype=tf.float64)),
        'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1,dtype=tf.float64)),
        'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.1,dtype=tf.float64)),
        'b5': tf.Variable(tf.random_normal([n_hidden_5], 0, 0.1,dtype=tf.float64)),
        'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1,dtype=tf.float64))
}
saver = tf.train.Saver()
# load the DNN weights and biases
with tf.Session() as sess:
        saver.restore(sess, path_to_save)
        hh1 = weights['h1'].eval()
        hh2 = weights['h2'].eval()
        hh3 = weights['h3'].eval()
        hh4 = weights['h4'].eval()
        hh5 = weights['h5'].eval()
        hout= weights['out'].eval()
        bb1 = biases['b1'].eval()
        bb2 = biases['b2'].eval()
        bb3 = biases['b3'].eval()
        bb4 = biases['b4'].eval()
        bb5 = biases['b5'].eval()
        bout = biases['out'].eval()

# ======================================================================
# ======================= Helper Functions  ============================
# ======================================================================

def pre_process(xx,index):
	if pre_post_process:
		xx-=means[index]
		if maxmin[index] != 0:
			xx/=maxmin[index]
	return xx

# post-process function
def post_process(xx,index):
	if pre_post_process:
                return xx*maxmin[index]+means[index]
        else:
                return xx

# relu function
def np_relu(xx):
	return np.maximum(xx, 0, xx)
# =====================================================================
# ============= DNN Prediction Function, Target Function ===============
# =====================================================================
def nn_predict_as_np(xx):
	if True:
                pred_layer_1 = np_relu(np.add(np.matmul(xx, hh1), bb1))
                pred_layer_2 = np_relu(np.add(np.matmul(pred_layer_1, hh2), bb2))
                pred_layer_3 = np_relu(np.add(np.matmul(pred_layer_2, hh3), bb3))
                pred_layer_4 = np_relu(np.add(np.matmul(pred_layer_3, hh4), bb4))
                pred_layer_5 = np_relu(np.add(np.matmul(pred_layer_4, hh5), bb5))
                pred_out_layer = np.add(np.matmul(pred_layer_5, hout),bout)
                return pred_out_layer

# ======================================================================
# =================== Simulating Annealing =============================
# ======================================================================

# ================== Set Parameters for Optimizer ======================
initial_guess =[[0. for i in range(N_Features)]]
initial_guess = [[(initial_guess[0][ii]-means[ii])/maxmin[ii] for ii in range(N_Features)]]
#print(post_process(nn_predict_as_np(initial_guess),N_Features))

xmin = [-0.5 for i in range(N_Features)]
xmax = [0.5 for i in range(N_Features)]
bounds = [(low, high) for low, high in zip(xmin, xmax)]

# ================== try directly minimize ============================
minimal = scipy.optimize.minimize(nn_predict_as_np,
                initial_guess,method="L-BFGS-B", jac=False, bounds=bounds,
		options={'disp':True,'ftol':0,'gtol':0.0001})

#print(minimal)
print("minimal Y: ", post_process(minimal['fun'][0],N_Features))
print("minimal X: ", [post_process(minimal['x'][ii],ii) for ii in range(N_Features)])

# =================== try basin-hopping ============================
minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds, jac = False)
minimal = scipy.optimize.basinhopping(nn_predict_as_np,initial_guess,
                 minimizer_kwargs=minimizer_kwargs,niter=20, disp=True)

#print(minimal)
print("minimal Y: ", post_process(minimal['fun'],N_Features))
print("minimal X: ", [post_process(minimal['x'][ii],ii) for ii in range(N_Features)])

# =================== try brute force =============================

variables = [ [pre_process(1,i),pre_process(0,i)] for i in range(N_Features)]
combine_list = list(itertools.product(*variables))
#print(len(combine_list))
#print(combine_list[1])
#print([list(combine_list[0])])
mmax_F = float('Inf')
mmax_X = None
for item in combine_list:	
	input_x = [list(item)]
	output_y = nn_predict_as_np(input_x)
	if output_y < mmax_F:
		mmax_F = output_y
		mmax_x = item
	ii-=1

print("minimal Y: ", post_process(mmax_F,N_Features))
print("minimal X: ", [post_process(mmax_x[ii],ii) for ii in range(N_Features)])

print("Features: ", COLUMNS)
