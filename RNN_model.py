import os
import sys
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import itertools
import copy
import math
import scipy.optimize
import matplotlib
matplotlib.use("Agg")				# to work without X-window
import matplotlib.pyplot as plt
from DNN_model import *
from Prep_data import * 
tf.logging.set_verbosity(tf.logging.ERROR)
from Create_Simulate_Data import y1, y2

def Predict_N_Step(sess, saver, pred, x_holder, input_data, keep_holder, n_step, n_features,index_input_state,
		index_output_state, index_out = -1, n_state =1, n_output=1, initial_state=None, keep_rate = 1, 
		path_to_save = "tmp/my_model.ckpt", process = True, means=None, maxmin=None,load = False):
	# === check input data shape
	if load:
		saver.restore(sess, path_to_save)
	if x_holder.get_shape().as_list()[1]!=n_features or input_data.shape[1]!=n_features or n_state>n_features:
		sys.exit("Error when predict time series: wrong input size")
	if pred.get_shape().as_list()[1]!=n_state+n_output:
		sys.exit("Error when predict time series: wrong output size")
	if initial_state == None:		# if no specified initial state, read from input_data
		initial_state = input_data[0][-n_state:]
	elif np.array(initial_state).shape[1]!=n_state:
		sys.exit("Error when predict time series: wrong initial state size")
	# === start predicting
	if process:
		if means == None or maxmin==None:
			sys.exit("Error when predict time series: can not perform data processing")
		predict_y = []	# predicted values (state and output labels)
		total = 0	# final value (sum of output of each step)
		for ii in range(n_step):			# predict values for n steps
			# create input features for current step
			fs =np.array([[input_data[ii][jj] for jj in range(n_features)]])
			# create state feautres for current step
			fs[0][index_input_state] = initial_state
			# calculate predicted states and output labels
			pred_value = Pred_DNN(sess=sess, saver=saver,pred= pred, x_holder=x_holder, x_data = fs, 
				keep_prob=keep_holder, keep_rate=keep_rate, path_to_save = path_to_save)
			# post process the predicted values
			pred_value= np.array([[ Post_Process(pred_value[0][jj],n_features+jj, means,maxmin) 
					for jj in range(n_state+n_output)]])
			# append predicted values to return list
			# add output to "total"
			predict_y.append(pred_value[0][:])
			total+=pred_value[0][index_out]
			# create state features for next step (preprocess current predicted state features )
			initial_state = np.array([ Pre_Process(pred_value[0][index_output_state[jj]],index_input_state[jj], means,maxmin)
					for jj in range(n_state)])
	# return predict values for each step and sum of output values
	return predict_y,total


def Predict_N_Step_np(input_data, py_weights, py_biases,DIM, n_step, n_features,index_input_state,
		index_output_state, index_out = -1, n_state =1, n_output=1, initial_state=None,
		process = True, means=None, maxmin=None,load = False):
	# === check input data shape
	if py_weights[0].shape[0]!=n_features or input_data.shape[1]!=n_features or n_state>n_features:
		sys.exit("Error when predict time series: wrong input size")
	if py_weights[-1].shape[1]!=n_state+n_output:
		sys.exit("Error when predict time series: wrong output size")
	if initial_state == None:	       # if no specified initial state, read from input_data
		initial_state = input_data[0][-n_state:]
	elif np.array(initial_state).shape[1]!=n_state:
		sys.exit("Error when predict time series: wrong initial state size")
	# === start predicting
	if process:
		if means == None or maxmin==None:
			sys.exit("Error when predict time series: can not perform data processing")
		predict_y = []  # predicted values (state and output labels)
		total = 0       # final value (sum of output of each step)
		for ii in range(n_step):			# predict values for n steps
			# create input features for current step
			fs =np.array([[input_data[ii][jj] for jj in range(n_features)]])
			# create state feautres for current step
			fs[0][index_input_state] = initial_state
			# calculate predicted states and output labels
			pred_value = Recreate_NN(fs,py_weights=py_weights,py_biases=py_biases,DIM=DIM)
			# post process the predicted values
			pred_value= np.array([[ Post_Process(pred_value[0][jj],n_features+jj, means,maxmin)
					for jj in range(n_state+n_output)]])
			# append predicted values to return list
			# add output to "total"
			predict_y.append(pred_value[0][:])
			total+=pred_value[0][index_out]
			# create state features for next step (preprocess current predicted state features )
			initial_state = np.array([ Pre_Process(pred_value[0][index_output_state[jj]],index_input_state[jj], means,maxmin)
					for jj in range(n_state)])
	# return predict values for each step and sum of output values
	return predict_y,total


if __name__ == "__main__":
	# ===========================================================
	# ========  Load data, create NN and perform training =======
	# ===========================================================
	path_to_save = "tmp/my_model.ckpt"
	N_Features = 24 
	N_Labels = 9
	input_fold = "input_data/"
	output_fold = "output_data/"
	X_train,Y_train,COLUMNS_ = Load_Prep_Data(input_fold+"train_set_preped.txt",N_Features,N_Labels)
	X_valid,Y_valid,COLUMNS_ = Load_Prep_Data(input_fold+"valid_set_preped.txt",N_Features,N_Labels,COLUMNS_)
	X_test, Y_test, COLUMNS_ = Load_Prep_Data(input_fold+"test_set_preped.txt",N_Features,N_Labels, COLUMNS_)

	x_ = tf.placeholder("float64", [None, N_Features])
	y_ = tf.placeholder("float64", [None,N_Labels])
	keep_ = tf.placeholder(tf.float64)
	DIM_ = [50,50,50,50,50]

	embed_layer_, embed_coeff_ = Create_Embed_Layer(x_, N_Features)	

	layers0, weights0, biases0 = Create_DNN_Model(embed_layer_, y_, keep_,n_input = N_Features,n_classes=N_Labels, DIM= DIM_)

	list_of_regular_ = [weights0]
	list_of_train_ = [weights0,biases0]

	list_of_weights_ = []  # do not use regularization	
	cost_, pred_ = Cost_Function(y_,layers0,list_of_regular_,beta = 0.0)
	optimizer_ = Optimizer(cost_, learning_rate = 0.00005,var_list=list_of_train_ )

	saver_ = tf.train.Saver()

	config = tf.ConfigProto()
	config = tf.ConfigProto(device_count = {'GPU':0 })

	with tf.Session(config=config) as sess0_:
		Train_DNN(sess0_, saver_, optimizer_, pred_, cost_,x_, X_train, X_valid, X_test,
			y_, Y_train, Y_valid, Y_test, keep_, dropout_rate = 1,path_to_save = path_to_save,
			training_epochs =0, batch_size=64, display_step=1, accuracy_step = 1, save_step = 10)
	# =================================================
	# =================================================
	# ============ DNN with time series ===============
	# =================================================
	# =================================================
	print "========= now entering DNN with time series ==========="
	n_step_ = 20			# number of time steps
	index_control_ = range(8)		# index of control variables 
						# namely, the variables in optimization
	index_input_state_ = range(16,24)	# index of input state variables
	index_output_state_ = range(8)		# index of output state variables
	index_out_     = -1			# index of final output 
						# the cost function in optimization

	n_state_ = len(index_input_state_)	# number of states
	if n_state_ != len(index_output_state_):
		sys.exit("different number of state-variables for input and output")
	n_output_ = 1 
	n_control_ = len(index_control_)	
	if n_output_ + n_state_ != N_Labels:
		sys.exit("wrong number of state-variables, output, or labels")

	ctrl_range = [-1.0,0.0,1.0]
	#ctrl_range = [-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0]

	means_,maxmin_ = Load_Process_Param(input_fold+"preprocess.txt",COLUMNS_)		# data for pre- and post processing

	predict_y1 = []			# predicted states
	predict_y2 = []			# predicted output
	true_y1 =[]			# true states
	true_y2 =[]			# true output
	# ==================================================================
	# ===== testing accuracy of DNN time series (with for loop) ========
	# ==================================================================
	with  tf.Session(config=config) as sess1_:
		# ===== initialize features for prediction (pre-processed)
		# for continuous input
#		fs = [[random.random()-0.5 for ii in range(N_Features)]]
#		for ii in range(n_state_):
#			fs[0][N_Features-n_state_+ii] = Pre_Process(0,N_Features-1, means_,maxmin_)	# initial state set to 0
		# for discrete input
		fs = [[Pre_Process(random.choice(ctrl_range),ii, means_,maxmin_) for ii in range(N_Features)]]
		for ii in range(n_state_):
			fs[0][index_input_state_[ii]] = Pre_Process(0,index_input_state_[ii], means_,maxmin_)	# initial state set to 0

		# ---------------  (for testing, manually set other variable )-----------------------------
#		for ii in range(n_control_):
#			fs[0][ii+n_control_] = Pre_Process(0.,ii+n_control_, means_,maxmin_)
		# --------------- end for testing ---------------------------------------------------------

		# ===== initialize features for true labels (post-processed)
		true_fs = [[Post_Process(fs[0][ii],ii, means_,maxmin_) for ii in range(N_Features)]]
		for ii in range(n_state_):
			true_fs[0][index_input_state_[ii]] = 0		# initial state set to 0

		# ===== load variables of DNN
		saver_.restore(sess1_, path_to_save)
		# ===== start predicting =========
		input_data =[]			# save the generated input data (control variable) for further testing
		for step in range(n_step_):
			input_data.append(fs[0])

			# ========= true lable, use post-processed data 
			true_value = y1(true_fs[0])+y2(true_fs[0])
			true_y1.append([true_value[index_output_state_[ii]] for ii in range(n_state_)])
			true_y2.append(true_value[index_out_])
			
			# ========= predict labels 
			# ==== get predicted labels; here input data (fs) and output data (pred_value) are pre-processed, 
			pred_value = Pred_DNN(sess1_, saver_, pred_, x_, fs, keep_, path_to_save = path_to_save)
			# ==== get post-processed labels, and append them to output result
			pred_value =[[ Post_Process(pred_value[0][index],N_Features+index, means_,maxmin_) for index in range(N_Labels)]]
			predict_y1.append([pred_value[0][index_output_state_[ii]] for ii in range(n_state_)])
			predict_y2.append(pred_value[0][index_out_])

			# ==== pre-process predicted label and generate inputs for the next prediction
			# continuous input features
#			fs = [[random.random()-0.5 for ii in range(N_Features)]]
#			for ii in range(n_state_):
#				fs[0][N_Features-n_state_+ii] =  Pre_Process(pred_value[0][ii],N_Features-n_state_+ii, means_,maxmin_)
			# discrete input features
	                fs = [[Pre_Process(random.choice(ctrl_range),ii, means_,maxmin_) for ii in range(N_Features)]]
			for ii in range(n_state_):
				fs[0][index_input_state_[ii]] =  Pre_Process(pred_value[0][index_output_state_[ii]],
						index_input_state_[ii], means_,maxmin_)

			# ---------------  (for testing, manually set other variable )-----------------------------
#			for ii in range(n_control_):
#				fs[0][ii+n_control_] = Pre_Process(0.,ii+n_control_, means_,maxmin_)
			# --------------- end for testing ---------------------------------------------------------

			# ==== generate inputs for next true label
			true_fs = [[Post_Process(fs[0][ii],ii, means_,maxmin_) for ii in range(N_Features)]]
			for ii in range(n_state_):
				true_fs[0][index_input_state_[ii]] = true_value[ii]

	# print true & predicted states VS time step
	fig = plt.figure(0)
	plt.plot(predict_y1,'r--', label = 'predict')
	plt.plot(true_y1,'b-.' , label = 'label')
#	plt.legend(loc='upper left')
	plt.xlabel('time step')
	plt.ylabel('value')
	plt.title("Y1")
	plt.grid(True)
	plt.savefig(input_fold+"test_Y1.png")
	plt.close(fig)
	# print true & predicted output VS time step
	fig = plt.figure(1)
	plt.plot(predict_y2,'r--', label = 'predict')
	plt.plot(true_y2,'b-.' , label = 'label')
	plt.legend(loc='upper left')
	plt.xlabel('time step')
	plt.ylabel('value')
	plt.title("Y2")
	plt.grid(True)
	plt.savefig(input_fold+"test_Y2.png")	
	plt.close(fig)

#	sys.exit()
	# ======= end of testing accuracy of DNN time series ========

	# ==================================================================
	# ================= test Predict_N_Step function ===================
	# ==================================================================
	input_data = np.array(input_data)
	with  tf.Session(config=config) as sess2_:
		saver_.restore(sess2_, path_to_save)
		predict_v, total_E = Predict_N_Step(sess=sess2_, saver=saver_, pred=pred_, x_holder=x_, 
			input_data = input_data, keep_holder=keep_, n_step=n_step_, n_features=N_Features,
			index_input_state = index_input_state_, index_output_state = index_output_state_,
			n_state =n_state_, n_output=1, index_out=index_out_, initial_state=None, keep_rate = 1, 
			path_to_save = "tmp/my_model.ckpt", process = True, means=means_, maxmin=maxmin_)
	# difference between Predict_N_Step and for loop
	print "======== testing Predicting function ========"
	print np.array(predict_v)[:,:-1] - (predict_y1)
	print np.array(predict_v)[:,-1] - predict_y2
	print "=========== should be closed to 0 ===========" 
#	sys.exit()

	# ==================================================================
	# =============== test Predict_N_Step_np function =================
	# ==================================================================
	with  tf.Session(config=config) as sess3_:
		saver_.restore(sess3_, path_to_save)
		py_weights_, py_biases_ = Load_Variable(sess3_, saver_,weights0,biases0, path_to_save = path_to_save)

	predict_v, total_E = Predict_N_Step_np(input_data=input_data, py_weights=py_weights_, py_biases=py_biases_,DIM = DIM_, n_step=n_step_,
		n_features=N_Features, index_input_state = index_input_state_,	index_output_state= index_output_state_, 
		index_out = index_out_, n_state = n_state_, n_output=1, means=means_, maxmin=maxmin_)
	# difference between Predict_N_Step and for loop
	print "======== testing Predicting function NP ========"
	print np.array(predict_v)[:,:-1] - (predict_y1)
	print np.array(predict_v)[:,-1] - predict_y2
	print "=========== should be closed to 0 ==========="

