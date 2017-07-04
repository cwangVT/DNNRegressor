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
from RNN_model import Predict_N_Step
from RNN_model import Predict_N_Step_np


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
			training_epochs = 0, batch_size=64, display_step=1, accuracy_step = 1, save_step = 10)
	# =================================================
	# ============ Create testing input ===============
	# =================================================
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

	means_,maxmin_ = Load_Process_Param(input_fold+"preprocess.txt",COLUMNS_)	# data for pre- and post processin
	print means_
	# ===== initialize features for prediction (pre-processed)
	# for continuous input
#	fs = [[random.random()-0.5 for ii in range(N_Features)]]
#	for ii in range(n_state_):
#		fs[0][N_Features-n_state_+ii] = Pre_Process(0,N_Features-1, means_,maxmin_)	# initial state set to 0
	# for discrete input
	fs = [[Pre_Process(random.choice(ctrl_range),ii, means_,maxmin_) for ii in range(N_Features)]]
	for ii in range(n_state_):
		fs[0][index_input_state_[ii]] = Pre_Process(0,index_input_state_[ii], means_,maxmin_)	# initial state set to 0

	# ---------------  (for testing, manually set other variable )-----------------------------
	for ii in range(n_control_):
		fs[0][ii+n_control_] = Pre_Process(0.,ii+n_control_, means_,maxmin_)
	# --------------- end for testing ---------------------------------------------------------

	input_data =[]			# generated input data (control variable) for further testing
	for step in range(n_step_):
		input_data.append(fs[0])
		# generate inputs for the next prediction
		# continuous input features
#		fs = [[random.random()-0.5 for ii in range(N_Features)]]
#		for ii in range(n_state_):
#			fs[0][N_Features-n_state_+ii] =  Pre_Process(pred_value[0][ii],N_Features-n_state_+ii, means_,maxmin_)
		# discrete input features
		fs = [[Pre_Process(random.choice(ctrl_range),ii, means_,maxmin_) for ii in range(N_Features)]]
		for ii in range(n_state_):
			fs[0][index_input_state_[ii]] =  Pre_Process(0.0,index_input_state_[ii], means_,maxmin_)

		# ---------------  (for testing, manually set other variable )-----------------------------
		for ii in range(n_control_):
			fs[0][ii+n_control_] = Pre_Process(0.,ii+n_control_, means_,maxmin_)
		# --------------- end for testing ---------------------------------------------------------


	input_data = np.array(input_data)
	# ==============================================
	# ==============================================
	# ================== try SA ====================
	# ==============================================
	# ==============================================
	print "========= now entering Simulating Annealing ==========="
	n_opt_dim = n_control_ * n_step_		# total number of dimensions

	# =============================
	# === parameters to tune ======
	# =============================

	n_sa_step = 12157665459056928801				# total step
	n_sa_step = 20000000
	n_sa_disp = 1000

	min_T = 0.2
	#min_T = 0.00
	d_T = Temperature = 10 + min_T		# initial Temperature
	#d_T = Temperature = 0.000001 + min_T
	decay_ratio_T = 0.99				# T decay to (T*decay_ratio_T) every (decay_step_T) steps
	decay_step_T = 60 

	min_S = 1
	d_S = n_selection = int(n_opt_dim-1)+min_S	# number of dimensions to change
	decay_ratio_S = 0.99				# select_ration S decay to (S*decay_ratio_S) every (decay_step_S) steps
	decay_step_S = 40 
	with  tf.Session(config=config) as sess2_:
		saver_.restore(sess2_, path_to_save)
		py_weights_, py_biases_ = Load_Variable(sess2_, saver_,weights0,biases0, path_to_save = path_to_save)
		print "Current Temperature is ", Temperature
		print "Current # of Selected DIM is ", n_selection
		# ====================================
		# ========== Start SA ================
		# ====================================

	if True:
		cost_t = []
		temp_t = []
		sele_t = []
		acpt_t = []
		dE_t = []
	# ================================================================
	# ====================== initial Energy ===========================
		total_E = float("inf")
	# ==== using Predict_N_Step with tf
	#	predict_v, total_E = Predict_N_Step(sess=sess2_, saver=saver_, pred=pred_, x_holder=x_,
	#		input_data = input_data, keep_holder=keep_, n_step=n_step_, n_features=N_Features, n_state =n_state_,
	#		index_input_state = index_input_state_, index_output_state = index_output_state_,
	#		n_output=1, index_out=index_out_, initial_state=None, keep_rate = 1,
	#		path_to_save = "tmp/my_model.ckpt", process = True, means=means_, maxmin=maxmin_)

	# ==== using Predict_N_Step_np with numpy
		predict_v, total_E = Predict_N_Step_np(input_data=input_data, py_weights=py_weights_, 
			py_biases=py_biases_,DIM = DIM_, n_step=n_step_, n_features=N_Features, 
			index_input_state = index_input_state_,  index_output_state= index_output_state_,
			index_out = index_out_, n_state = n_state_, n_output=1, means=means_, maxmin=maxmin_)

		for ii in range(n_sa_step):
			tmp = copy.deepcopy(input_data)
		# =========================================================================
		# ========= Randomly select n_selection dimensions to change ==============
			selected_dim = np.random.choice(n_opt_dim, n_selection)
			for index in selected_dim:
				jj = index/n_control_
				kk = index_control_[index%n_control_]
				# input_data[jj][kk] is selected to change
		# ==========================================================================
		# ========== Randomly set variables in INPUT ===============================
		# ========== Continuous Variable
		#		tmp[jj][kk] += (random.random()-0.5)
		#		if tmp[jj][kk]>0.5:
		#			tmp[jj][kk]=0.5
		#		if tmp[jj][kk]<-0.5:
		#			tmp[jj][kk]=0.5]

		# ========== Discrete Variable 
		#	(always change) 
				new_tmp = Pre_Process(random.choice(ctrl_range),kk, means_,maxmin_)
				while tmp[jj][kk] == new_tmp:
					new_tmp = Pre_Process(random.choice(ctrl_range),kk, means_,maxmin_)
				tmp[jj][kk] = new_tmp
		#	(may not change)
		#		tmp[jj][kk] = Pre_Process(random.choice([-1,0.0,1]),kk, means_,maxmin_)

		# ==========================================================================
		# ========== Calculate total E of new INPUT ================================
		# ========== using Predict_N_Step, with TF
		#	predict_v, tmp_E = Predict_N_Step(sess=sess2_, saver=saver_, pred=pred_, x_holder=x_,
		#		input_data =tmp , keep_holder=keep_, n_step=n_step_, n_features=N_Features, n_state =n_state_,
		#		index_input_state = index_input_state_, index_output_state = index_output_state_,
		#		n_output=1, index_out=index_out_, initial_state=None, keep_rate = 1,
		#		path_to_save = "tmp/my_model.ckpt", process = True, means=means_, maxmin=maxmin_)

		# ========== using Predict_N_Step_np, with numpy
			predict_v, tmp_E = Predict_N_Step_np(input_data=tmp, py_weights=py_weights_, py_biases=py_biases_,DIM = DIM_, n_step=n_step_,
				n_features=N_Features, index_input_state = index_input_state_,  index_output_state= index_output_state_,
				index_out = index_out_, n_state = n_state_, n_output=1, means=means_, maxmin=maxmin_)

		# ========= using True testing Function
		#	tmp_E=0
		#	for jj in range(n_step_):
		#		current_tmp = [  Post_Process(tmp[jj][kk],kk,means_,maxmin_) for kk in range(N_Features) ]
		#		tmp_E += sum(y2(current_tmp))	
		# ======================================================================
		# ==================== Calculate Acception Prob ========================
			dE_t.append(tmp_E-total_E)
			
			if total_E > tmp_E:
				acc_prob =1
			else:	
				acc_prob = math.exp((total_E - tmp_E )/Temperature );
			if random.random()<= acc_prob:
				input_data = copy.deepcopy(tmp)
				total_E = tmp_E
			#	print "===============",total_E
		# ====================================================================
		# ================== save useful info ================================
			temp_t.append(Temperature)
			cost_t.append(total_E)
			sele_t.append(n_selection)
			acpt_t.append(acc_prob)
			#dE_t.append(tmp_E-total_E)
			if (ii+1) % (decay_step_T) == 0:
				d_T *= decay_ratio_T
				Temperature = d_T+min_T
			#	print "Current Temperature is ", Temperature
			if (ii+1) % (decay_step_S) == 0:
				d_S *= decay_ratio_S
				n_selection =int(round(d_S))+min_S
			#	print "Current # of Selected DIM is ", n_selection
			if (ii+1) % n_sa_disp == 0:
				print "current step", ii+1
				print "current cost", total_E
				print "current temp", Temperature
				print "current sele", n_selection
	# ===========================================
	# ============= END OF SA ===================
	# ===========================================

		fig = plt.figure(3)
		plt.plot(temp_t,'r--', label = 'temperature')
		plt.legend(loc='upper right')
		plt.xlabel('Step')
		plt.ylabel('value')
		plt.title("Temperature")
		plt.grid(True)
		plt.savefig(output_fold+"SA_temp.png")
		plt.close(fig)

		fig = plt.figure(4)
		plt.plot(cost_t,'r--', label = 'cost function')
		plt.legend(loc='upper right')
		plt.xlabel('Step')
		plt.ylabel('value')
		plt.title("Cost Function")
		plt.grid(True)
		plt.savefig(output_fold+"SA_cost.png")
		plt.close(fig)

		fig = plt.figure(5)
		plt.plot(sele_t,'r--', label = '# of DIM')
		plt.legend(loc='upper right')
		plt.xlabel('Step')
		plt.ylabel('value')
		plt.title("Selected Dimension")
		plt.grid(True)
		plt.savefig(output_fold+"SA_sele.png")
		plt.close(fig)

		fig = plt.figure(6)
		plt.plot(acpt_t,'r', label = 'accept prob', linewidth=0.1)
#		plt.bar(np.arange(n_sa_step),acpt_t)
		plt.legend(loc='upper right')
		plt.xlabel('Step')
		plt.ylabel('value')
		plt.title("Accept Prob")
		plt.grid(True)
		plt.savefig(output_fold+"SA_acpt.png")
		plt.close(fig)

		fig = plt.figure(7)
		plt.plot(dE_t,'r--', label = 'dE')
		plt.legend(loc='upper right')
		plt.xlabel('Step')
		plt.ylabel('value')
		plt.title("dE")
		plt.grid(True)
		plt.savefig(output_fold+"SA_dE.png")
		plt.close(fig)

		Save_Data(output_fold+'accept_rate.txt',acpt_t)
		Save_Data(output_fold+'Cost_function.txt',cost_t)
		# =========== optimized input ==================
		for ii in range(n_step_):
			print ['%5.3f' %Post_Process(input_data[ii][jj],jj,means_,maxmin_) for jj in index_control_]
		print "total_E", total_E

#		predict_v, tmp_E = Predict_N_Step(sess=sess2_, saver=saver_, pred=pred_, x_holder=x_,
#			input_data =input_data , keep_holder=keep_, n_step=n_step_, n_features=N_Features, n_state =n_state_,
#			n_output=1, index_out=index_out_, initial_state=None, keep_rate = 1,
#			index_input_state = index_input_state_, index_output_state = index_output_state_,
#			path_to_save = "tmp/my_model.ckpt", process = True, means=means_, maxmin=maxmin_)

		predict_v, tmp_E = Predict_N_Step_np(input_data=input_data, py_weights=py_weights_, py_biases=py_biases_,DIM = DIM_, n_step=n_step_,
			n_features=N_Features, index_input_state = index_input_state_,  index_output_state= index_output_state_,
			index_out = index_out_, n_state = n_state_, n_output=1, means=means_, maxmin=maxmin_)


#		pred_v = Pred_DNN(sess2_, saver_, pred_, x_, input_data, keep_, path_to_save = path_to_save)

		print "predicted_E", tmp_E

#		print "predicted_value", predict_v
#		print "pred_value", pred_v
#		print "posted", [Post_Process(pred_v[0][ii],N_Features+ii,means_,maxmin_ )for ii in range(N_Labels)]
		# =========== true output =====================
		input_data0 = copy.deepcopy(input_data)
		true_E=0
		for ii in range(n_step_):
			for jj in range(N_Features):
				input_data0[ii][jj] = Post_Process(input_data[ii][jj],jj,means_,maxmin_)
			true_E += sum(y2(input_data0[ii]))

		print "true_E",true_E
		# manually set minimal to compare
		input_data0 = copy.deepcopy(input_data)
		for ii in range(n_step_):
			for jj in index_control_:
				tmpx2 = Post_Process(input_data[ii][jj+8],jj+8,means_,maxmin_)
				cut_off = 1e-10
				if abs(tmpx2 +1.0) < cut_off:
					tmpx1 = -0.
				elif abs(tmpx2+0.75) < cut_off:
					tmpx1 = -0.
				elif abs(tmpx2+0.5) < cut_off:
					tmpx1 = -0.
				elif abs(tmpx2+0.25) < cut_off:
					tmpx1 = -0.
				elif abs(tmpx2)< cut_off:
					tmpx1 = -0.
				elif abs(tmpx2-0.25)< cut_off:
					tmpx1 = -0.
				elif abs(tmpx2-0.5)< cut_off:
					tmpx1 = -0.
				elif abs(tmpx2-0.75)< cut_off:
					tmpx1 = -0.
				elif abs(tmpx2-1)<cut_off:
					tmpx1 = -0.
				else:
					print "tmpx2",tmpx2
#					sys.exit("error input")
				input_data0[ii][jj] = Pre_Process(tmpx1,jj,means_,maxmin_)
					
#		predict_v, tmp_E = Predict_N_Step(sess=sess2_, saver=saver_, pred=pred_, x_holder=x_,
#			input_data =input_data0 , keep_holder=keep_, n_step=n_step_, n_features=N_Features, n_state =8,
#			n_output=1, index_out=index_out_, initial_state=None, keep_rate = 1,
#			index_input_state = index_input_state_, index_output_state = index_output_state_,
#			path_to_save = "tmp/my_model.ckpt", process = True, means=means_, maxmin=maxmin_)

		predict_v, tmp_E = Predict_N_Step_np(input_data=input_data0, py_weights=py_weights_, py_biases=py_biases_,DIM = DIM_, n_step=n_step_,
			n_features=N_Features, index_input_state = index_input_state_,  index_output_state= index_output_state_,
			index_out = index_out_, n_state = n_state_, n_output=1, means=means_, maxmin=maxmin_)


		# ========== True minimal E and corresponding input ==========
		print "================ true optimal =========="
		print "predicted",tmp_E
		for ii in range(n_step_):
			print  ['%5.3f' % Post_Process(input_data0[ii][jj],jj,means_,maxmin_) for jj in index_control_]
#			print  ['%5.3f' % Post_Process(input_data0[ii][jj+8],jj+8,means_,maxmin_) for jj in index_control_]
		input_data00 = copy.deepcopy(input_data0)
		true_E=0
		for ii in range(n_step_):
			for jj in range(N_Features):
				input_data00[ii][jj] = Post_Process(input_data0[ii][jj],jj,means_,maxmin_)
			true_E += sum(y2(input_data00[ii]))

		print "true_E",true_E



		# ========== difference between predict and true solusion =======
		print "============== Different ================="
		for ii in range(n_step_):
			print  ['%5.3f' % ( Post_Process(input_data[ii][jj],jj,means_,maxmin_)-
				Post_Process(input_data0[ii][jj],jj,means_,maxmin_)) for jj in index_control_]


