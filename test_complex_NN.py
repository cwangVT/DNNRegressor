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
from DNN_model import *

if __name__ == "__main__":
	path_to_save = "tmp/my_model.ckpt"
	N_Features = 5 
	N_Labels = 1
	input_fold = "input_data/"
	output_fold = "output_data/"

	X_train,Y_train,X_valid,Y_valid,X_test,Y_test,X_pred,Y_pred = Load_Prep_Data(input_fold,N_Features,N_Labels)	

	x_ = tf.placeholder("float64", [None, N_Features])
	y_ = tf.placeholder("float64", [None,1])
	keep_ = tf.placeholder(tf.float64)

	layers0, weights0, bias0 = Create_DNN_Model(x_, y_, keep_,n_input = N_Features, DIM=[50,50,50,50,50])
	layers1, weights1, bias1 = Create_DNN_Model(x_, y_, keep_,n_input = N_Features, DIM=[50,50,50,50,50])
        layers2, weights2, bias2 = Create_DNN_Model(x_, y_, keep_,n_input = N_Features, DIM=[50,50,50,50,50])
        layers3, weights3, bias3 = Create_DNN_Model(x_, y_, keep_,n_input = N_Features, DIM=[50,50,50,50,50])
        layers4, weights4, bias4 = Create_DNN_Model(x_, y_, keep_,n_input = N_Features, DIM=[50,50,50,50,50])

	try:
		layers_ii = tf.concat(1, [layers0[-1],layers1[-1],layers2[-1],layers3[-1],layers4[-1]])		# for older version of tensorflow
	except:
		layers_ii = tf.concat([layers0[-1],layers1[-1],layers2[-1],layers3[-1],layers4[-1]], 1) 	# for newer version of tensorflow

	layers_, weights_, bias_ = Create_DNN_Model(layers_ii, y_, keep_, n_input=N_Features,n_classes=N_Labels, DIM=[5,5,5,5,5])

	list_of_weights_=[weights0,weights1,weights2,weights3,weights4,weights_]
	list_of_weights_ = []  # do not use regularization	
	cost_, pred_ = Cost_Function(y_,layers_,list_of_weights_,beta = 0.0)
	optimizer_ = Optimizer(cost_, learning_rate = 0.0001 )
	saver_ = tf.train.Saver()

	config = tf.ConfigProto()
	config = tf.ConfigProto(device_count = {'GPU':0 })

	with tf. Session(config=config) as sess:
		Train_DNN(sess, saver_, optimizer_, pred_, cost_,x_, X_train, X_valid, X_test,
			y_, Y_train, Y_valid, Y_test, keep_, dropout_rate = 1,path_to_save = path_to_save,
			training_epochs = 100, batch_size=64, display_step=1, accuracy_step = 10, save_step = 20)
		test_value = Pred_DNN(sess, saver_, pred_, x_, X_test, keep_, path_to_save = path_to_save)
		pred_value = Pred_DNN(sess, saver_, pred_, x_, X_pred, keep_, path_to_save = path_to_save)
	xcoor = range(100)
	Plot_data(xcoor,test_value[0:100],xcoor,Y_test[0:100],0,output_fold+"test_result.png")
	xcoor = range(len(Y_pred))
	Plot_data(xcoor,pred_value,xcoor,Y_pred,1,output_fold+"pred_result.png",lty1='r--',lty2='b-.')
