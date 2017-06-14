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
<<<<<<< HEAD
def Load_Prep_Data(input_fold,nf,nl):
	train = pd.read_csv(input_fold+"train_set_preped.txt", skipinitialspace=True)
	COLUMNS = train.columns.values.tolist()
	if len(COLUMNS)!=nf+nl:
		sys.exit("data if file: train_set_preped.txt has different columns")
        valid = pd.read_csv(input_fold+"valid_set_preped.txt", skipinitialspace=True)
	if COLUMNS!= valid.columns.values.tolist():
		sys.exit("data if file: valid_set_preped.txt has different columns")
        test = pd.read_csv(input_fold+"test_set_preped.txt", skipinitialspace=True)
	if COLUMNS!= test.columns.values.tolist():
		sys.exit("data if file: test_set_preped.txt has different columns")
        pred = pd.read_csv(input_fold+"predict_set_preped.txt", skipinitialspace=True)
	if COLUMNS!= pred.columns.values.tolist():
		sys.exit("data if file: predict_set_preped.txt has different columns")

        FEATURES = COLUMNS[:nf]
        LABEL = COLUMNS[nf:]

	Xtrain = train.loc[:,FEATURES].values
	Ytrain = train.loc[:,LABEL].values
	Xtest = test.loc[:,FEATURES].values
	Ytest = test.loc[:,LABEL].values
	Xvalid = valid.loc[:,FEATURES].values
	Yvalid = valid.loc[:,LABEL].values
	Xpred = pred.loc[:,FEATURES].values
	Ypred = pred.loc[:,LABEL].values
	return Xtrain,Ytrain,Xvalid,Yvalid,Xtest,Ytest,Xpred,Ypred
=======
>>>>>>> ae4213b9a7b5babc1a7d117b88dcbdca00be9c56

def Create_Variables(n_input, n_classes=1, DIM=[50,50,50,50,50]):
	n_layer = len(DIM)
	weights = [tf.Variable(tf.random_normal([n_input, DIM[0]],0, 0.1,dtype=tf.float64))]
	for ii in range(n_layer-1):
		weights.append(tf.Variable(tf.random_normal([DIM[ii], DIM[ii+1]],0, 0.1,dtype=tf.float64)))
	weights.append(tf.Variable(tf.random_normal([DIM[n_layer-1], n_classes],0, 0.1,dtype=tf.float64)))
	biases = []
	for ii in range(n_layer):
		biases.append(tf.Variable(tf.random_normal([DIM[ii]], 0, 0.1,dtype=tf.float64)))
	biases.append(tf.Variable(tf.random_normal([n_classes], 0, 0.1,dtype=tf.float64)))
	return weights,biases

def Create_DNN_Model(x_holder, y_holder, keep_prob, n_input, n_classes=1, DIM=[50,50,50,50,50]):
	if x_holder.get_shape().as_list()[1]!=n_input:
		sys.exit("Error in creating DNN: input layer size (%d) not equal to n_input(%d)" % (x_holder.get_shape().as_list()[1],n_input))
	if y_holder.get_shape().as_list()[1]!=n_classes:
		sys.exit("Error in creating DNN: output layer size(%d) not equal to n_classes(%d)" % (y_holder.get_shape().as_list()[1],n_classes))

	weights, biases = Create_Variables(n_input, n_classes, DIM)
	n_layer = len(DIM)
	layers = [ tf.nn.relu(tf.add(tf.matmul(x_holder, weights[0]), biases[0]))]
	layers[0] = tf.nn.dropout(layers[0],keep_prob)
	for ii in range(n_layer-1):
		layers.append(tf.nn.relu(tf.add(tf.matmul(layers[ii], weights[ii+1]), biases[ii+1])))
		layers[ii+1]=tf.nn.dropout(layers[ii+1],keep_prob)
	layers.append(tf.add(tf.matmul(layers[-1], weights[-1]), biases[-1]))
	return  layers , weights, biases

def Predict_Function(layers):
	return layers[-1]

<<<<<<< HEAD
def Regularize_Function(list_of_weights):
	regularizers = tf.zeros([],dtype = tf.float64)
	for weights in list_of_weights:
		n_layer = len(weights)
		for ii in range(n_layer):
			regularizers = regularizers + tf.nn.l2_loss(weights[ii])
	return regularizers

def Cost_Function(y_holder, last_layers, list_of_weights, beta = 0):
	pred = Predict_Function(last_layers)
	regularizers = Regularize_Function(list_of_weights)
=======
def Regularize_Function(weights):
	n_layer = len(weights)-1
	regularizers = tf.nn.l2_loss(weights[n_layer])
	for ii in range(n_layer):
		regularizers = regularizers + tf.nn.l2_loss(weights[ii])
	return regularizers

def Cost_Function(y_holder,layers, weights, beta = 0):
	pred = Predict_Function(layers)
	regularizers = Regularize_Function(weights)
>>>>>>> ae4213b9a7b5babc1a7d117b88dcbdca00be9c56
	cost = tf.reduce_mean(tf.square(pred-y_holder))
	cost = cost+beta*regularizers
	return cost,pred

def Optimizer(cost, learning_rate = 0.001):
	return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

def Train_DNN(sess, saver, optimizer, pred, cost, x_holder, x_train, x_valid, x_test, y_holder, y_train,
		 y_valid, y_test, keep_prob, dropout_rate=1, path_to_save = "tmp/my_model.ckpt",
		training_epochs = 100, batch_size=64, display_step=1, accuracy_step = 10, save_step = 20):
	sess.run(tf.initialize_all_variables())
	total_batch = int(x_train.shape[0]/batch_size)
	try:
		saver.restore(sess, path_to_save)
		print("[*]---------------- start from restored result ------")
	except:
		print("[*]---------------- start from initialization ------")
	for epoch in range(training_epochs):
		avg_cost = 0.0
		for i in range(total_batch):
			batch_x = x_train[i*batch_size:(i+1)*batch_size]
			batch_y = y_train[i*batch_size:(i+1)*batch_size]
			_, c, p = sess.run([optimizer, cost, pred],
				feed_dict={x_holder: batch_x,y_holder: batch_y, keep_prob:dropout_rate}) 
			avg_cost += c/total_batch

		label_value = batch_y
		estimate = p
		if (epoch+1) % display_step == 0:
			print ("Epoch:", '%04d' % (epoch+1), "cost=", \
				"{:.9f}".format(avg_cost))
			print ("[*]----------------------------")
			for i in xrange(3):
				print ("label value:", label_value[i], \
					"estimated value:", estimate[i])
				print ("[*]============================")
		if (epoch+1) % accuracy_step == 0:
			accuracy = sess.run(cost,
				feed_dict={x_holder:x_valid, y_holder: y_valid, keep_prob:1})
			print ("Accuracy:", accuracy)
		if (epoch+1) % save_step == 0 or (epoch+1) == training_epochs:
			saver.save(sess, path_to_save)
			print ("model_saved to: "+path_to_save)
			print ("[*]_________________________")
	print ("Optimization Finished!")
<<<<<<< HEAD
	accuracy = sess.run(cost, feed_dict={x_holder:x_test, y_holder: y_test, keep_prob:1})
        print ("Final Accuracy:", accuracy)


def Pred_DNN(sess, saver, pred, x_holder, x_data, keep_prob,  path_to_save = "tmp/my_model.ckpt"):
=======
	accuracy = sess.run(cost, feed_dict={x_holder:X_test, y_holder: Y_test, keep_prob:1})
        print ("Final Accuracy:", accuracy)


def Pred_RNN(sess, saver, pred, x_holder, x_data, keep_prob):
>>>>>>> ae4213b9a7b5babc1a7d117b88dcbdca00be9c56
	try:
		saver.restore(sess, path_to_save)
                print("[*]------ successfully load saved model ------")
	except:
		print("[*]------ error: can not load saved model ----------")
		return None
	return sess.run(pred, feed_dict={x_holder: x_data, keep_prob:1})

<<<<<<< HEAD
def Plot_data(x1,y1,x2,y2,index,filename,title = 'predicted(red) vs label(blue)',lty1='rs--',lty2='b^-.'):
	plt.figure(index)
	plt.plot(x1,y1, lty1, x2,y2, lty2)
=======
def Plot_data(x1,y1,x2,y2,index,filename,title = 'predicted(blue) vs label(red)'):
	plt.figure(index)
	plt.plot(x1,y1, 'rs--', x2,y2, 'b^-.')
>>>>>>> ae4213b9a7b5babc1a7d117b88dcbdca00be9c56
	plt.xlabel('samples')
	plt.ylabel('value')
	plt.title(title)
	plt.grid(True)
	plt.savefig(filename)

if __name__ == "__main__":
	path_to_save = "tmp/my_model.ckpt"
<<<<<<< HEAD
	N_Features = 5
	N_Labels = 1
	input_fold = "input_data/"
	output_fold = "output_data/"

	X_train,Y_train,X_valid,Y_valid,X_test,Y_test,X_pred,Y_pred = Load_Prep_Data(input_fold,N_Features,N_Labels)	

	x_ = tf.placeholder("float64", [None, N_Features])
	y_ = tf.placeholder("float64", [None,N_Labels])
	keep_ = tf.placeholder(tf.float64)

	layers_, weights_, bias_ = Create_DNN_Model(x_, y_, keep_,n_input = N_Features,n_classes = N_Labels, DIM=[50,50,50,50,50])
	list_of_weights_ = [weights_]
	list_of_weights_ = []	# do not use regularization
	cost_, pred_ = Cost_Function(y_,layers_,list_of_weights_,beta = 0.001)


=======
	N_Features = 16
	input_fold = "input_data/"
	output_fold = "output_data/"

	train_set = pd.read_csv(input_fold+"train_set_preped.txt", skipinitialspace=True)
	valid_set = pd.read_csv(input_fold+"valid_set_preped.txt", skipinitialspace=True)
	test_set = pd.read_csv(input_fold+"test_set_preped.txt", skipinitialspace=True)
	pred_set = pd.read_csv(input_fold+"predict_set_preped.txt", skipinitialspace=True)
	COLUMNS = train_set.columns.values.tolist()
	FEATURES = COLUMNS[:-1]
	LABEL = [COLUMNS[-1]]

	X_train = train_set.loc[:,FEATURES].values
	Y_train = train_set.loc[:,LABEL].values
	X_test = test_set.loc[:,FEATURES].values
	Y_test = test_set.loc[:,LABEL].values
	X_valid = valid_set.loc[:,FEATURES].values
	Y_valid = valid_set.loc[:,LABEL].values
	X_pred = pred_set.loc[:,FEATURES].values
	Y_pred = pred_set.loc[:,LABEL].values
	
	x_ = tf.placeholder("float64", [None, N_Features])
	y_ = tf.placeholder("float64", [None,1])
	keep_ = tf.placeholder(tf.float64)

	layers_, weights_, bias_ = Create_DNN_Model(x_, y_, keep_,n_input = N_Features, DIM=[50,50,50,50,50])
	cost_, pred_ = Cost_Function(y_,layers_,weights_)
>>>>>>> ae4213b9a7b5babc1a7d117b88dcbdca00be9c56
	optimizer_ = Optimizer(cost_, learning_rate = 0.0001 )
	saver_ = tf.train.Saver()

	config = tf.ConfigProto()
	config = tf.ConfigProto(device_count = {'GPU':0 })

	with tf. Session(config=config) as sess:
		Train_DNN(sess, saver_, optimizer_, pred_, cost_,x_, X_train, X_valid, X_test,
<<<<<<< HEAD
			y_, Y_train, Y_valid, Y_test, keep_, dropout_rate = 1,training_epochs = 20, path_to_save = path_to_save)
		test_value = Pred_DNN(sess, saver_, pred_, x_, X_test, keep_, path_to_save = path_to_save)
		pred_value = Pred_DNN(sess, saver_, pred_, x_, X_pred, keep_, path_to_save = path_to_save)
	xcoor = range(100)
	Plot_data(xcoor,test_value[0:100],xcoor,Y_test[0:100],0,output_fold+"test_result.png")
	xcoor = range(len(Y_pred))
	Plot_data(xcoor,pred_value,xcoor,Y_pred,1,output_fold+"pred_result.png",lty1='r--',lty2='b-.')
=======
			y_, Y_train, Y_valid, Y_test, keep_, dropout_rate = 1)
		test_value = Pred_RNN(sess, saver_, pred_, x_, X_test, keep_)
		pred_value = Pred_RNN(sess, saver_, pred_, x_, X_pred, keep_)
	xcoor = range(100)
	Plot_data(xcoor,test_value[0:100],xcoor,Y_test[0:100],0,output_fold+"test_result.png")
	xcoor = range(len(Y_pred))
	Plot_data(xcoor,pred_value,xcoor,Y_pred,1,output_fold+"pred_result.png")
>>>>>>> ae4213b9a7b5babc1a7d117b88dcbdca00be9c56
