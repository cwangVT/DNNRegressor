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
from Prep_data import Save_Data
def Load_Prep_Data(filename,nf,nl,COLUMNS=None):	
	# Load dataset from csv files
	# return the features and labels of dataset in the format of numpy array

	# input parameters:
	# input_fold: path to the files
	# nf: number of Features
	# nl: number of Labels
	# COLUMNS: name of Features
	
	# output result:
	# X(Y)data: Features (Labels) of dataset.

	# load data from csv (pandas Dataframe work)
	
	dataset = pd.read_csv(filename, skipinitialspace=True)
	if COLUMNS == None:
		COLUMNS = dataset.columns.values.tolist()
		if len(COLUMNS)!=nf+nl:
			sys.exit("different number of features in file")
	else:
		if len(COLUMNS)!=nf+nl:
			sys.exit("different number of features in input COLUMNS")
		if COLUMNS!= dataset.columns.values.tolist():
			sys.exit("different number or name of COLUMNS")

        FEATURES = COLUMNS[:nf]
        LABEL = COLUMNS[nf:]

	# seperate features (X) and labels (Y), and keep the values only
	Xdata = dataset.loc[:,FEATURES].values
	Ydata = dataset.loc[:,LABEL].values
	return Xdata,Ydata,COLUMNS

def Create_Embed_Layer(x_holder,n_input):
	# Create the "embedding layer"
	if x_holder.get_shape().as_list()[1]!=n_input:
                sys.exit("Error in creating DNN: input layer size (%d) not equal to n_input(%d)" % (x_holder.get_shape().as_list()[1],n_input))
	embed_coeff = tf.Variable(np.array([[1.0 for ii in range(n_input)]]),dtype=tf.float64)
	try:
		embed_layer = tf.multiply(x_holder,embed_coeff)
	except:
		embed_layer = tf.mul(x_holder,embed_coeff)
#	embed_layer = x_holder*embed_coeff
	return embed_layer, embed_coeff

def Create_DNN_Model(x_holder, y_holder, keep_prob, n_input, n_classes=1, DIM=[50,50,50,50,50]):
	# Create the Deep NN

	# input paramerters:
	# x_holder: input data of NN
	# y_holder: output data of NN
	# keep_prob: dropout parameter
	# n_input: dimension of input data (# of Features)
	# n_classes: dimension of output data (# of Labels)
	# DIM: size of each hidden layer

	# output result:
	# layers: neurons in each NN layer, i.e. layers[i] are neurons in the ith layer
	# weights: linear combination weights for each NN layer.
	# biases: linear combination biases for each NN layer.
	
	# According to the forward propgation of NN, the relations between layers, weights and biases:
	# layers[i] = activation_function(weights[i]*layers[i-1]+biases[i])
	# ReLU is used as activation function

	if x_holder.get_shape().as_list()[1]!=n_input:
		sys.exit("Error in creating DNN: input layer size (%d) not equal to n_input(%d)" % (x_holder.get_shape().as_list()[1],n_input))
	if y_holder.get_shape().as_list()[1]!=n_classes:
		sys.exit("Error in creating DNN: output layer size(%d) not equal to n_classes(%d)" % (y_holder.get_shape().as_list()[1],n_classes))

	weights, biases = Create_Variables(n_input, n_classes, DIM)	# create the weights and biases
	n_layer = len(DIM)	# n_layer: number of hidden layers
	# create the first hidden layer based on input layer
	layers = [ tf.nn.relu(tf.add(tf.matmul(x_holder, weights[0]), biases[0]))]
	layers[0] = tf.nn.dropout(layers[0],keep_prob)		# dropout
	# create the second to the last hidden layer based on previous layer
	for ii in range(n_layer-1):
		layers.append(tf.nn.relu(tf.add(tf.matmul(layers[ii], weights[ii+1]), biases[ii+1])))
		layers[ii+1]=tf.nn.dropout(layers[ii+1],keep_prob)
	# create the output layer, layer[out] = layer[-1]*weight[-1]+biases[-1]
	# Notice: no activation function in output layer
	layers.append(tf.add(tf.matmul(layers[-1], weights[-1]), biases[-1]))
	# Notice: if there are n hidden layers, then len(layers)=len(weights)=len(biases)=n+1
	return  layers , weights, biases

def Create_Variables(n_input, n_classes=1, DIM=[50,50,50,50,50]):
	# Create TF-Variables weights and biases for NN

	# input parameters:
	# n_input: dimension of input data (# of Features)
	# n_classes: dimension of output data (# of Labels)
	# DIM: size of each hidden layer

	# weights: linear combination weights for each NN layer.
	# biases: linear combination biases for each NN layer.

	# shape of weights[i] depends on the size of layer i and layer i-1, 
	# 	namely, weights[i] is a matrix of size (DIM[ii-1], DIM[ii])
	# shape of biases[i] is the size of layer i, namely,
	#	namely, biases[i] is a array of size DIM[i]

	n_layer = len(DIM)      # n_layer: number of hidden layers

	# create weights	
	weights = [tf.Variable(tf.random_normal([n_input, DIM[0]],0, 0.1,dtype=tf.float64))]
	for ii in range(n_layer-1):
		weights.append(tf.Variable(tf.random_normal([DIM[ii], DIM[ii+1]],0, 0.1,dtype=tf.float64)))
	weights.append(tf.Variable(tf.random_normal([DIM[n_layer-1], n_classes],0, 0.1,dtype=tf.float64)))

	# create biases
	biases = []
	for ii in range(n_layer):
		biases.append(tf.Variable(tf.random_normal([DIM[ii]], 0, 0.1,dtype=tf.float64)))
	biases.append(tf.Variable(tf.random_normal([n_classes], 0, 0.1,dtype=tf.float64)))
	return weights,biases

def Predict_Function(layers):
	# Output Labels of NN
	# usually it is just the last layer of NN, namely layers[-1]
	return layers[-1]

def Regularize_Function(list_of_weights):
	# Consider Regularization for cost function

	# input parameters:
	# list_of_weights: weights that need to be regularized

	# output result:
	# regularizers: contributation to cost function due to regularization
	regularizers = tf.zeros([],dtype = tf.float64)

	for weights in list_of_weights:
		n_layer = len(weights)
		for ii in range(n_layer):
			regularizers = regularizers + tf.nn.l2_loss(weights[ii])
	return regularizers

def Cost_Function(y_holder, last_layers, list_of_weights, beta = 0):
	# Compute the Cost Function of the NN

	# input parameters:
	# y_holder: ground truth value (label)
	# last_layers: NN that generate the final output
	# list_of_weights: weights for regularization
	# beta: coefficient of regularization

	# output results:
	# cost: cost function
	# pred: predicted value (label)
	pred = Predict_Function(last_layers)
	regularizers = Regularize_Function(list_of_weights)
	cost = tf.reduce_mean(tf.square(pred-y_holder))
	cost = cost+beta*regularizers
	return cost,pred

def Optimizer(cost, learning_rate = 0.001, var_list = None ):
	# optimizer that optimizes TF-Variables (weights and biases) to minimize cost
	return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, var_list = var_list)

def Train_DNN(sess, saver, optimizer, pred, cost, x_holder, x_train, x_valid, x_test, y_holder, y_train,
		 y_valid, y_test, keep_prob, dropout_rate=1, path_to_save = "tmp/my_model.ckpt",
		training_epochs = 100, batch_size=64, display_step=1, accuracy_step = 10, save_step = 20, 
		save_output=True, output_fold = "output_data/"):
	# Train DNN

	# input parameters:
	# sess: TF session to perform the training
	# saver:  model saver
	# optimizer: cost function optimizer
	# pred: predicted result
	# cost: cost function
	# x_holder: input place-holder of NN
	# y_holder: output place-holder of NN
	# x_train(x_valid, x_test): training (validation, testing) data
	# y_train(y_valid, y_test): training (validation, testing) labels
	# keep_prob: dropout place-holder
	# dropout_rate
	# path_to_save
	# training_epochs
	# batch_size
	# display_step: print status info every display_step
	# accuracy_step: calculate the accuracy using validation set every accuracy_step
	# save_step: save model every save_step
	sess.run(tf.initialize_all_variables())		# initialize variables

	total_batch = int(x_train.shape[0]/batch_size)	# number of batches
	try:
		saver.restore(sess, path_to_save)	# load saved model
		print("[*]----------- start from restored result ------")
	except:
		print("[*]----------- start from initialization ------")

	cost_epoch = []
	for epoch in range(training_epochs):		# run for training_epoch
		avg_cost = 0.0
		list_sample = [ii for ii in range(x_train.shape[0])]
		random.shuffle(list_sample)
		
		for i in range(total_batch):		# run each batch
			# create batch i
	#		batch_x = x_train[i*batch_size:(i+1)*batch_size]
	#		batch_y = y_train[i*batch_size:(i+1)*batch_size]
			ibatch = list_sample[i*batch_size:(i+1)*batch_size]
			batch_x = x_train[ibatch]
			batch_y = y_train[ibatch]
#			print batch_x.shape, batch_y.shape,i,total_batch
#			sys.exit()
			# run optimizer, cost function and pred values based on current batch
			_, c, p = sess.run([optimizer, cost, pred],
				feed_dict={x_holder: batch_x,y_holder: batch_y, keep_prob:dropout_rate})
			avg_cost += c/total_batch	# calculate average cost function

		label_value, estimate = batch_y, p	# label and predicted label for the last batch
		cost_epoch.append(avg_cost)
		if (epoch+1) % display_step == 0:
			print ("Epoch: %04d, cost= %.9f" % (epoch+1,avg_cost))	# display the average cost
			print ("[*]----------------------------")
			# display the first 5 true and predicted labels in the last batch
			for i in range(5):
				print ("label value: %s\nestimated value: %s" %( label_value[i],estimate[i]))
				print ("[*]============================")
		if (epoch+1) % accuracy_step == 0:
			# calculate and display the cost of validation data
			accuracy = sess.run(cost,
				feed_dict={x_holder:x_valid, y_holder: y_valid, keep_prob:1})
			print ("[*]=============== Accuracy:%.14e =================="%( accuracy))
		if (epoch+1) % save_step == 0 or (epoch+1) == training_epochs:
			# save model at save_step and at the last step
			saver.save(sess, path_to_save)
			print ("model_saved to: "+path_to_save)
			print ("[*]_________________________")
	print ("Optimization Finished!")
	accuracy = sess.run(cost, feed_dict={x_holder:x_test, y_holder: y_test, keep_prob:1})
	print ("[*]============ Final Accuracy:%.14e =================="%( accuracy))
	if save_output:
		fig = plt.figure(0)
		plt.plot(cost_epoch, 'r--', label = 'cost')
		plt.legend(loc='upper right')
		plt.xlabel('step')
		plt.ylabel('cost')
		plt.title("cost")
		plt.grid(True)
		plt.savefig(output_fold+"train_cost.png")
		plt.close(fig)
		Save_Data(output_fold+"train_cost.txt",cost_epoch)
def Pred_DNN(sess, saver, pred, x_holder, x_data, keep_prob, keep_rate = 1,  path_to_save = "tmp/my_model.ckpt", load = False):
	if load:
		try:
			saver.restore(sess, path_to_save)
	#		print("[*]------ successfully load saved model ------")
		except:
			print("[*]------ error: can not load saved model ----------")
			sys.exit()
	return sess.run(pred, feed_dict={x_holder: x_data, keep_prob:1})

def Load_Variable(sess, saver,tf_weights, tf_biases, path_to_save="tmp/my_model.ckpt"):
	saver.restore(sess, path_to_save)
	py_weights = []
	for weight in tf_weights:
		py_weights.append(weight.eval())
	py_biases = []
	for bias in tf_biases:
		py_biases.append(bias.eval())
	return py_weights, py_biases	

def np_relu(xx):
        return np.maximum(xx, 0, xx)

def Recreate_NN(xx,py_weights,py_biases,DIM=[50,50,50,50,50]):
	n_layer = len(DIM) 
	if len(py_weights)!= n_layer+1 or  len(py_biases)!=n_layer+1:
		sys.exit("can not recreate NN because of different n_layer")
	layers = [np_relu(np.add(np.matmul(xx, py_weights[0]), py_biases[0]))]
	for ii in range(n_layer-1):
		layers.append(np_relu(np.add(np.matmul(layers[ii], py_weights[ii+1]), py_biases[ii+1])))
	return  np.add(np.matmul(layers[-1], py_weights[-1]),py_biases[-1])

def Plot_Pred_True(x1,y1,x2,y2,index,filename,title = 'predicted vs label',lty1='rs--',lty2='b^-.'):
	fig = plt.figure(index)
	plt.plot(x1,y1, lty1, label = 'predict')
	plt.plot( x2,y2, lty2, label = 'label')
	plt.legend(loc='upper right')
	plt.xlabel('samples')
	plt.ylabel('value')
	plt.title(title)
	plt.grid(True)
	plt.savefig(filename)
	plt.close(fig)


if __name__ == "__main__":
	path_to_save = "tmp/my_model.ckpt"
	N_Features = 24 
	N_Labels = 9
	input_fold = "input_data/"
	output_fold = "output_data/"

	X_train,Y_train,COLUMNS_ = Load_Prep_Data(input_fold+"train_set_preped.txt",N_Features,N_Labels)	
	X_valid,Y_valid,COLUMNS_ = Load_Prep_Data(input_fold+"valid_set_preped.txt",N_Features,N_Labels,COLUMNS_)
	X_test, Y_test, COLUMNS_ = Load_Prep_Data(input_fold+"test_set_preped.txt",N_Features,N_Labels, COLUMNS_)


	# =============================
	# ========= Create NN =========
	# =============================
	x_ = tf.placeholder("float64", [None, N_Features])
	y_ = tf.placeholder("float64", [None,N_Labels])
	keep_ = tf.placeholder(tf.float64)
	DIM_ =[50,50,50,50,50]
	embed_layer_, embed_coeff_ = Create_Embed_Layer(x_, N_Features)

	layers_, weights_, biases_ = Create_DNN_Model(embed_layer_, y_, keep_,n_input = N_Features,n_classes = N_Labels, DIM=DIM_)

	list_of_regular_ = [weights_]
	list_of_train_ = [weights_,biases_]
	list_of_regular_ = []	# do not use regularization
	cost_, pred_ = Cost_Function(y_,layers_,list_of_regular_,beta = 0.0)

	optimizer_ = Optimizer(cost_, learning_rate = 0.00005, var_list=list_of_train_ )
	saver_ = tf.train.Saver()

	config = tf.ConfigProto()
	config = tf.ConfigProto(device_count = {'GPU':0 })
	# ===========================
	# ===== train NN ============
	# ===========================

	with tf.Session(config=config) as sess0_:
		Train_DNN(sess0_, saver_, optimizer_, pred_, cost_,x_, X_train, X_valid, X_test,
			y_, Y_train, Y_valid, Y_test, keep_, dropout_rate = 1,path_to_save = path_to_save,
			training_epochs =100, batch_size=64, display_step=1, accuracy_step = 10, save_step = 50)

	with tf.Session(config=config) as sess1_:
		test_value = Pred_DNN(sess1_, saver_, pred_, x_, X_test, keep_, path_to_save = path_to_save, load = True)
	xcoor = range(100)
	Plot_Pred_True(xcoor,test_value[0:100],xcoor,Y_test[0:100],0,output_fold+"test_result.png")

	# ============================
	# ===== predicting ===========
	# ============================

	with tf.Session(config=config) as sess2_:
		for ii in range(N_Features):
			X_pred, Y_pred, COLUMNS_ = Load_Prep_Data(input_fold+"pred_set_preped"+str(ii)+".txt",N_Features,N_Labels, COLUMNS_)
			pred_value = Pred_DNN(sess2_, saver_, pred_, x_, X_pred, keep_, path_to_save = path_to_save, load = True)
			xcoor = range(len(Y_pred))
			Plot_Pred_True(xcoor,pred_value,xcoor,Y_pred,ii+1,output_fold+"pred_result"+str(ii)+".png",lty1='r--',lty2='b-.')
	# ==============================
	# ===== test recreate NN =======
	# ==============================
	with tf.Session(config=config) as sess3_:
		py_weights_, py_biases_ = Load_Variable(sess3_, saver_,weights_,biases_, path_to_save = path_to_save) 

	pred_recreate = Recreate_NN(X_pred,py_weights_,py_biases_,DIM=DIM_)
	diff =pred_recreate - pred_value
	print "======== prediction from recreate DNN using Numpy ============"
	print diff[:30,:]
	print "======== difference should close to 0 ========================"
