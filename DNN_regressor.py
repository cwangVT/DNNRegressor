from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
pre_post_process = True

# set the verbosity mode to show detailed info during train
#tf.logging.set_verbosity(tf.logging.INFO)

# ============================================================================
# ====================== Load Data ===========================================
# ============================================================================

# Set the number of features
N_Features = 20 

# Set the names of the features, label
FEATURES = ["x"+str(i+1) for i in range(N_Features) ]
LABEL = "y"
COLUMNS = FEATURES+[LABEL]

# load the train dataset from csv file.
# here read from train_set.txt, skip leading space
# skip first row, and use COLUMNS to set the names of columes
train_set = pd.read_csv(input_fold+"train_set.txt", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)

# load the validation data from csv file
valid_set = pd.read_csv(input_fold+"valid_set.txt", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)

# load the test dataset from csv file.
test_set = pd.read_csv(input_fold+"test_set.txt", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)

# load the predict dataset from csv file.
pred_set = pd.read_csv(input_fold+"predict_set.txt", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)

# get the size of each dataset
n_sample = train_set.shape[0]
n_test = test_set.shape[0]
n_valid = valid_set.shape[0]
n_pred = pred_set.shape[0]

# change format fromr pd-dataframe to python array
train_set = [[train_set[kk].values[ii] for kk in COLUMNS ]
        for ii in range(n_sample)]
valid_set =  [[valid_set[kk].values[ii] for kk in COLUMNS]
        for ii in range(n_valid)]
test_set = [[test_set[kk].values[ii] for kk in COLUMNS]
        for ii in range(n_test)]
pred_set = [[pred_set[kk].values[ii] for kk in COLUMNS]
        for ii in range(n_pred)]

# =============================================================
# ========= perform normalization (preprocess)=================
# =============================================================
# after processing: x_norm = [x - mean(x)]/[max(x)-min(x)]

if pre_post_process:
	means=[0 for ii in range(N_Features+1)]
	maxx =[-float('Inf') for ii in range(N_Features+1)]
	minn =[ float('Inf') for ii in range(N_Features+1)]
	maxmin = [0 for ii in range(N_Features+1)]
	for ii in range(n_sample):
		for jj in range(N_Features+1):
			means[jj]+= train_set[ii][jj]/n_sample
			maxx[jj] = max(maxx[jj],train_set[ii][jj])
			minn[jj] = min(minn[jj],train_set[ii][jj])

	for jj in range(N_Features+1):
		maxmin[jj] = maxx[jj]-minn[jj]
		if maxmin[jj] == 0:
			print( "============= feature "+str(jj)+" never change ======")

def pre_process(xx,index):
	if pre_post_process:
		xx-=means[index]	
		if maxmin[index] != 0:
			xx/=maxmin[index]
	return xx
# print original data
# print(train_set[0])
train_set = [[pre_process(train_set[ii][kk],kk) for kk in range(N_Features+1)]
	for ii in range(n_sample)]
valid_set = [[pre_process(valid_set[ii][kk],kk) for kk in range(N_Features+1)]
	for ii in range(n_valid)]
test_set =  [[pre_process(test_set[ii][kk],kk) for kk in range(N_Features+1)]
	for ii in range(n_test)]
pred_set =  [[pre_process(pred_set[ii][kk],kk) for kk in range(N_Features+1)]
	for ii in range(n_pred)]

# print pre-processed data, check the pre-process function
#print(train_set[0])
# Separate feature X and label Y
X_train = [[train_set[ii][kk] for kk in range(N_Features)] 
	for ii in range(n_sample)]
Y_train = [[train_set[ii][N_Features]] for ii in range(n_sample)]

X_valid = [[valid_set[ii][kk] for kk in range(N_Features)]
	for ii in range(n_valid)]
Y_valid = [[valid_set[ii][N_Features]] for ii in range(n_valid)]

X_test = [[test_set[ii][kk] for kk in range(N_Features)]
        for ii in range(n_test)]
Y_test = [[test_set[ii][N_Features]] for ii in range(n_test)]

X_pred = [[pred_set[ii][kk] for kk in range(N_Features)]
        for ii in range(n_pred)]
Y_pred = [[pred_set[ii][N_Features]] for ii in range(n_pred)]

# =============================================================
# ============== Set Parameters for NN ========================
# =============================================================
learning_rate = 0.0001
training_epochs = 1000
batch_size = 10
display_step = 1
accuracy_step = 10
save_step = 20
dropout_rate = 1	# keep rate, 1 means do not drop
beta = 0.0		# for L2 Regularization, 0 means no regularization
path_to_save = "tmp/my_model.ckpt"

# Network Structure Parameters
n_hidden_1 = 50
n_hidden_2 = 50
n_hidden_3 = 50
n_hidden_4 = 50
n_hidden_5 = 50
n_input = len(FEATURES) 	# number of input features
n_classes = 1			# for regressin, only one output class

# ============================================================================
# ====================== Create NN ===========================================
# ============================================================================

# NN input placeholders, training data will feed to them
x_holder = tf.placeholder("float64", [None, n_input])
y_holder = tf.placeholder("float64", [None,1])
keep_prob = tf.placeholder(tf.float64)

# Create the DNN regressor framework
def multilayer_perceptron(x_holder, weights, biases,keep_prob):
	layer_1 = tf.nn.relu(tf.add(tf.matmul(x_holder, weights['h1']), biases['b1']))
	layer_1 = tf.nn.dropout(layer_1, keep_prob)

	layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
	layer_2 = tf.nn.dropout(layer_2, keep_prob)

	layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
	layer_3 = tf.nn.dropout(layer_3, keep_prob)

	layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4']))
	layer_4 = tf.nn.dropout(layer_4, keep_prob)

	layer_5 = tf.nn.relu(tf.add(tf.matmul(layer_4, weights['h5']), biases['b5']))
	layer_5 = tf.nn.dropout(layer_5, keep_prob)

	out_layer = tf.add(tf.matmul(layer_5, weights['out']),biases['out'])
	return out_layer

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

# Set the predicted value of NN
pred = multilayer_perceptron(x_holder, weights, biases,keep_prob)

# Set L2 Regularization
regularizers = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + \
		tf.nn.l2_loss(weights['h3']) + tf.nn.l2_loss(weights['h4']) + \
		tf.nn.l2_loss(weights['h5']) + tf.nn.l2_loss(weights['out'])

# Set loss function
cost = tf.reduce_mean(tf.square(pred-y_holder))
cost = cost+beta*regularizers

# Set optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Set saver to save models and variables
saver = tf.train.Saver()

# ========================================================================
# =============== Run the DNN ============================================
# ========================================================================

with tf.Session() as sess:
	# initialize variable
	sess.run(tf.initialize_all_variables())
	try:
		saver.restore(sess, path_to_save)
		print("[*]---------------- start from restored result ------")
	except:
		print("[*]---------------- start from initialization ------")
	for epoch in range(training_epochs):
	# ===============================================
	# ============== One Training Epoch =============
	# ===============================================
		avg_cost = 0.0
		total_batch = int(n_sample/batch_size)
		for i in range(total_batch-1):
		# ======================================
		# ======== One Training Batches ========
		# ======================================
			batch_x = X_train[i*batch_size:(i+1)*batch_size]
			batch_y = Y_train[i*batch_size:(i+1)*batch_size]
			# ===================================
			# ===== Key Work: Optimization ======
			# ===================================
			_, c, p = sess.run([optimizer, cost, pred], 
				feed_dict={x_holder: batch_x,y_holder: batch_y, keep_prob:dropout_rate})
			# aver loss
			avg_cost += c/total_batch
		# ========================================
		# testing/validation during each epoch ===
		# ========================================
		# prediction with the last batch
		label_value = batch_y
		estimate = p
		err = label_value - estimate
		print ("num batch:", total_batch)

# ================ Debug/check point ===============================
#	should make sure that pred, y have exactly the same shape 
#		print(tuple(pred.get_shape().as_list()))
#		print(tuple(y.get_shape().as_list()))
#		print(tuple(regularizers.get_shape().as_list()))
#		print(tuple(cost.get_shape().as_list()))
# ==================================================================
		# print info 
		if (epoch+1) % display_step == 0:
			# print average cost 
			print ("Epoch:", '%04d' % (epoch+1), "cost=", \
				"{:.9f}".format(avg_cost))
			print ("[*]----------------------------")
			# print the predicted result for the
			# first 3 samples in train dataset
			for i in xrange(3):
				print ("label value:", label_value[i], \
					"estimated value:", estimate[i])
				print ("[*]============================")
		if (epoch+1) % accuracy_step == 0:
		# ========================================================
		# ============= test the accuracy use validation data ====
		# ========================================================
			accuracy = sess.run(cost, 
				feed_dict={x_holder:X_valid, y_holder: Y_valid, keep_prob:1})
			print ("Accuracy:", accuracy)
		if (epoch+1) % save_step == 0 or (epoch+1) == training_epochs:
			saver.save(sess, path_to_save)
			print ("model_saved to: "+path_to_save)
			print ("[*]_________________________")
	print ("Optimization Finished!")
	# ===================================
	# ==== end of training ==============
	# ===== start testing ===============
	# ===================================
	accuracy = sess.run(cost, feed_dict={x_holder:X_test, y_holder: Y_test, keep_prob:1})
	print ("Accuracy:", accuracy)
	test_vals = sess.run(pred, feed_dict={x_holder: X_test, keep_prob:1})
	# ==================================
	# ========== predicting ============
	# ==================================
	pred_vals = sess.run(pred, feed_dict={x_holder: X_pred, keep_prob:1})

# ==================================================================
# ====================== end of NN =================================
# ==================================================================

# post-process function
def post_process(xx,index):
	if pre_post_process:
		return xx*maxmin[index]+means[index]
	else:
		return xx 
# print post processed data, check post-process function
#print([post_process(train_set[0][i],i) for i in range(N_Features+1)])

# save the testing data
Y_test = [[post_process(Y_test[ii][0],N_Features)] for ii in range(len(Y_test))]
test_vals = [[post_process(test_vals[ii][0],N_Features)] for ii in range(len(Y_test))]

ftmp= open(output_fold+"test_result.txt","w")
for ii in range(len(test_vals)):
	ftmp.write(str(test_vals[ii][0])+"\t"+str(Y_test[ii][0])+"\n")
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
Y_pred = [[post_process(Y_pred[ii][0],N_Features)] for ii in range(len(Y_pred))]
pred_vals = [[post_process(pred_vals[ii][0],N_Features)] for ii in range(len(pred_vals))]
plot_x = [[post_process(X_pred[ii][0],0)] for ii in range(len(X_pred))]

ftmp= open(output_fold+"pred_result.txt","w")
for ii in range(len(pred_vals)):
        ftmp.write(str(pred_vals[ii][0])+"\t"+str(Y_pred[ii][0])+"\n")
ftmp.close()
# plotting predicted vs labeled for testing dataset

plt.figure(1)
plt.plot(plot_x,Y_pred, 'r--', plot_x,pred_vals, 'b-.')
plt.xlabel('samples')
plt.ylabel('value')
plt.title('predicted(blue) vs label(red)')
plt.grid(True)
plt.savefig(output_fold+"pred_result.png")



# ======================================================================
# =================== Simulating Annealing =============================
# ======================================================================

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

def np_relu(xx):
	return np.maximum(xx, 0, xx)

def nn_predict_as_np(xx):
	if True:
                pred_layer_1 = np_relu(np.add(np.matmul(xx, hh1), bb1))
                pred_layer_2 = np_relu(np.add(np.matmul(pred_layer_1, hh2), bb2))
                pred_layer_3 = np_relu(np.add(np.matmul(pred_layer_2, hh3), bb3))
                pred_layer_4 = np_relu(np.add(np.matmul(pred_layer_3, hh4), bb4))
                pred_layer_5 = np_relu(np.add(np.matmul(pred_layer_4, hh5), bb5))
                pred_out_layer = np.add(np.matmul(pred_layer_5, hout),bout)
                return pred_out_layer
	
initial_guess =[[0. for i in range(N_Features)]]# [[tf.cast(0.0,tf.float64) for i in range(N_Features)]]
#initial_guess = [[-4.07251505153,12.2169885084, 8.93008115044,12.4504141069,-0.286904094387, 11.5846259272,2.69020332846,0.660371587514,-4.10793877444,-10.6211903571]]
initial_guess = [[(initial_guess[0][ii]-means[ii])/maxmin[ii] for ii in range(N_Features)]]
print("=======",post_process(nn_predict_as_np(initial_guess),N_Features))

xmin = [-0.5 for i in range(N_Features)]
xmax = [0.5 for i in range(N_Features)]
bounds = [(low, high) for low, high in zip(xmin, xmax)]

minimal = scipy.optimize.minimize(nn_predict_as_np,
                initial_guess,method="L-BFGS-B", jac=False, bounds=bounds, options={'disp':True,'factr':0,'ftol':0,'gtol':0.0001})

#print(minimal)
print("minimal Y: ", post_process(minimal['fun'][0],N_Features))
#print(minimal['x'])
print("minimal X: ", [post_process(minimal['x'][ii],ii) for ii in range(N_Features)])


minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds, jac = False)
minimal = scipy.optimize.basinhopping(nn_predict_as_np,initial_guess,
                 minimizer_kwargs=minimizer_kwargs,niter=20, disp=True)

print(minimal)
print("minimal Y: ", post_process(minimal['fun'],N_Features))
print("minimal X: ", [post_process(minimal['x'][ii],ii) for ii in range(N_Features)])
