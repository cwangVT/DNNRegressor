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
N_Features = 16 

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
# transfer from np DataFrame to array
X_train = X_train.values
Y_train = Y_train.values
X_test = X_test.values
Y_test = Y_test.values
X_valid = X_valid.values
Y_valid = Y_valid.values
X_pred = X_pred.values
Y_pred = Y_pred.values
# =============================================================
# ============== Set Parameters for NN ========================
# =============================================================
learning_rate = 0.0001
training_epochs = 100
batch_size = 64 
display_step = 1
accuracy_step = 10
save_step = 20
dropout_rate = 1.0	# keep rate, 1 means do not drop
beta = 0.00		# for L2 Regularization, 0 means no regularization

# Network Structure Parameters
n_hidden_1 = 50
n_hidden_2 = 50
n_hidden_3 = 50
n_hidden_4 = 50
n_hidden_5 = 50
n_input = N_Features	 	# number of input features
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

config = tf.ConfigProto()
# Do not use GPU
config = tf.ConfigProto(device_count = {'GPU':0 })

with tf.Session(config=config) as sess:
	# initialize variable
	#sess.run(tf.global_variables_initializer())
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
				feed_dict={x_holder: batch_x,
					y_holder: batch_y,
					keep_prob:dropout_rate})
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
		# ========== test the accuracy using validation data =====
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
#print([post_process(train_set_values[0][i],i) for i in range(N_Features+1)])

# save the testing data
Y_test = [[post_process(Y_test[ii][0],LABEL[0])] for ii in range(len(Y_test))]
test_vals = [[post_process(test_vals[ii][0],LABEL[0])] for ii in range(len(Y_test))]

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
Y_pred = [[post_process(Y_pred[ii][0],LABEL[0])] for ii in range(len(Y_pred))]
pred_vals = [[post_process(pred_vals[ii][0],LABEL[0])] for ii in range(len(pred_vals))]
plot_x = [[post_process(X_pred[ii][0],COLUMNS[0])] for ii in range(len(X_pred))]

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


