# DNNRegressor
Create and train a deep neural network regressor to fit a complex target function. Then create an optimizer to find the optimized input with minimal target function.

Random Dataset Creater (Python)

	create_data.py: create polynomial and sin function of continuous variables.

	create_discrete_data.py: create polynomial and sin function of discrete variables.

Data preprocessing & Deep Neural Network Regressor (Tensorflow)

	API_DNN_regressor.py: Data preprocessing and create a deep neural network with tf.contrib.learn.DNNRegressor API

	DNN_regressor.py:Data preprocessing and create a deep neural network within developer level (create the network manually)

Input Optimizer (Scipy, Numpy)

	optimizer: find optimized input that generate the smallest target function (namely the predict function of trained DNN)
