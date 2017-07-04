# DNNRegressor
Create and train a deep neural network regressor to fit a complex target function. 
Then create an optimizer to find the optimized input with minimal target function.

Random Dataset Creater (Python)

	Create_Simulate_Data.py: create testing dataset.

	Prep_data.py: Preprocessing on Raw dataset.

Deep Neural Network Regressor (Tensorflow)

	DNN_model.py: Create a deep neural network within developer level (create the network manually)

	RNN_model.py: perform time series predicting using DNN. Similar to RNN

Input Optimizer (Scipy, Numpy)

	SA_model: find optimized input that generate the smallest target function (namely the predict function of trained DNN) using simulating annealing
