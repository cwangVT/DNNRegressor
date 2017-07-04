import math
import scipy.optimize
import random
import copy
# from DNN_model import np_relu

def y1(xxx):
	return  [xxx[ii+16] + (xxx[ii]+xxx[ii+8])**2/4 for ii in range(8)]

def y2(xxx):
	return  [sum([ (xxx[ii]+xxx[ii+8])**2/4.0 for ii in range(8)])+math.sin(-0.5*math.pi+4*sum([xxx[ii] for ii in range(8)]))]

def random_xx(amp):
	tmp = []
	for jj in range(N_Features):
		tt = 0
		while tt==0:
			tt = (random.random()-0.5)*amp[jj]
		tmp.append(tt)
	return tmp

def create_data(n_features, n_labels, n_data, amp, dataset=[]):
	if amp == None:
		amp = [1.0 for ii in range(n_features)]
	
	if dataset == []:
		for ii in range(n_data):
			xx = random_xx(amp) 
			xx= xx+y1(xx)
			xx= xx+y2(xx)
			dataset.append(xx)
	else:
		for jj in range(n_data):
			xx = dataset[jj][:n_features]
			xx= xx+y1(xx)
			xx= xx+y2(xx)
			dataset[jj]=xx
	return dataset

def create_file(filename, dataset, n_features, n_labels, n_data, head=None):
	if head == None:
		for ii in range(n_features):
			head = head+"x"+str(ii+1)+","
		for ii in range(n_labels):
			head = head+"y"+str(ii+1)+","
		head = head[:-1]+"\n"
	ftmp = open(filename,'w')
	ftmp.write(head)
	for ii in range(n_data):
		xx = dataset[ii]
		data = []
		for jj in range(n_features+n_labels):
			data.append(str( xx[jj]))
		dataline = ",".join(data)
		ftmp.write(dataline+"\n")
	ftmp.close()

	
if __name__ == "__main__":
	input_fold = "input_data/"
	N_Features = 24
	N_Labels = 9

	# set amp of each feature
	amp = [2 for ii in range(N_Features)]		# generate data in [-amp/2, amp/2]
	amp[16:24] = [20 for jj in range(16,24)]

	# set dataset size
	n_train_set = 100000
	n_valid_set = 5000
	n_test_set = 10000
	n_pred_set = 1000

	# set headline
	head = ""
	for ii in range(N_Features):
		head = head+"x"+str(ii+1)+","
	for ii in range(N_Labels):
		head = head+"y"+str(ii+1)+","
	head = head[:-1]+"\n"

	# create dataset files
	train_set = create_data(n_features=N_Features, n_labels = N_Labels, n_data=n_train_set, amp = amp)
	create_file(filename = input_fold+"train_set.txt", dataset = train_set, n_features=N_Features, 
		n_labels = N_Labels, n_data=n_train_set, head= head)

	valid_set = create_data(n_features=N_Features, n_labels = N_Labels, n_data=n_valid_set, amp = amp)
	create_file(filename = input_fold+"valid_set.txt", dataset = valid_set, n_features=N_Features, 
		n_labels = N_Labels, n_data=n_valid_set, head= head)

	test_set = create_data(n_features=N_Features, n_labels = N_Labels, n_data=n_test_set, amp = amp)
	create_file(filename = input_fold+"test_set.txt", dataset = test_set, n_features=N_Features,
		n_labels = N_Labels, n_data=n_test_set, head= head)

	# create prediction files	

	base_set = create_data(n_features=N_Features, n_labels = N_Labels, n_data=1, amp = amp)[0]
	for ii in range(N_Features):
		pred_set = []
		for jj in range(n_pred_set):
			tmp_set = copy.deepcopy(base_set)
			tmp_set[ii] = -0.5*amp[ii]+amp[ii]/float(n_pred_set)*jj
			pred_set.append(tmp_set)
		pred_set = create_data(n_features=N_Features, n_labels = N_Labels, n_data=n_pred_set, amp = amp,dataset=pred_set)
		create_file(filename = input_fold+"pred_set"+str(ii)+".txt", dataset = pred_set, 
			n_features=N_Features, n_labels = N_Labels, n_data=n_pred_set, head= head )

	"""
	f_data.close()

	#fpred.close()

	#minimal = scipy.optimize.basinhopping(yy,[0.0 for i in range(N_Features)])
	#print(minimal)
	"""
