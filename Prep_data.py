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

def Pre_Process(xx,index,means,maxmin):
	xx-=means[index]
	if maxmin[index] != 0:
		xx/=maxmin[index]
	return xx

def Post_Process(xx,index,means,maxmin):
	return xx*maxmin[index]+means[index]

def Load_Raw_Data(filename,nf,nl,COLUMNS=None):
        # Load dataset from csv files

        # input parameters:
        # filename: path and name of files
        # nf: number of Features
        # nl: number of Labels
        # COLUMNS: name of Features

        # output result:
        # dataset: dataset in pandas framework 
	# COLUMNS: name of Features

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

        return dataset, COLUMNS

def Calcul_Process_Param(dataset,COLUMNS):
	means = dataset.mean().to_dict()
	maxx = dataset.max().to_dict()
	minn = dataset.min().to_dict()
	maxmin = {}
	for jj in COLUMNS:
		maxmin[jj] = maxx[jj]-minn[jj]
		if maxmin[jj] == 0:
			print( "======= Feature "+jj+" never change")
	return means, maxmin

def Create_Prep_Data(dataset, COLUMNS, n_data, means, maxmin):
	dataset =pd.DataFrame([[Pre_Process(dataset[kk][ii],kk,means,maxmin)
		for kk in COLUMNS] for ii in range(n_data)],columns =COLUMNS)
	return dataset

def Create_Prep_File(filename,dataset):
	dataset.to_csv(filename, index=False)

def Save_Process_Param(filename,COLUMNS,means,maxmin):
	f_proc = open(filename,"w")
	f_proc.write("feature,mean,maxmin\n")
	for ii in COLUMNS:
		f_proc.write(ii+","+str(means[ii])+","+str(maxmin[ii])+"\n")
	f_proc.close()

def Load_Process_Param(filename,COLUMNS):
	f_proc = open(filename,"r")
	f_proc.readline()       # skip the title line
	d1 = {}
	d2 = {}
	for ff in f_proc:
		tt = ff.strip().split(",")
		d1[tt[0]]=float(tt[1])		# means
		d2[tt[0]]=float(tt[2])		# maxmin
	f_proc.close()
	means =[]
	maxmin = []
	for item in COLUMNS:
		means.append(d1[item])
		maxmin.append(d2[item])
	return means, maxmin

def Save_Data(filename,data):
	if all(isinstance(elem, float) or isinstance(elem,int) for elem in data):
	        ftmp = open(filename,'w')
		for item in data:
			ftmp.write(str(item))
			ftmp.write("\n")
        	ftmp.close()
	elif all(isinstance(elem, list) for elem in data) and  all(isinstance(subelem, float) \
			or isinstance(subelem,int) for elem in data):
		ftmp = open(filename,'w')
		for item in data:
			ftmp.write(str(item)[1:-1])
			ftmp.write("\n")
		ftmp.close()
	else:
		sys.exit("can not save data")

if __name__ == "__main__":
	# ==============================================================================
	# ====================== Set Environment Parameters ============================
	# ==============================================================================
	input_fold = "input_data/"
	output_fold = "output_data/"
	path_to_save = "tmp/my_model.ckpt"
	pre_post_process = True


	# ============================================================================
	# ====================== Load Raw Data =======================================
	# ============================================================================

	# Set the number of features
	N_Features = 24 
	N_Labels = 9

	# load the train dataset from csv file
	# here assuming that the csv file has head line
	train_set,COLUMNS_ = Load_Raw_Data(input_fold+"train_set.txt",N_Features,N_Labels) 
	valid_set,COLUMNS_ = Load_Raw_Data(input_fold+"valid_set.txt",N_Features,N_Labels,COLUMNS_)
	test_set,COLUMNS_ = Load_Raw_Data(input_fold+"test_set.txt",N_Features,N_Labels,COLUMNS_)

	# get the size of each dataset
	n_sample = train_set.shape[0]
	n_test = test_set.shape[0]
	n_valid = valid_set.shape[0]

	# =============================================================
	# ================ perform preprocess =========================
	# =============================================================
	# after processing: x_norm = [x - mean(x)]/[max(x)-min(x)]
	means_, maxmin_ = Calcul_Process_Param(train_set,COLUMNS_)
	# saving processing parameters
	Save_Process_Param(input_fold+"preprocess.txt",COLUMNS=COLUMNS_,means=means_,maxmin=maxmin_)

	# preforming pre-processing	
	train_set = Create_Prep_Data(dataset=train_set,COLUMNS = COLUMNS_,n_data= n_sample,
		means = means_, maxmin = maxmin_)
	test_set = Create_Prep_Data(dataset=test_set,COLUMNS = COLUMNS_,n_data= n_test,
		means = means_, maxmin = maxmin_)
	valid_set = Create_Prep_Data(dataset= valid_set,COLUMNS = COLUMNS_,n_data= n_valid,
		means = means_, maxmin = maxmin_)

	Create_Prep_File(input_fold+"train_set_preped.txt",train_set)
	Create_Prep_File(input_fold+"valid_set_preped.txt",valid_set)
	Create_Prep_File(input_fold+"test_set_preped.txt",test_set)

	# saving processing parameters
	# load the predict dataset from csv file.
	for _f in range(N_Features):
		pred_set,COLUMNS_ = Load_Raw_Data(input_fold+"pred_set"+str(_f)+".txt",N_Features,N_Labels)

		n_pred = pred_set.shape[0]
		pred_set = Create_Prep_Data(dataset= pred_set,COLUMNS = COLUMNS_,n_data= n_pred,
			means = means_, maxmin = maxmin_)
		Create_Prep_File(input_fold+"pred_set_preped"+str(_f)+".txt",pred_set)
