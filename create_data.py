import math
import scipy.optimize
import random

input_fold = "input_data/"
N_Features = 20 
amp = 20
coeff = [random.random() for i in range(N_Features)]
power = [random.choice([2,2,2]) for i in range(N_Features)]
bias =  [(random.random()-0.5)*amp/2 for i in range(N_Features)]
coeff.append(None)
power.append(None)
bias.append(None)
def yy(xxx):
    result = 0
    for ii in range(N_Features):
	#result += coeff[ii]*(xxx[ii]-bias[ii])**power[ii]
	#result += coeff[ii]*(xxx[ii]-bias[ii]-0.1*amp*math.sin(xxx[ii]))**power[ii]
	result += coeff[ii]*(xxx[ii]-bias[ii])**power[ii]-amp*math.sin(xxx[ii])
    return result
    #return (xxx-)**2+(xx2+5)**2+(xx3-5)**2
    #return math.sqrt(xx1*xx1+xx2*xx2)
    #return xx1*xx2
    #return 100*math.sin(xx1+xx2)
    #return xx1*math.sin(xx2)
    #return xx1*xx2-math.sin(xx1)-math.cos(xx2)

ftmp = open(input_fold+"train_set.txt","w")
fvalid = open(input_fold+"valid_set.txt","w")
ftest = open(input_fold+"test_set.txt","w")
fpred = open(input_fold+"predict_set.txt","w")

head = ""
for ii in range(N_Features):
    head = head+"x"+str(ii+1)+","
head = head[:-1]
ftest.write(head + ",y\n")
ftmp.write(head + ",y\n")
fvalid.write(head + ",y\n")
fpred.write(head + ",y\n")

n_train_set = 100000
n_valid_set = 10000
n_test_set = 10000
n_pred_set = 10000
train_set = []
test_set= []


def random_xx():
	tmp = []
	for jj in range(N_Features):
		tt = 0
		while tt==0:
			tt = (random.random()-0.5)*amp
		tmp.append(tt)
	return tmp

for ii in range(n_train_set):
        xx = random_xx()
	xx.append(yy(xx))
	data = []
	for jj in range(N_Features+1):
		data.append(str( xx[jj]))
	dataline = ",".join(data)
        ftmp.write(dataline+"\n")
ftmp.close()

for ii in range(n_valid_set):
	xx = random_xx()
	xx.append(yy(xx))
	data = []
	for jj in range(N_Features+1):
		data.append(str(xx[jj]))

        dataline = ",".join(data)
        fvalid.write(dataline+"\n")
fvalid.close()

for ii in range(n_test_set):
	xx = random_xx()
	xx.append(yy(xx))
	data = []
	for jj in range(N_Features+1):
		data.append(str(xx[jj]))
	dataline = ",".join(data)
	ftest.write(dataline+"\n")
ftest.close()

xx0=[-0.5*amp for ii in range(N_Features)]
step_x = float(amp)/n_pred_set
for ii in range(n_pred_set):
        xx = [item+step_x*ii for item in xx0]
        if xx[0]==0:
                xx = [step_x for item in xx0]
        xx.append(yy(xx))
        data = []
        for jj in range(N_Features+1):
		data.append(str(xx[jj]))
        dataline = ",".join(data)
        fpred.write(dataline+"\n")
fpred.close()

f_data=open(input_fold+"norm_data.txt","w")
f_data.write("coeff,power,bias,ave, max, min\n")
for ii in range(N_Features):
	f_data.write(str(coeff[ii])+",")
	f_data.write(str(power[ii])+",")
	f_data.write(str(bias[ii])+",")
	f_data.write("\n")
f_data.close()

#fpred.close()

#minimal = scipy.optimize.basinhopping(yy,[0.0 for i in range(N_Features)])
#print(minimal)
