
import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt 
from scipy.special import expit
import time

train_x_doc = pd.read_csv("pa3_train_X.csv")
train_y_doc = pd.read_csv("pa3_train_y.csv")
dev_x_doc = pd.read_csv("pa3_dev_X.csv")
dev_y_doc = pd.read_csv("pa3_dev_y.csv")

training_setX = train_x_doc.copy()
training_setY=train_y_doc.copy()
dev_setX = dev_x_doc.copy()
dev_setY = dev_y_doc.copy()

training_setX = np.array (training_setX)
training_setY = np.array (training_setY)
dev_setX = np.array (dev_setX)
dev_setY = np.array (dev_setY)

def kernal_function (x,y,p):
    return np.power(x.dot(y.T)+1, p)

 
kernal_matrix1 = kernal_function(training_setX,training_setX,1)
kernal_matrix21 = kernal_function(training_setX,dev_setX,1)
kernal_matrix2 = kernal_function(training_setX,training_setX,2)
kernal_matrix22 = kernal_function(training_setX,dev_setX,2)
kernal_matrix3 = kernal_function(training_setX,training_setX,3)
kernal_matrix23 = kernal_function(training_setX,dev_setX,3)
kernal_matrix4 = kernal_function(training_setX,training_setX,4)
kernal_matrix24 = kernal_function(training_setX,dev_setX,4)
kernal_matrix5 = kernal_function(training_setX,training_setX,5)
kernal_matrix25 = kernal_function(training_setX,dev_setX,5)

r_count,c_count = training_setX.shape  #shape of training x
r_count2,c_count2 = dev_setX.shape     #shape of validation x
p =1
alpha = [0 for i in range (0,r_count)]
alpha = np.array (alpha)                         # ONE DIMENSION

training_setY = training_setY.reshape (r_count,) #one dimension
dev_setY = dev_setY.reshape (r_count2,)

Max_iteration = 100
Begin_iteration = 0
counter_s = 1

acc_array1 =[]
acc_array2 = []
acc_array3 = []
acc_array4 = []
acc_array5 = []

acc_array21 =[]
acc_array22 = []
acc_array23 = []
acc_array24 = []
acc_array25 = []

best_ac = 0
best_acc = 0
best_array = []
best_array1 = []

acc_array_dev = []
acc_array2_w_dev = []
sum_u = 0
sum_u2 = 0
sum_matrix = []
sumofalpha = 0
time1 =[]
time2=[]
time3=[]
time4=[]
time5=[]
acc_time = []
pre_time = 0

# Batch kernel perceptron

pre_time2 = 0
time_batch = []
acc2 = 0
acc_array = []
alpha = [0.0 for i in range (0,r_count)]
alpha = np.array(alpha)
alpha = alpha.reshape (r_count,)
training_setY = training_setY.reshape (r_count,)
learning_rate = 1
Begin_iteration = 0
pre_val = []
time_acc=[]
while (Begin_iteration < Max_iteration):
    acc = 0
    acc2 =0

    start = time.time ()
    u =np.sign (np.inner (alpha * training_setY, kernal_matrix1)*training_setY)
    for i in range (0, r_count):
        if (u[i] <=0):
            alpha[i] = alpha [i]+ learning_rate * 1
    end = time. time()
    u =np.sign (np.inner (alpha * training_setY, kernal_matrix1)*training_setY)
    for i in range (0,r_count):
        if u[i]> 0: 
            acc += 1

    u2 =np.sign (np.inner (alpha * training_setY, kernal_matrix21.T)*dev_setY)
    for i in range (0,r_count2):
        if u2[i]> 0: 
            acc2 += 1

    time_batch.append (end-start)
    for i in range (len(time_batch)):
        if i ==0:
            pre_time2 = time_batch[0]
        else:
            pre_time2 = sum (time_batch[0:i])
    time_acc.append(pre_time2)
    acc_array.append(acc/r_count)
    pre_val.append (acc2/r_count2)
    Begin_iteration +=1

print ("Training",acc_array)
print ("       ")
print ("       ")
print ("validation:",pre_val)

acc_learning_rate001 =[]
alpha = [0.0 for i in range (0,r_count)]
alpha = np.array(alpha)
alpha = alpha.reshape (r_count,)
training_setY = training_setY.reshape (r_count,)
learning_rate = 0.01
Begin_iteration = 0
while (Begin_iteration < Max_iteration):
    acc = 0
    u =np.sign (np.inner (alpha * training_setY, kernal_matrix1)*training_setY)
    for i in range (0, r_count):
        if (u[i] <=0):
            alpha[i] = alpha [i]+ learning_rate * 1
    u =np.sign (np.inner (alpha * training_setY, kernal_matrix1)*training_setY)
    for i in range (0,r_count):
        if u[i]> 0: 
            acc += 1
    acc_learning_rate001.append(acc/r_count)
    Begin_iteration +=1

#plt.title("Speed")
#plt.plot (time_acc, label = 'batch (training)')
#plt.xlabel('iteration')
#plt.ylabel('time')
#plt.plot (acc_array, label = 'Kernal perceptron P=1(training (Learning_rate =1))')
#plt.plot (pre_val, label = 'Kernal perceptron P=1(valiation (Learning_rate =1))')
#plt.plot (acc_learning_rate001, label = 'Kernal perceptron P=1(training (Learning_rate =0.01))')
#plt.plot (acc_array1, label = 'Kernal perceptron P=1(training)')
#plt.plot (acc_array21, label = 'Kernal perceptron P=1(validation)')
#plt.plot (acc_array2, label = 'Kernal perceptron P=2(training)')
#plt.plot (acc_array22, label = 'Kernal perceptron P=2(validation)')
#plt.plot (acc_array3, label = 'Kernal perceptron P=3(training)')
#plt.plot (acc_array23, label = 'Kernal perceptron P=3(validation)')
#plt.plot (acc_array4, label = 'Kernal perceptron P=4(training)')
#plt.plot (acc_array24, label = 'Kernal perceptron P=4(validation)')
#plt.plot (acc_array5, label = 'Kernal perceptron P=5(training)')
#plt.plot (acc_array25, label = 'Kernal perceptron P=5(validation)')
#plt.title("Best_Accuracy")
#plt.plot (best_array, label = 'best accuracy (training)')
#plt.plot (best_array1, label = 'best accuracy (validation)')
#plt.xlabel('P')
#plt.ylabel('acc')
#plt.xlabel('iterations')
#plt.ylabel('acc')
#plt.title("Time for P =1")
#plt.plot (acc_time, label = 'P=1(training)')
#plt.xlabel('Iteration')
#plt.ylabel('Time')
#plt.plot (time2, label = 'P=2(training)')
#plt.plot (time3, label = 'P=3(training)')
#plt.plot (time4, label = 'P=4(training)')
#plt.plot (time5, label = 'P=5(training)')
plt.legend()
plt.show()




