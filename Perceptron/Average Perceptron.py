
import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt 
from scipy.special import expit

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

r_count,c_count = training_setX.shape
r_count2,c_count2 = dev_setX.shape

w = np.zeros([c_count,1])
w_average = w = np.zeros([c_count,1])

Max_iteration = 100
Begin_iteration = 0
counter_s = 1

acc_array =[]
acc_array2 = []
acc_array_dev = []
acc_array2_w_dev = []
best_acc = 0
best_iteration = 0
w_matrix = []
w_average_matrix = []

while (Begin_iteration < Max_iteration):

    acc_w = 0
    acc_average_w = 0
    acc_w_2 =0
    acc_average_w_2= 0
    for i in range (0,r_count):
       condition = training_setY [i] * np.dot(w.T,training_setX[i])
       if (condition[0] <= 0):
           w = w + (np.multiply(training_setX[i] , training_setY[i]).reshape ([c_count,1]))
    
       w_average = (counter_s * w_average + w )/(counter_s+1)
       counter_s = counter_s +1 
       if (i == r_count -1):
           w_matrix.append (w)
           w_average_matrix.append(w_average)

    y_prediction_1 = np.dot (training_setX,w)
    y_prediction_2 = np.dot (training_setX,w_average)
    dev_y_predict_1 =np.dot (dev_setX,w)
    dev_y_predict_2 = np.dot (dev_setX,w_average)

    for i in range (0,r_count):
        if (y_prediction_1[i] * training_setY[i]>0):
            acc_w +=1
        if(y_prediction_2[i] * training_setY[i]>0):
            acc_average_w +=1
    for i in range (0,r_count2):
        if (dev_y_predict_1[i] * dev_setY[i]>0):
            acc_w_2 +=1
        if(dev_y_predict_2[i] * dev_setY[i]>0):
            acc_average_w_2 +=1            
    if best_acc < acc_average_w_2:
        best_acc = acc_average_w_2
        best_iteration = Begin_iteration + 1
    acc_array.append (acc_w/r_count)
    acc_array2.append (acc_average_w/r_count)

    acc_array_dev.append (acc_w_2/r_count2)
    acc_array2_w_dev.append (acc_average_w_2/r_count2)

    Begin_iteration += 1


print ("The best accuracy of validation set:", best_acc/ r_count2)
print ("The best iteration of validation set:", best_iteration)

plt.title("Accuracy  of Average perceptron (Training v.s Validation)")
plt.plot (acc_array2, label = 'Average perceptron (training)')
plt.plot (acc_array2_w_dev, label = 'Average perceptron (validation)')
plt.xlabel('iterations')
plt.ylabel('acc')
plt.legend()
plt.show()




