import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt 
from scipy.special import expit

def normalized(x):
    return (x - x.min()) / (x.max() - x.min())

def second_normalized(x,y,z):
    return (x-z)/(y-z)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

train_x_doc = pd.read_csv("pa2_train_X.csv")
train_y_doc = pd.read_csv("pa2_train_y.csv")
dev_x_doc = pd.read_csv("pa2_dev_X.csv")
dev_y_doc = pd.read_csv("pa2_dev_y.csv")

training_setX = train_x_doc.copy()
training_setY=train_y_doc.copy()
dev_setX = dev_x_doc.copy()
dev_setY = dev_y_doc.copy()

max_age = training_setX['Age'].max()
min_age = training_setX['Age'].min()
max_an = training_setX['Annual_Premium'].max()
min_an = training_setX['Annual_Premium'].min()
max_vin = training_setX['Vintage'].max()
min_vin = training_setX['Vintage'].min()

normalize_column = ['Age', 'Annual_Premium', 'Vintage']
training_setX[normalize_column] = training_setX[normalize_column].apply(normalized)

dev_setX['Age'] = second_normalized (dev_setX['Age'],max_age,min_age)
dev_setX['Annual_Premium'] = second_normalized (dev_setX['Annual_Premium'],max_an,min_an)
dev_setX['Vintage'] = second_normalized (dev_setX['Vintage'],max_vin,min_vin)

learning_rate = 0.01
iterations = 2500
regu_par = [1.,0.1,0.01,0.001,0.0001,0.00001]
r_count,c_count = training_setX.shape

y = training_setY

acc_array0 = []
acc_array1 = []
acc_array2 = []
acc_array3 = []
acc_array4 = []
acc_array5 = []
acc_compare = 0
best_acc=0
best_w =[]
best_w_vector =[]
acc_vector1 = []

k = 0
index_k=0

for parameter in regu_par:
    w = np.zeros([c_count,1])
    while (k<iterations):
        
        gradient = learning_rate / r_count * np.sum (np.dot(training_setX.T,(y-(sigmoid (np.dot (training_setX,w))))),1)
        gradient = gradient.reshape ([197,1])
        w = w + gradient
        for j in range (0,len(w)-1):
            w[j] = w[j] - learning_rate*regu_par[index_k]*w[j]
        acc = 0
        y_predict = np.round(sigmoid (np.dot (training_setX,w)))
        result = y-y_predict
        result = np.array (result)
        for i in range (0,len(result)):
            if result[i]==0:
                acc +=1
        if (best_acc < acc/len(result)):
            best_acc = acc/len(result)
            best_w = w

        if index_k == 0:
            acc_array0.append(acc/len(result))
        elif index_k == 1:
            acc_array1.append(acc/len(result))
        elif index_k == 2:
            acc_array2.append(acc/len(result))
        elif index_k == 3:
            acc_array3.append(acc/len(result))
        elif index_k == 4:
            acc_array4.append(acc/len(result))
        else:
            acc_array5.append(acc/len(result))
        #print (acc/len(result))
        k+=1
    k=0
    best_w_vector.append (best_w)
    acc_vector1.append (best_acc)
    index_k +=1
    best_acc = 0

acc =0
dev_acc_array =[]
valid_best_acc= 0
for k2 in range (0,len(regu_par)):
    validation_predict = np.round (sigmoid (np.dot(dev_setX,best_w_vector[k2])))
    validation_result = dev_setY-validation_predict
    validation_result = np.array(validation_result)
    for i in range (0,len(validation_result)):
        if validation_result[i]==0:
              acc +=1
    if valid_best_acc < acc:
        valid_best_acc = acc
    dev_acc_array.append(acc/len(validation_result))
    acc = 0
      

print ("Start draw the graph")
print ("Best_w_vecotr:",best_w_vector)
print ("Training:", acc_vector1)
print ("Validation:",dev_acc_array)
iteration_axis = [num for num in range (1,11)]

plt.title("Accuracy v.s Regulariztion parameter  (learning rate =0.01, iterations =5000) ")
plt.plot (acc_vector1, label = 'train_set', marker = 'o')
plt.plot (dev_acc_array, label = 'validation_set',  marker = 'x')
plt.xlabel('Regulariztion parameter (10^(-x))')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
