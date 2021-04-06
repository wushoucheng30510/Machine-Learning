print ("Linear Regression")
print ("==========================")

import numpy as np
import pandas as pd                  #Grabing and Transforming data
import matplotlib.pyplot as ansis    #plot data to let us explore it
import math
import datetime as dt
import random

price_prediction = pd.read_csv('PA1_train.csv')
price_prediction = price_prediction.drop (['id'], axis=1)
priceend = price_prediction ['price']

today = dt.date.today()
month = []
day = []
year = []
i=0
intdummy = []
longdays = []
int_feature=[]
count_zipcode = 0
count_zipcode_matrix = []
count_water=0
count_water_matrix=[]
normalized_matrix=[]
w_vectors = [ 0.5 for i in range(1,20)]
w_vector = w_vectors.copy()
iteration_matrix = []
iterations = 0
output_int = 0
w_vector_output=[]
convergence_matrix = []
mse_matrix=[]
numbreaks = 0

ori_dummy = price_prediction ['dummy']
pastdate = price_prediction ['date']
ori_bed = price_prediction ['bedrooms']
ori_bat = price_prediction ['bathrooms']
ori_sqft_living= price_prediction ['sqft_living']
ori_sqft_lot= price_prediction ['sqft_lot']
ori_floors= price_prediction ['floors']
ori_waterfront= price_prediction ['waterfront']
ori_view= price_prediction ['view']
ori_condition= price_prediction ['condition']
ori_grade= price_prediction ['grade']
ori_sqft_above = price_prediction ['sqft_above']
ori_sqft_basement = price_prediction ['sqft_basement']
ori_yr_built= price_prediction ['yr_built']
ori_yr_renovated= price_prediction ['yr_renovated']
ori_zipcode= price_prediction ['zipcode']
ori_lat= price_prediction ['lat']
ori_long= price_prediction ['long']
ori_sqft_living15= price_prediction ['sqft_living15']
ori_sqft_lot15= price_prediction ['sqft_lot15']

ori_sqft_basement_r = [1 for i in range (0,10000)]

for i in range (0,10000):
    if ori_sqft_basement[i] == 0:
        ori_sqft_basement_r [i] = 0       # it means the house
    else:
        ori_sqft_basement_r [i] = 1       # it means the house


for pastdays in pastdate:
    pastdays = str (pastdays)
    pastday = pastdays.split ('/')
    month.append (pastday [0])                      #split the value in date
    day.append (pastday [1])
    year.append(pastday [2])

for j in range (0,10000):
    day[j] = int (day[j])
    month [j] = int (month[j])                      #covert it to int
    year [j] = int (year[j])                        
    howmanydays = (today.day - day[j]) +(today.month-month[j])*30 + (today.year-year[j])*365        #caluate how long it has been
    longdays.append (howmanydays)

def normalize(matrix,matrix_2):
    max = np.max(matrix)
    min = np.min(matrix)
    for i in range (0,len (matrix)):
        z = (matrix [i] - min)/(max -min)
        matrix_2.append(z)

    return matrix_2

comfor = []
for i in range (0,10000):
    comfor.append (ori_bat[i]/ori_bed[i])

original_feature= [  
                   ori_dummy,
                   #longdays,
                   year,
                   ori_bed,
                   ori_bat,
                   ori_sqft_living,
                   ori_sqft_lot,
                   ori_floors,
                   ori_waterfront,
                   ori_view,
                   ori_condition,
                   ori_grade,
                   ori_sqft_above,
                   ori_sqft_basement,
                   ori_yr_built,
                   ori_yr_renovated,
                   ori_zipcode,
                   ori_lat,
                   ori_sqft_living15,
                   ori_sqft_lot15,
                   ]


int_feature = np.array(original_feature)
int_price = np.array(priceend)

for i in range (1,19):
    int_feature[i] = (normalize(int_feature[i],normalized_matrix))
    normalized_matrix = []

feature_matrix = np.array(int_feature)
y_prediction = np.dot(w_vector,feature_matrix)
mse = (1/len (y_prediction)) * sum ((y_prediction-int_price)*(y_prediction-int_price))

endnum = 10000
print ("Take off the attribute: waterfront")
print ("The MSE from previous DEV MSE is:7.591909234998686")
convergence =0.6

print ("Iterations setting:",endnum)
print ("==========================")

learning_rates_matrix = [0.1] #0.1,0.01]#,0.001,0.0001,0.00001,0.000001,0.0000001]
for learning_rates in learning_rates_matrix:
    iterations =0
    w_vector = w_vectors.copy()
    convergence = 0.6
    Condiction_break = 0
    while ( convergence >0.5 ):
    
        y_prediction = np.dot(w_vector,feature_matrix)
        y2 = y_prediction - int_price
        Dy = np.reshape(y2,(1,len(y2)))
        a=np.dot(Dy,feature_matrix.T)
        gradient = (2/len(y_prediction)) *sum(a)
        w_vector = w_vector - learning_rates * gradient
        multipy_gradient = (gradient * gradient) 
        convergence_sum = sum (multipy_gradient)
        convergence = math.sqrt(convergence_sum)
        iterations +=1
        if (iterations >endnum):
            Condiction_break = 1
            numbreaks +=1
            mse_matrix.append('too long')
            convergence_matrix.append('too long')
            break
        
    if Condiction_break == 0:
       
        iteration_matrix.append(iterations)
        convergence_matrix.append(convergence)
        w_vector_output.append(w_vector)
        mse = (1/len (y_prediction)) * sum ((y_prediction-int_price)*(y_prediction-int_price))
        mse_matrix.append(mse)
        output_int+=1


for i in range (0,len(learning_rates_matrix)-numbreaks):
    print ("Learning_matrix:",learning_rates_matrix[i])
    print ("Iterations:",iteration_matrix[i])
    print ("MSE:",mse_matrix[i] )
    print ("Convergence:",convergence_matrix[i])
    print ("==========================")
for i in range (len(learning_rates_matrix)-numbreaks,len(learning_rates_matrix)):
    print ("Learning rate:",learning_rates_matrix[i])
    print ("Iterations>500000")
    print ("MSE:","It takes too long")
    print ("Convergence:", "It takes too long")
    print ("==========================")

price_prediction2 = pd.read_csv ("PA1_dev.csv")
price_prediction2 = price_prediction2.drop (['id'], axis=1)
priceend2 = price_prediction2 ['price']

month2 = []
day2 = []
year2 = []
longdays2 = []

ori_dummy2 = price_prediction2 ['dummy']
pastdate2 = price_prediction2 ['date']
ori_bed2 = price_prediction2 ['bedrooms']
ori_bat2 = price_prediction2 ['bathrooms']
ori_sqft_living2= price_prediction2 ['sqft_living']
ori_sqft_lot2= price_prediction2 ['sqft_lot']
ori_floors2= price_prediction2 ['floors']
ori_waterfront2= price_prediction2 ['waterfront']
ori_view2= price_prediction2 ['view']
ori_condition2= price_prediction2 ['condition']
ori_grade2= price_prediction2 ['grade']
ori_sqft_above2 = price_prediction2 ['sqft_above']
ori_sqft_basement2 = price_prediction2 ['sqft_basement']
ori_yr_built2= price_prediction2 ['yr_built']
ori_yr_renovated2= price_prediction2 ['yr_renovated']
ori_zipcode2= price_prediction2 ['zipcode']
ori_lat2= price_prediction2 ['lat']
ori_long2= price_prediction2 ['long']
ori_sqft_living152= price_prediction2 ['sqft_living15']
ori_sqft_lot152= price_prediction2 ['sqft_lot15']

for pastdays in pastdate2:
    pastdays = str (pastdays)
    pastday = pastdays.split ('/')
    month2.append (pastday [0])                      #split the value in date
    day2.append (pastday [1])
    year2.append(pastday [2])

for j in range (0,5597):
    day2[j] = int (day2[j])
    month2 [j] = int (month2[j])                      #covert it to int
    year2 [j] = int (year2[j])                        
    howmanydays = (today.day - day2[j]) +(today.month-month2[j])*30 + (today.year-year2[j])*365        #caluate how long it has been
    longdays2.append (howmanydays)

ori_sqft_basement2_r = [1 for i in range (0,5597)]

for i in range (0,5597):
    if ori_sqft_basement2[i] == 0:
        ori_sqft_basement2_r [i] = 0       # it means the house
    else:
        ori_sqft_basement2_r [i] = 1       # it means the apartment

comfor2 = []
for i in range (0,5597):
    comfor2.append (ori_bat2[i]/ori_bed2[i])
original_feature2= [  
                   ori_dummy2,
                   #longdays2,
                   year2,
                   ori_bed2,
                   ori_bat2,
                   ori_sqft_living2,
                   ori_sqft_lot2,
                   ori_floors2,
                   ori_waterfront2,
                   ori_view2,
                   ori_condition2,
                   ori_grade2,                                           #orignal features in csv file  (15)
                   ori_sqft_above2,
                   ori_sqft_basement2,
                   ori_yr_built2,
                   ori_yr_renovated2,
                   ori_zipcode2,
                   ori_lat2,
                   ori_sqft_living152,
                   ori_sqft_lot152,
                   ]
int_feature2 = np.array(original_feature2)
int_price2 = np.array(priceend2)



normalized_matrix2 = []

for i in range (1,19):
    int_feature2[i] = (normalize(int_feature2[i],normalized_matrix2))
    normalized_matrix2 = []
output_matrix = []


feature_matrix2 = np.array(int_feature2)
for i in range (-1,0):
    y_prediction2 = np.dot(w_vector_output[i],feature_matrix2)
    mse = (1/len (y_prediction2)) * sum ((y_prediction2-int_price2)*(y_prediction2-int_price2))
    
    if i == -1:
        print ("MSE form DEV from learning rate:0,1" , mse)
    if i == -2:
        print ("MSE form DEV from learning rate:0,01" , mse)
    #if i == -2:
    #    print ("MSE form DEV from learning rate:0,001" , mse)
    #if i == -1:
    #    print ("MSE form DEV from learning rate:0,0001" , mse)
