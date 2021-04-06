import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt 
from scipy.special import expit

#Read training data into DataFrame
train_x_doc = pd.read_csv("pa4_train_X.csv")
train_y_doc = pd.read_csv("pa4_train_y.csv",names = ['y_label'])
dev_x_doc = pd.read_csv("pa4_dev_X.csv")
dev_y_doc = pd.read_csv("pa4_dev_y.csv",names = ['y_label'])

training_setX = train_x_doc.copy()
training_setY=train_y_doc.copy()
dev_setX = dev_x_doc.copy()
dev_setY = dev_y_doc.copy()

train_x = np.array(training_setX)
train_y = np.array(training_setY)
dev_x = np.array(dev_setX)
dev_y = np.array(dev_setY)

x_data_r_count,x_data_c_count = train_x.shape


def features(x_data , y_data):
    
    y_pos = 0
    y_neg = 0

    for i in range (0,len (y_data)):
        if y_data [i] == 0:
              y_neg +=1
        else:
              y_pos +=1
    

    if len(y_data) == 0:
        h_y = 0
    else:
        if y_neg == 0:
            if y_pos ==0:
                h_y = 0
            else:
                h_y = y_pos/len(y_data)*math.log(y_pos/len(y_data),2)* (-1)
        else:
            if y_pos ==0:
                h_y =  y_neg / len(y_data)  * math.log( y_neg / len(y_data),2) *(-1)
            else:
                h_y =  (y_neg / len(y_data)  * math.log( y_neg / len(y_data),2) + y_pos/len(y_data)*math.log(y_pos/len(y_data),2))* (-1)
    
    best_gain = 0
    xr_count,xc_count = x_data.shape
    leftx = []
    lefty = []
    rightx = []
    righty = []

    for i in range (0,xc_count):
        y1_0 = 0
        y1_1 = 0
        y2_0 = 0
        y2_1 = 0
        
        a = x_data[:,i]
        for j in range (0, xr_count):
            if a[j] ==0:
                if y_data [j] ==0:
                    y1_0 += 1
                elif y_data [j]==1:
                    y1_1 +=1
            else:
                if y_data [j] ==0:
                    y2_0 += 1
                elif y_data [j]==1:
                    y2_1 +=1

            total = y1_0 + y1_1 + y2_0 + y2_1
            if y1_0+y1_1 == 0:
                p_left_1 = 0
                p_left_2 = 0
            else:
                p_left_1 = y1_0/(y1_0+y1_1)
                p_left_2 = y1_1/(y1_0+y1_1)
            if y2_0+y2_1 ==0:
                p_right_1 =0
                p_right_2 = 0
            else:
                p_right_1 = y2_0/(y2_0+y2_1)
                p_right_2 = y2_1/(y2_0+y2_1)

        if p_left_1 ==0:
           if (p_left_2 ==0):
               left_entropy = 0
           else:
               left_entropy = p_left_2 * math.log (p_left_2,2) *-1 
        else:
            if (p_left_2 ==0):
               left_entropy = p_left_1 * math.log (p_left_1,2) * -1
            else:
                left_entropy = (p_left_1 * math.log (p_left_1,2) + p_left_2 * math.log (p_left_2,2)) * -1
        if p_right_1 ==0:
           if (p_right_2 ==0):
               right_entropy = 0
           else:
               right_entropy = p_right_2 * math.log (p_right_2,2) *-1
        else:
            if (p_right_2 ==0):
               right_entropy = p_right_1 * math.log (p_right_1,2) * -1
            else:
               right_entropy = (p_right_1 * math.log (p_right_1,2) + p_right_2 * math.log (p_right_2,2)) * -1
       
        if total == 0:
            conditional_entropy = 0
        else:
            conditional_entropy = (y1_0+y1_1)/total*left_entropy + (y2_0+y2_1)/total * right_entropy
        gain = h_y- conditional_entropy
        if (gain <= 0):
            gain = 0
        if best_gain <= gain:
             best_gain = gain
             best_feature = i
        if best_gain ==0:
            best_feature = None
    return best_feature



def Next_data(x_data, y_data ,best_feature):
    if best_feature == None:
        return 0,0,0,0,0,0
    leftx = []
    lefty = []
    rightx = []
    righty = []
    xr_count,xc_count = x_data.shape
    a = x_data [:,best_feature]
    for k in range (0, xr_count):
        if a[k] == 0:
             leftx.append (x_data[k])
             lefty.append (y_data[k])
        else:
             rightx.append(x_data[k])
             righty.append(y_data[k])

    
    leftx = np.array (leftx)
    rightx = np.array (rightx)
    lefty = np.array (lefty)
    righty = np.array (righty)

    row_count, col_count = lefty.shape
    y0_count = 0
    y1_count = 0
    for i in range (0, row_count):
        if lefty[i] == 0:
            y0_count += 1
        else:
            y1_count += 1
    
    leftmax = max(y0_count,y1_count)
    if leftmax == y0_count:
        yleft = 0
    else:
        yleft = 1

    row_count, col_count = righty.shape
    y0_count = 0
    y1_count = 0
    for i in range (0, row_count):
        if righty[i] == 0:
            y0_count += 1
        else:
            y1_count += 1
    rightmax = max(y0_count,y1_count)
    if rightmax == y0_count:
        yright = 0
    else:
        yright = 1
    return leftx, lefty, rightx, righty, leftmax, rightmax, yleft, yright 


class Node:

    def __init__(self, feature, leftx, lefty, rightx, righty, leftmax, rightmax, yleft, yright, y_prediction):

        self.left = None
        self.right = None
        self.feature = feature
        self.leftx = leftx
        self.lefty = lefty
        self.rightx = rightx
        self.righty = righty
        self.leftmax = leftmax
        self.rightmax = rightmax
        self.yleft = yleft
        self.yright = yright
        self.y_prediction = y_prediction
        self.leaf = False



def decision_tree (depth):
    level = depth + 1
    list = []
    new_list = []
    acc_array = []
    for i in range (0,level):
          if i == 0:
             best = features(train_x , train_y)
             leftx, lefty, rightx, righty, leftmax, rightmax, yleft, yright = Next_data (train_x, train_y, best)
             root = Node (best, leftx, lefty, rightx, righty, leftmax, rightmax, yleft, yright, 0)

             if (level == 1):
                 root.leaf = True
             list.append(root)
          else:
                for j in range (0, len(list)):
                    q = 0
                    w = 0
                    best = features(list[j].leftx, list[j].lefty)
                    if (best != None):
                        y_prediction = list[j].yleft
                        leftx, lefty, rightx, righty, leftmax, rightmax, yleft, yright = Next_data (list[j].leftx, list[j].lefty, best)
                        list[j].left = Node (best, leftx, lefty, rightx, righty, leftmax, rightmax, yleft, yright, y_prediction)
                        new_list.append(list[j].left)
                    else:
                        q=1
                    best = features(list[j].rightx, list[j].righty)
                    if (best != None):
                        y_prediction = list[j].yright
                        leftx, lefty, rightx, righty, leftmax, rightmax, yleft, yright = Next_data (list[j].rightx, list[j].righty, best)
                        list[j].right = Node (best, leftx, lefty, rightx, righty, leftmax, rightmax, yleft, yright, y_prediction)
                        new_list.append(list[j].right)

                    else:
                        w=1
                    if q ==1:
                        if w==1:
                            list[j].leaf = True

                list = new_list
                if i == level -1 :
                    for j in range (0, len(list)):
                        list[j].leaf = True

                new_list = []

          
    return root

acc_array = []


def print_decision_tree(tree,depth):
    list_tree = []
    new_list = []
    level = depth+1
    sum = 0
    for i in range (0,level):
        if i == 0:
            list_tree.append(tree)
        else:
            for k in range (0,len (list_tree)):
                print("level",i-1)
                print ("tree_left", list_tree[k].left)
                print ("tree_right",list_tree[k].right)
                print ("tree_feature",list_tree[k].feature)
                #print ("tree_leftx", len(list_tree[k].leftx))
                #print ("tree_rightx", len(list_tree[k].rightx))
                #print ("y_prediction(left)", list_tree[k].yleft)
                #print ("y_prediction(right)",list_tree[k].yright)
                print ("tree_leftmax",list_tree[k].leftmax)
                print("tree_rightmax",list_tree[k].rightmax)
                print ("tree_node_predict", list_tree[k].y_prediction)
                print ("Leaf?", list_tree[k].leaf)

                if list_tree[k].left != None:
                    new_list.append(list_tree[k].left)
                else:
                    if i != level -1:
                        sum +=  list_tree[k].leftmax

                if list_tree[k].right != None:
                    new_list.append(list_tree[k].right)
                else:
                    if i != level -1:
                        sum +=  list_tree[k].rightmax

                print ("======================================")

                if i == level -1:
                    sum += list_tree[k].rightmax + list_tree[k].leftmax
            if i == level - 1:        
                for index in range (0, len(new_list)):
                        print ("level",i)
                        print ("tree_left", new_list[index].left)
                        print ("tree_right",new_list[index].right)
                        print ("tree_feature",new_list[index].feature)
                        #print ("tree_leftx", len(new_list[index].leftx))
                        #print ("tree_rightx", len(new_list[index].rightx))
                        #print ("y_prediction(left)", new_list[index].yleft)
                        #print ("y_prediction(right)",new_list[index].yright)
                        #print ("tree_leftmax",new_list[index].leftmax)
                        #print("tree_rightmax",new_list[index].rightmax)
                        print ("tree_node_predict", new_list[index].y_prediction)
                        print ("Leaf?", new_list[index].leaf)
                        print ("======================================")
                

            
            list_tree = new_list
            new_list = []
    return sum   


def validation_acc(x_data, y_data, tree):
    
    count = 0
    root = tree
    node = root
    y_hat= [] 
    xr_count,xc_count = x_data.shape
    for i in range (0, xr_count):
        a = 0
        while (a==0):
            if (x_data[i][node.feature] == 0):
                if node.left != None:
                    node = node.left
                else:
                    y = node.y_prediction
                    y_hat.append (y)
                    a =1 
            elif (x_data[i][node.feature] == 1):
                if node.right != None:
                    node = node.right
                else:
                    y =node.y_prediction
                    y_hat.append(y)
                    a = 1

        node = root
    for k in range (0,xr_count):
        if y_data[k] == y_hat[k]:
            count +=1

    return count


#depth = [2,5,10,20,25,30,40,50]
depth_label = ['2','5','10','20','25','30','40','50']
depth = [2]
x_label = [0,1,2,3,4,5,6,7]
acc_array = []
dacc_array = []
for i in range (0, len(depth)):

    a = decision_tree (depth[i])

    acc = print_decision_tree(a,depth[i])
    print ("Accuracy",acc/len(train_y))
    acc_array.append (acc/len (train_y))
    b = validation_acc(dev_x,dev_y,a)
    dacc_array.append (b/len(dev_y))


plt.title("Accuracy (train_data v.s validation_data)")
plt.plot (acc_array, label = 'train_set', marker = 'o')
plt.plot (dacc_array, label = 'validation_set', marker = 'x')
plt.xticks (x_label, depth_label)
plt.xlabel('Depth of the tree')
plt.ylabel('Accuracy')
plt.legend()
plt.show()