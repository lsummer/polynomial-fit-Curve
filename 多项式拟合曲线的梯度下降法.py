# -*- coding: utf-8 -*-

"""
Created on Wed Oct 19 08:20:47 2016

@author: midsummer
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random


DataMaxSize =14   #训练数据个数
DALL = DataMaxSize/2*3
ORDER = 9           #最高维数
alpha = 0.03         #步长
e=0.000001           #精度
lamda = 1
def func_(x):
    """
    函数：sin（0.4*pi*x）
    """
    return sin(0.4*pi*x)
    
def CostFunction(Y,my_Y):
    loss = 0
    for i in range(len(Y)):
        loss+=pow((my_Y[i]-Y[i]),2)
    
    loss = 0.5*loss*1.0/DataMaxSize
    return loss

def CostFunctionReg(Y,my_Y,B):
    loss = 0
    b_ = array(B).reshape(1,ORDER+1)[0]
    for i in range(len(Y)):
        loss+=pow((my_Y[i]-Y[i]),2)
    for j in range(len(b_)):
        loss+=lamda*b_[j]/(2.5**j)*b_[j]/(2.5**j)
    loss = 0.5*loss*1.0/DataMaxSize
    return loss
    
def CreatData():
    """
    1.功能：生成数据，并加入噪声
    2.定义域（-2.5，+2.5）
    3.采用函数：func_(x)
    """  
    RandNumberX = arange(-2.5,2.5,5.0/DALL)
    RandNumberY=[]
    X=[]
    Xc=[]
    Y = []
    Yc=[]
    for i in range(len(RandNumberX)):
        if (i+1)%3==0:
            Xc.append(RandNumberX[i])
        else:
            X.append(RandNumberX[i])
    for x in RandNumberX:
        #RandNumberY.append(func_(x)+random.lognormvariate(0, 1)) #正态分布
        RandNumberY.append(func_(x)+ uniform(-0.2, 0.2))
    for i in range(len(RandNumberY)):
        if (i+1)%3==0:
            Yc.append(RandNumberY[i])
        else:
            Y.append(RandNumberY[i])
    return X,Y,Xc,Yc,RandNumberX,RandNumberY
def getXmat(Xc):
    X_matrix=[]  
    for i in range(ORDER+1):  
        X_matrix.append(array(Xc)**i)  
    X_matrix = mat(X_matrix).T
    #print X_matrix
    return X_matrix

def SumMy_Y_Y(my_Y,Y,X,j):
    total=0.0
    for i in range(0,len(Y)):
        total += ((my_Y[i]-Y[i])*X[i][j])
    return total   
    
def SumMy_Y_Y_REGULATION(my_Y,Y,X,j,thata):
    total=0.0
    if j==0:
        for i in range(0,len(Y)):
            total += (((my_Y[i]-Y[i])*X[i][j]))
        return total
    for i in range(0,len(Y)):
        total += (((my_Y[i]-Y[i])*X[i][j])+lamda/(DataMaxSize*thata[j]*(2.5**j)))
    return total   
def reFeature(B):
    tha_ = (array(B).reshape(1,ORDER+1))[0]
    b=[]
    for i in range(ORDER+1):
        b.append(tha_[i]/(2.5**i))
    
    return mat(array(b).reshape(ORDER+1,1))
    
def BatchGradientDescentReg(my_Y,Y,X_mat,thata):
    """
    批量梯度下降(加惩罚项)
    """    
    my_copy=[]
    for i in my_Y:
      my_copy.append(i)
    
    error = CostFunctionReg(Y,my_copy,thata)
    X = array(X_mat)
   
    while(1):
        new_thata = []
        for i in range(0,len(thata)):
            #temp = SumMy_Y_Y(my_copy,Y,X, i)
            temp = SumMy_Y_Y_REGULATION(my_copy,Y,X, i,thata)
            flag = thata[i] - alpha*temp/DataMaxSize
            new_thata.append(flag)
        
        thata = new_thata
        
        my_copy = array(dot(X_mat,array(new_thata).reshape(len(thata),1)).reshape(1,len(Y)))[0]
        new_error = CostFunctionReg(Y,my_copy,thata)  
        #print abs(new_error -error)
        if abs(new_error -error) <=e:
            break
        
        error = new_error
    B= reFeature(thata)
    return my_copy,B


def BatchGradientDescent(my_Y,Y,X_mat,thata):
    """
    批量梯度下降(未加惩罚项)
    """    
    my_copy=[]
    for i in my_Y:
      my_copy.append(i)
    
    error = CostFunction(Y,my_copy)
    X = array(X_mat)
   
    while(1):
        new_thata = []
        for i in range(0,len(thata)):
            #temp = SumMy_Y_Y(my_copy,Y,X, i)
            temp = SumMy_Y_Y(my_copy,Y,X, i)
            flag = thata[i] - alpha*temp/DataMaxSize
            new_thata.append(flag)
        
        thata = new_thata
        
        my_copy = array(dot(X_mat,array(new_thata).reshape(len(thata),1)).reshape(1,len(Y)))[0]
        new_error = CostFunction(Y,my_copy)  
        print abs(new_error -error)
        if abs(new_error -error) <=e:
            break
        
        error = new_error
        
    B= reFeature(thata)
    return my_copy,B

    
if __name__=="__main__":
    X,Y,Xc,Yc,ALLX,ALLY = CreatData()
    Xc_= getXmat(Xc)  
    '''
    #批量梯度下降
    #初始的系数矩阵取B（即thata）
    '''    
    X_lemad = array(X)/2.5  #特征收缩    
    X_mat_=[]  
    for i in range(ORDER+1):  
        X_mat_.append(X_lemad**i)  
    X_mat_ = mat(X_mat_).T    
    #thata = array(B.reshape(1,len(B)))[0] +uniform(0,0.1)
    X_matrix = getXmat(X)  
    thata_ = []
    for i in range(10):
        thata_.append(uniform(-1,1))
    thata0 = mat(array(thata_).reshape(ORDER+1,1))
    thata1 = mat([1,1])
    thata1 = thata0
    
    my_Y = array(dot(X_matrix,thata0).reshape(1,len(Y)))[0]
    
    thata = array(thata0.reshape(1,ORDER+1))[0]
    Y_bgd,B = BatchGradientDescent(my_Y,Y,X_mat_,thata)
    
    thata = array(thata0.reshape(1,len(B)))[0]
    Y_bgd_reg,B_reg = BatchGradientDescentReg(my_Y,Y,X_mat_,thata)
    
    
    
    my_Yc = array(dot(Xc_,B).reshape(1,len(Yc)))[0]
    my_Yc_reg = array(dot(Xc_,B_reg).reshape(1,len(Yc)))[0]
        
    loss_tlsm = CostFunction(Yc,my_Yc)# 方均根误差   
    loss_tlsm_reg = CostFunction(Yc,my_Yc_reg)
    loss_tlsm_1 = CostFunction(Y,Y_bgd)# 方均根误差           
    loss_tlsm_reg_1 = CostFunction(Y,Y_bgd_reg)    
    print "without Regulation: "+str(loss_tlsm_1)
    print "with Regulation: "+str(loss_tlsm_reg_1)   
    
    plt.figure("MachineLearningProjectOne")
    plt.figtext(0.3,0.85,'Cost Without Regulation:'+str(loss_tlsm),color='red',ha='center')
    plt.figtext(0.3,0.75,'Cost With Regulation:'+str(loss_tlsm_reg),color='red',ha='center')  
    plt.title("NumberOfTrainData:"+str(DataMaxSize))
    plt.xlabel('x axis')# make axis labels
    plt.ylabel('y axis')
    plt.plot(Xc,Yc,'ob',label="$check data$")
    plt.plot(X,Y,'or',label="$train data$")
    plt.plot(X,Y_bgd,'b',label = '$BGD$')#蓝色为梯度下降所求的回归曲线  
    plt.plot(X,Y_bgd_reg,'r',label = '$BGD-REG$')#蓝色为梯度下降所求的回归曲线 
    
    plt.legend(loc=4)# make legend
    plt.show()
    
