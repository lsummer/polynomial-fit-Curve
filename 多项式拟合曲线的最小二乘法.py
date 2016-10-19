# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:02:09 2016

@author: Xiya Lv
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import random


DataMaxSize = 10   #训练数据个数
DALL = DataMaxSize/2*3
ORDER = 9           #最高维数
alpha = 0.03         #步长
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
    for j in b_:
        loss+=lamda*j*j
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
    
def TheLeastSquareMethod(X,Y):
    """
    最小二乘法
    """

    regula = eye(ORDER+1)
    X_matrix=[]  
    for i in range(ORDER+1):  
        X_matrix.append(array(X)**i)  
    X_matrix = mat(X_matrix).T
    Y_matrix = array(Y).reshape((len(Y),1)) 
    
    X_matrix_T = X_matrix.T
    #print dot(X_matrix_T,X_matrix)
    B = dot(dot(dot(X_matrix_T,X_matrix).I,X_matrix_T),Y_matrix)
    B1 = dot(dot( (dot(X_matrix_T,X_matrix)+lamda*regula).I,X_matrix_T),Y_matrix)
    result = dot(X_matrix,B)
    result_reg = dot(X_matrix,B1)
    return X_matrix,Y_matrix,B,result,result_reg,B1


if __name__=="__main__":
    X,Y,Xc,Yc,ALLX,ALLY = CreatData()
        
    X_matrix, Y_matrix, B,Y_tlsm,Y_tlsm_reg,B_reg = TheLeastSquareMethod(X,Y)
     #最小二乘法求得的回归曲线,矩阵表示
    
    Xc_= getXmat(Xc)
    
    
    my_Yc = array(dot(Xc_,B).reshape(1,len(Yc)))[0]
    my_Yc_reg = array(dot(Xc_,B_reg).reshape(1,len(Yc)))[0]
        
    loss_tlsm = CostFunction(Yc,my_Yc)# 方均根误差   
    loss_tlsm_reg = CostFunction(Yc,my_Yc_reg)
    loss_tlsm_1 = CostFunction(Y,Y_tlsm)# 方均根误差           
    loss_tlsm_reg_1 = CostFunction(Y,Y_tlsm_reg)    
    print loss_tlsm_1
    print loss_tlsm_reg_1    
    
    plt.figure("MachineLearningProjectOne")
    plt.figtext(0.3,0.85,'Cost Without Regulation:'+str(loss_tlsm),color='red',ha='center')
    plt.figtext(0.3,0.75,'Cost With Regulation:'+str(loss_tlsm_reg),color='red',ha='center')  
    plt.title("NumberOfTrainData:"+str(DataMaxSize))
    plt.xlabel('x axis')# make axis labels
    plt.ylabel('y axis')
    plt.plot(Xc,Yc,'ob',label="$check data$")
    plt.plot(X,Y,'or',label="$train data$")
    plt.plot(X,Y_tlsm,'r',label = "$LS$")#红色的为最小二乘法求得的回归曲线
    plt.plot(X,Y_tlsm_reg,'b',label = "$LS-REG$")#红色的为最小二乘法求得的回归曲线
    
    plt.legend(loc=4)# make legend
    plt.show()
    
