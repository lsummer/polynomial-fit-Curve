# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:56:13 2016

@author: midsummer
"""

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
    
def bicgstab(X,Y,my_Y,B):
    '''
    #稳定双共轭梯度下降
    '''
    my_Y_copy=[]
    for i in my_Y:
      my_Y_copy.append(i)
      
    error = CostFunction(Y,my_Y_copy)
    
    R0star = Y - dot(X,B)
    R0 = Y - dot(X,B)
    rho0 = 1
    alp0 = 1
    w0 = 1
    V0 =mat(zeros(len(Y)).reshape(len(Y),1))
    P0 = mat(zeros(len(Y)).reshape(len(Y),1))
    #print R0
    while 1:
        rho1 = array(dot(R0star.T, R0))[0][0]
        beta = (rho1/rho0) * (alp0/w0)
        P1 = R0 + beta*(P0 - w0*V0)
        
        V1 = dot(X,P1)
        alp0 = rho1/(array(dot(R0star.T,V1))[0][0])
        h = B + alp0 * P1
        my_Y_copy = array(dot(X,array(h).reshape(len(h),1)).reshape(1,len(Y)))[0]
        new_error = CostFunction(Y,my_Y_copy) 
        if abs(new_error -error) <=e:
            B=h
            break
        error = new_error
        S = R0 - alp0*V1
        
        t = dot(X,S)
        w1 = array(dot(t.T, S))[0][0]/array(dot(t.T, t))[0][0]
        B = h + w1*S
        my_Y_copy = array(dot(X,array(B).reshape(len(B),1)).reshape(1,len(Y)))[0]
        new_error = CostFunction(Y,my_Y_copy) 
        if abs(new_error -error) <=e:
            break
        R0 = S - w1 * t
        rho0 = rho1
        P0 = P1
        V0 =V1
        w0 = w1
        error = new_error     
    return dot(X,B),B

def bicgstabReg(X,Y,my_Y,B):
    '''
    #稳定双共轭梯度下降
    '''
    my_Y_copy=[]
    for i in my_Y:
      my_Y_copy.append(i)
      
    error = CostFunctionReg(Y,my_Y_copy,B)
    
    R0star = Y - dot(X,B)
    R0 = Y - dot(X,B)
    rho0 = 1
    alp0 = 1
    w0 = 1
    V0 =mat(zeros(len(Y)).reshape(len(Y),1))
    P0 = mat(zeros(len(Y)).reshape(len(Y),1))
    #print R0
    while 1:
        rho1 = array(dot(R0star.T, R0))[0][0]
        beta = (rho1/rho0) * (alp0/w0)
        P1 = R0 + beta*(P0 - w0*V0)
        
        V1 = dot(X,P1)
        alp0 = rho1/(array(dot(R0star.T,V1))[0][0])
        h = B + alp0 * P1
        my_Y_copy = array(dot(X,array(h).reshape(len(h),1)).reshape(1,len(Y)))[0]
        new_error = CostFunctionReg(Y,my_Y_copy,h) 
        if abs(new_error -error) <=e:
            B=h
            break
        #error = new_error
        S = R0 - alp0*V1
        
        t = dot(X,S)
        w1 = array(dot(t.T, S))[0][0]/array(dot(t.T, t))[0][0]
        B = h + w1*S
        my_Y_copy = array(dot(X,array(B).reshape(len(B),1)).reshape(1,len(Y)))[0]
        new_error = CostFunctionReg(Y,my_Y_copy,B) 
       # print abs(new_error -error)
        if abs(new_error -error) <=e:
            break
        R0 = S - w1 * t
        rho0 = rho1
        P0 = P1
        V0 =V1
        w0 = w1
        error = new_error       
    return dot(X,B),B



if __name__=="__main__":
    X,Y,Xc,Yc,ALLX,ALLY = CreatData()
        
    
    X_matrix = getXmat(X)
    Xc_= getXmat(Xc)
    thata_ = []
    for i in range(10):
        thata_.append(uniform(-1,1))
    thata0 = mat(array(thata_).reshape(ORDER+1,1))
    thata1 = mat([1,1])
    thata1 = thata0
    my_Y = array(dot(X_matrix,thata0).reshape(1,len(Y)))[0]
    
    Y_temp = array(Y).reshape(DataMaxSize,1)
    Y_bcig,B = bicgstab(X_matrix,Y_temp,my_Y,thata0)
    Y_bcig_reg,B_reg = bicgstabReg(X_matrix,Y_temp,my_Y,thata1)
    
    my_Yc = array(dot(Xc_,B).reshape(1,len(Yc)))[0]
    my_Yc_reg = array(dot(Xc_,B_reg).reshape(1,len(Yc)))[0]
        
    loss_tlsm = CostFunction(Yc,my_Yc)# 方均根误差   
    loss_tlsm_reg = CostFunction(Yc,my_Yc_reg)
    loss_tlsm_1 = CostFunction(Y,Y_bcig)# 方均根误差           
    loss_tlsm_reg_1 = CostFunction(Y,Y_bcig_reg)    
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
    plt.plot(X,Y_bcig,'r',label = "$BCIG$")#红色的为最小二乘法求得的回归曲线
    plt.plot(X,Y_bcig_reg,'b--',label = "$BCIG-REG$")#红色的为最小二乘法求得的回归曲线
    
    plt.legend(loc=4)# make legend
    plt.show()
    
