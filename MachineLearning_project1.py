# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:02:09 2016

@author: Xiya Lv
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import random
DataMaxSize = 10   #数据个数
ORDER = 9           #最高维数
alpha = 0.03         #步长
e=0.000001           #精度
lamda = 0.1
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
    RandNumberX = arange(-2.5,2.5,5.0/DataMaxSize)
    RandNumberY=[]
    for x in RandNumberX:
        #RandNumberY.append(func_(x)+random.lognormvariate(0, 1)) #正态分布
        RandNumberY.append(func_(x)+ uniform(-0.2, 0.2))
    return RandNumberX,RandNumberY
    
def TheLeastSquareMethod(X,Y):
    """
    最小二乘法
    """
    '''
    regulation=[]
    r=[]
    for i in range(ORDER+1):
        r.append(0)
    regulation.append(r)
    for i in range(1,ORDER+1):
        r=[]
        for j in range(0, ORDER+1):
            if i==j:
                r.append(1)
            else:
                r.append(0)
        regulation.append(r)
    regula = mat(regulation)
    '''
    regula = eye(ORDER+1)
    X_matrix=[]  
    for i in range(ORDER+1):  
        X_matrix.append(X**i)  
    X_matrix = mat(X_matrix).T
    Y_matrix = array(Y).reshape((len(Y),1)) 
    
    X_matrix_T = X_matrix.T
    #print dot(X_matrix_T,X_matrix)
    B = dot(dot(dot(X_matrix_T,X_matrix).I,X_matrix_T),Y_matrix)
    B1 = dot(dot( (dot(X_matrix_T,X_matrix)+lamda*regula).I,X_matrix_T),Y_matrix)
    result = dot(X_matrix,B)
    result_reg = dot(X_matrix,B1)
    return X_matrix,Y_matrix,B,result,result_reg

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
        total += (((my_Y[i]-Y[i])*X[i][j])+lamda/DataMaxSize*thata[j])
    return total   

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
    return my_copy

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
        #print abs(new_error -error)
        if abs(new_error -error) <=e:
            break
        
        error = new_error
    return my_copy

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
    return dot(X,B)

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
        new_error = CostFunction(Y,my_Y_copy) 
        if abs(new_error -error) <=e:
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
    return dot(X,B)
    
if __name__=="__main__":
    X,Y = CreatData()
    ###############################################################################
    '''
    最小二乘法
    '''
    X_matrix, Y_matrix, B,Y_tlsm,Y_tlsm_reg = TheLeastSquareMethod(X,Y)
     #最小二乘法求得的回归曲线,矩阵表示
    
    my_Y = array(Y_tlsm.reshape(1,len(Y)))[0]
    loss_tlsm = CostFunction(Y,my_Y)# 方均根误差
    
    #################################################################################
    
    ################################################################################
    '''
    #批量梯度下降
    #初始的系数矩阵取B（即thata）
    '''    
    X_lemad = X/2.5  #特征收缩    
    X_mat_=[]  
    for i in range(ORDER+1):  
        X_mat_.append(X_lemad**i)  
    X_mat_ = mat(X_mat_).T    
    #thata = array(B.reshape(1,len(B)))[0] +uniform(0,0.1)
    thata0 = B+uniform(0,0.1)
    thata1 = mat([1,1])
    thata1 = thata0
    #####################
    #thata = []
    #for i in range(10):
    #    thata.append(uniform(-1,1))
    ####################
    
    my_Y = array(dot(X_matrix,thata0).reshape(1,len(Y)))[0]
    
    thata = array(thata0.reshape(1,len(B)))[0]
    Y_bgd = BatchGradientDescent(my_Y,Y,X_mat_,thata)
    
    thata = array(thata0.reshape(1,len(B)))[0]
    Y_bgd_reg = BatchGradientDescentReg(my_Y,Y,X_mat_,thata)
    ###################################################################################
       
    ######################################################################################################3
    '''
    #双共轭梯度法(BCIG)
    '''
    #thata = array(B.reshape(1,len(B)))[0] 
    if(DataMaxSize==ORDER+1):
        Y_temp = array(Y).reshape(DataMaxSize,1)
        Y_bcig = bicgstab(X_matrix,Y_temp,my_Y,thata0)
        Y_bcig_reg = bicgstabReg(X_matrix,Y_temp,my_Y,thata1)
    ######################################################################################3######
    '''
    画图显示
    '''
    plt.figure("MachineLearningProjectOne")
    plt.title("polynomial fit curve in 3 ways")
    plt.xlabel('x axis')# make axis labels
    plt.ylabel('y axis')
    plt.plot(X,Y,'or',label="$origional data$")
    plt.plot(X,Y_tlsm,'r',label = "$LS$")#红色的为最小二乘法求得的回归曲线
    plt.plot(X,Y_tlsm_reg,'r--',label = "$LS-REG$")#红色的为最小二乘法求得的回归曲线
    plt.plot(X,Y_bgd,'b',label = '$BGD$')#蓝色为梯度下降所求的回归曲线  
    plt.plot(X,Y_bgd_reg,'b--',label = '$BGD-REG$')#蓝色为梯度下降所求的回归曲线 
    if(DataMaxSize==ORDER+1):
        plt.plot(X,Y_bcig,'y--',label = '$BCIG$')#绿色为稳定双共轭梯度所求的回归曲线
        plt.plot(X,Y_bcig_reg,color='y',label = '$BCIG-REG$')#绿色为稳定双共轭梯度所求的回归曲线
    plt.legend(loc=4)# make legend
    plt.show()
    
