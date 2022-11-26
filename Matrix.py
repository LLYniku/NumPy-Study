# coding=gbk
import numpy as np
import math
from numpy.linalg import matrix_rank

"""
代码运行指导：
ctrl+f5运行之后，选择分解的形式，每一种分解都会给出矩阵的行列式和最后解的值
输入矩阵在代码尾端修改，修改mat的值即可，右侧的等式值b也在尾端固定位置输入，目前给了一个输入，可直接运行
LU分解QR分解和URV分解函数代码来自网络，整合由本人自己整合
作者：天文学院 刘燎原 202228002509035 课程：矩阵分析与应用 指导老师：李保滨
"""

#代码最开始，设置结果输出统一格式，保留三位有效数字，取消科学记数法输出
def structure():
    np.set_printoptions(precision=3, floatmode='fixed') #三位有效数字输出
    np.set_printoptions(suppress=True)  # 取消科学计数法输出

#首先进行LU分解LU_mat(A)函数实现，A为输入矩阵！
def LU_mat(A):   #分解矩阵A的函数，得到L和U:
    n = len(A)
    L=np.eye(len(A))
    U=np.zeros(np.shape(A))
    for k in range(n):
        U[0,0]=A[0,0]
        if k==0:
            for j in range(1,n):
                U[0,j]=A[0,j]
                L[j,0]=A[j,0]/U[0,0] 
        else:
            for j in range(k,n):
                m=0
                for r in range(k):
                    m+=L[k,r]*U[r,j]
                U[k,j]=A[k,j]-m
            for i in range(k+1,n):
                m=0
                for r in range(k):
                    m+=L[i,r]*U[r,k]
                L[i,k]=(A[i,k]-m)/U[k,k]
    print("L={}\n".format(L),"U={}\n".format(U))
    return L,U
 
#在函数LU_fun(A,b)中，A为输入矩阵，b是函数最后的解
def LU_fun(A,b):  #定义一个LU分解函数，自变量是A,b
    m1,m2=A.shape  # m,n分别代表矩阵A的行数和列数
    n = len(A)
    if A[0,0]==0:
        print("no answer")     
    if m1<m2:
        print("这是一个解空间")
    else:
        L,U=LU_mat(A)
        y=b
        for k in range(1,n):
            m=0
            for r in range(k):
                m+=L[k,r]*y[r]
            y[k]-=m
        print("y={}".format(y))
        x=y
        x[n-1]=y[n-1]/U[n-1,n-1]
        for i in range(n-2,-1,-1):
            m=0
            for k in range(i+1,n):
                m+=U[i,k]*x[k]
            x[i]=(x[i]-m)/U[i,i]
    print("x={}".format(x))
    return x

#Gram-Schmidt进行的QR分解
def SchmitOrth(mat:np.array):
    cols = mat.shape[1]

    Q = np.copy(mat)
    R = np.zeros((cols, cols))

    for col in range(cols):
        for i in range(col):
            k =  np.sum(mat[:, col] * Q[:, i]) / np.sum( np.square(Q[:, i]) )
            Q[:, col] -= k*Q[:, i]
        Q[:, col] /= np.linalg.norm(Q[:, col])

        for i in range(cols):
            R[col, i] = Q[:, col].dot( mat[:, i] )

    return Q, R

#Householder reduction进行的QR分解
def HouseHolder(mat:np.array):
    cols = mat.shape[1]

    Q = np.eye(cols)
    R = np.copy(mat)

    for col in range(cols-1):
        a = np.linalg.norm(R[col:, col])
        e = np.zeros((cols- col))
        e[0] = 1.0
        num = R[col:, col] -a*e
        den = np.linalg.norm(num)

        u = num / den
        H = np.eye((cols))
        H[col:, col:] = np.eye((cols- col))- 2*u.reshape(-1, 1).dot(u.reshape(1, -1))
        R = H.dot(R)

        Q = Q.dot(H)

    return Q, R

#Givens reduction进行的QR分解
def GivenRot(mat:np.array):
    rows, cols = mat.shape

    R = np.copy(mat)
    Q = np.eye(cols)

    for col in range(cols):
        for row in range(col+1, rows):
            if abs(R[row, col]) < 1e-6:
                continue

            f = R[col, col]
            s = R[row, col]
            den = np.sqrt( f*f+ s*s)
            c = f / den
            s = s / den

            T = np.eye(rows)
            T[col, col], T[row, row] = c, c
            T[row, col], T[col, row] = -s, s

            R = T.dot(R)

            Q = T.dot(Q)
    
    return Q.T, R

#矩阵的URV分解所需的初变换
def Householder_Reduction_1(matrix):
    """
    处理输入不为方阵的情况
    :param matrix: 需分解的矩阵
    :return:  Q，R
    """
    matrix= np.array(matrix)
    m,n = matrix.shape
    num = min( m , n ) - 1
    if m > n :
        num = num + 1
    P = np.zeros((m,m,num))
    Q = np.eye(m,m)
    for i in range(m):
        P[i, i, :] = 1
    for i in range(num) :
        e = np.zeros((m-i,1))
        e[ 0 ,] = 1
        I = np.eye( m - i , m - i )
        u = np.reshape(matrix[ i : , i ],(m-i,1)) - np.reshape(math.sqrt(np.sum(matrix[ i : , i ] * matrix[ i : , i ])) * e,(m-i,1))
        temp = I - 2*np.dot(u,u.T)/np.dot(u.T,u)
        P[ i : , i : , i ] = np.copy(temp)
        matrix = np.dot(P[ : , : , i ], matrix )
        Q = np.dot(P[ : , : , i ], Q )
    return matrix , Q.T

#矩阵的URV分解
def URV_Factorization(matrix):
    '''
    :param matrix: 需分解的矩阵
    :return: m*m的正交矩阵U，V为n*n的正交矩阵V R为m*n的矩阵
    '''
    np.set_printoptions( precision=3, suppress=True )
    Q_one,R_one =Householder_Reduction_1(matrix)
    P = R_one.T
    temp = Q_one[ : matrix_rank(Q_one), :]
    Q_two,R_two = Householder_Reduction_1( temp.T )
    Q = R_two.T
    T = Q_two[ : matrix_rank(Q_two) , :]
    R = np.zeros_like(matrix,dtype=float)
    R[ : matrix_rank(Q_two) , : matrix_rank(Q_two)] = T.T
    # print( np.dot(np.dot(P.T,R),Q.T))
    print("U")
    print(P.T)
    print("R")
    print(R)
    print("V")
    print( Q.T)
    return P.T, R, Q.T

def solution(mat,b):#本代码为矩阵求解与行列式求解代码
        print("下面求矩阵的解\n")
        LU_fun(mat, b)
        print("下面求矩阵行列式的值\n")
        structure()
        print(np.linalg.det(mat))

#a为分解类型，A为输入矩阵，b为方程组右端值
def Decomposition_Type(a,A,b):
    if a == 1:#执行LU分解程序
        structure()
        LU_mat(A)
        solution(A,b)

    elif a == 2:#执行GS分解
        structure()
        A = A.astype(np.float32)
        q, r = SchmitOrth(A)
        print("Q: \n", q)
        print("R: \n", r)
        solution(A,b)

    elif a == 3:#执行HouseHolder分解
        structure()
        A = A.astype(np.float32)
        q, r = HouseHolder(A)
        print("Q: \n", q)
        print("R: \n", r)
        solution(A,b)
    elif a == 4:#执行旋转算子GivenRot分解
        structure()
        q, r = GivenRot(A)
        print("Q: \n", q)
        print("R: \n", r)
        solution(A,b)
    elif a == 5:#执行矩阵的URV分解
        structure()
        URV_Factorization(A)
        solution(A,b)

      
        
if __name__ == '__main__':
    #此处输入需要分解的矩阵
    mat = np.array([[1,2,3],[2,5,2],[3,1,5]])
    # mat = np.array( [   [0.0,-20.0,-14.0],
    #                     [3.0,27.0,-4.0],
    #                     [4.0,11.0,-2.0]     ])
    #此处修改b值
    b = np.array([14,18,20])
    # input输入转化为整型，input默认输入为字符串格式需要进行格式转换
    num = int(input('请输入想要分解的类型：1——LU分解，2——Schmidt分解，3——Householder分解，4——Givens reduction分解，5——URV分解'))
    Decomposition_Type( num , mat , b)



#代码运行指导：ctrl+f5运行之后，选择分解的形式，每一种分解都会给出矩阵的行列式和最后解的值
#输入矩阵在上方修改，修改mat的值即可，右侧的等式值b也在上方输入
#LU分解QR分解和URV分解函数代码来自网络，整合由本人自己整合
#作者：天文学院 刘燎原 202228002509035 课程：矩阵分析与应用 指导老师：李保滨

