# coding=gbk
import numpy as np
import math
from numpy.linalg import matrix_rank

"""
��������ָ����
ctrl+f5����֮��ѡ��ֽ����ʽ��ÿһ�ַֽⶼ��������������ʽ�������ֵ
��������ڴ���β���޸ģ��޸�mat��ֵ���ɣ��Ҳ�ĵ�ʽֵbҲ��β�˹̶�λ�����룬Ŀǰ����һ�����룬��ֱ������
LU�ֽ�QR�ֽ��URV�ֽ⺯�������������磬�����ɱ����Լ�����
���ߣ�����ѧԺ ����ԭ 202228002509035 �γ̣����������Ӧ�� ָ����ʦ�����
"""

#�����ʼ�����ý�����ͳһ��ʽ��������λ��Ч���֣�ȡ����ѧ���������
def structure():
    np.set_printoptions(precision=3, floatmode='fixed') #��λ��Ч�������
    np.set_printoptions(suppress=True)  # ȡ����ѧ���������

#���Ƚ���LU�ֽ�LU_mat(A)����ʵ�֣�AΪ�������
def LU_mat(A):   #�ֽ����A�ĺ������õ�L��U:
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
 
#�ں���LU_fun(A,b)�У�AΪ�������b�Ǻ������Ľ�
def LU_fun(A,b):  #����һ��LU�ֽ⺯�����Ա�����A,b
    m1,m2=A.shape  # m,n�ֱ�������A������������
    n = len(A)
    if A[0,0]==0:
        print("no answer")     
    if m1<m2:
        print("����һ����ռ�")
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

#Gram-Schmidt���е�QR�ֽ�
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

#Householder reduction���е�QR�ֽ�
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

#Givens reduction���е�QR�ֽ�
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

#�����URV�ֽ�����ĳ��任
def Householder_Reduction_1(matrix):
    """
    �������벻Ϊ��������
    :param matrix: ��ֽ�ľ���
    :return:  Q��R
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

#�����URV�ֽ�
def URV_Factorization(matrix):
    '''
    :param matrix: ��ֽ�ľ���
    :return: m*m����������U��VΪn*n����������V RΪm*n�ľ���
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

def solution(mat,b):#������Ϊ�������������ʽ������
        print("���������Ľ�\n")
        LU_fun(mat, b)
        print("�������������ʽ��ֵ\n")
        structure()
        print(np.linalg.det(mat))

#aΪ�ֽ����ͣ�AΪ�������bΪ�������Ҷ�ֵ
def Decomposition_Type(a,A,b):
    if a == 1:#ִ��LU�ֽ����
        structure()
        LU_mat(A)
        solution(A,b)

    elif a == 2:#ִ��GS�ֽ�
        structure()
        A = A.astype(np.float32)
        q, r = SchmitOrth(A)
        print("Q: \n", q)
        print("R: \n", r)
        solution(A,b)

    elif a == 3:#ִ��HouseHolder�ֽ�
        structure()
        A = A.astype(np.float32)
        q, r = HouseHolder(A)
        print("Q: \n", q)
        print("R: \n", r)
        solution(A,b)
    elif a == 4:#ִ����ת����GivenRot�ֽ�
        structure()
        q, r = GivenRot(A)
        print("Q: \n", q)
        print("R: \n", r)
        solution(A,b)
    elif a == 5:#ִ�о����URV�ֽ�
        structure()
        URV_Factorization(A)
        solution(A,b)

      
        
if __name__ == '__main__':
    #�˴�������Ҫ�ֽ�ľ���
    mat = np.array([[1,2,3],[2,5,2],[3,1,5]])
    # mat = np.array( [   [0.0,-20.0,-14.0],
    #                     [3.0,27.0,-4.0],
    #                     [4.0,11.0,-2.0]     ])
    #�˴��޸�bֵ
    b = np.array([14,18,20])
    # input����ת��Ϊ���ͣ�inputĬ������Ϊ�ַ�����ʽ��Ҫ���и�ʽת��
    num = int(input('��������Ҫ�ֽ�����ͣ�1����LU�ֽ⣬2����Schmidt�ֽ⣬3����Householder�ֽ⣬4����Givens reduction�ֽ⣬5����URV�ֽ�'))
    Decomposition_Type( num , mat , b)



#��������ָ����ctrl+f5����֮��ѡ��ֽ����ʽ��ÿһ�ַֽⶼ��������������ʽ�������ֵ
#����������Ϸ��޸ģ��޸�mat��ֵ���ɣ��Ҳ�ĵ�ʽֵbҲ���Ϸ�����
#LU�ֽ�QR�ֽ��URV�ֽ⺯�������������磬�����ɱ����Լ�����
#���ߣ�����ѧԺ ����ԭ 202228002509035 �γ̣����������Ӧ�� ָ����ʦ�����

