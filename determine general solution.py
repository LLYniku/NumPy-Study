import numpy as np

def swag(a, b):     #交换矩阵某两行函数
    for i in range(0, len(a)):
        t = a[i]
        a[i] = b[i]
        b[i] = t

def print_matrix( info, matrix ):       #打印矩阵
    print (info)
    for i in range( 0, matrix.shape[0]): # shape[n]读取n维度的长度 axis[n] 本句意思是从第一行到最后一行
        print('[', end='')               #每次开头读取之后都加一个[
        for j in range( 0, matrix.shape[1]):
            if(j == matrix.shape[1] - 1):   # 向量 b
                print( '|', end=''),                           # 将系数矩阵和向量b用“|”隔开
            print("%.3f" %matrix[i][j], end=' ')          #总长5位，小数点后2位
            if j == matrix.shape[1] - 1:                  # 输出矩阵元素m[i][j]
                print(']', end=' ')
                print('\n')

def check(matrix, i, row, col):         #判定函数，用于检测当前处理的矩阵的行的值是否全为0，若全为零，则从矩阵最底行开始遍历找出不全为零的行与此行交换，再对交换后的行进行处理
    if 0.00 in set(matrix[i]) and len(set(matrix[i])) == 1:     #判定当前行是否全为零，将当前行集合化看其是否只有一个0元素
        for j in range(row - 1, i ,-1):                         #从矩阵最后一行开始寻找不全为零的一行
            try:
                if not(0.00 in set(matrix[j]) and len(set(matrix[j])) == 1):    #判定这行是否全为零
                    swag(matrix[i], matrix[j])                                  #交换
                    select(matrix, i, col)                                      #交换后再进行阶梯化处理
                    break
            except:                     #如果遍历发现没有不为零的行了，则直接返回
                return

def select(matrix, i, col):             #矩阵的阶梯化处理函数
    if 0.00 in set(matrix[i]) and len(set(matrix[i])) == 1:     #主要针对矩阵最后一行，judge函数无法对最后一行进行非零判定
        return
    for k in range(0, i):                                       #i是当前处理的矩阵的行数，这个循环是对矩阵的第i行进行阶梯化处理
        
        temp = matrix[i][k] / matrix[k][k]                      #根据主元matrix[k][k]确定让matrix[i][k]归零的除数
        #temp = round(temp,3)
        #print(temp)
        if temp == 0:                                           #主元为0不执行操作
            continue
        for j in range(0, col):                                 #对这一行的所有数进行操作，其中和主元同一列的那个数会归零
            matrix[i][j] = matrix[i][j] - matrix[k][j] * temp
            #matrix[i][j] = round(matrix[i][j],6)                                            #修改的部分
            #print(matrix[i][j])

def solve(matrix):                      #算法的主体部分
    row = matrix.shape[0]                       #得到矩阵的行数
    col = matrix.shape[1]                       #得到矩阵的列数
    for i in range(0, row):                     #检查矩阵某一行的主元是否为0，为零则遍历寻找不为零的行与其交换
        if matrix[i][i] == 0:                   #易得matrix[i][i]就为第i行主元
            for j in range(i + 1, row):
                if matrix[j][i] != 0:
                    swag(matrix[i], matrix[j])
                    break
        select(matrix, i, col)                  #进行阶梯化操作
        check(matrix, i, row, col)              #判定阶梯化后会不会使此行的值全化为零，若全化为零则将其与其他不为零的行交换，再进行阶梯化处理

def to_one(matrix):                     #将矩阵化为行最简形矩阵
    row = matrix.shape[0]
    col = matrix.shape[1]
    for i in range(0, row):                         #将矩阵每一行主元全部归一
        temp = matrix[i][i]
        for j in range(i, col):
            matrix[i][j] = matrix[i][j] / temp
    for i in range(0, row - 1):                     #对矩阵每一行非主元进行归零处理
        for j in range(i + 1, col - 1):             #从主元后开始遍历对每一个数进行归零处理
            temp = matrix[i][j]                     #之前将矩阵的主元都化为1，则此处令其归零的除数就是其本身 （x / 1 = x）
            for k in range(j, col):                 # matrix[i][k]是第i行的第k个值，对此行的第j个数字进行归零处理，此行其他的数字也要做相同操作
                matrix[i][k] = matrix[i][k] - matrix[j][k] * temp   #matrix[j][k]是matrix[i][k]对应的主元行上对应的列值

def judge(matrix):                      #由阶梯化后的矩阵判定线性方程组解的个数
    row = matrix.shape[0]
    col = matrix.shape[1]
    vanumlist = []
    for i in range(0, col):             #因为由solve函数确定的是一个行阶梯型矩阵，则仅需判定最后一行非零数字的个数。若无或者有多个，则证明多解；有一个则无解；有两个则证明有唯一解
        if matrix[row - 1][i] != 0:
            vanumlist.append(matrix[row - 1][i])    #每遍历到一个非零值就将其写入vanumlist这个list中，最后计算vanumlist的长度即可
    if len(vanumlist) == 1:
        print( '方程组无解')
    elif len(vanumlist) == 2:
        to_one(matrix)                  #若有唯一解，则将矩阵进一步化成行最简形矩阵求出解
        print_matrix('将矩阵最简阶梯化为：', matrix)
        
        for i in range(0, row):
            print("x%d = %4.2f" %(i,matrix[i][col - 1]), end="  ")
    else:
        print( '方程组有多解')

matrix = np.array([ [ 2, 1, 1, 4],
                   [ 4, 2, 1, 6],
                  [ 6, 3, 1, 8],
                  [ 8, 4, 1, 10]
                  ],dtype=float)

''' matrix = np.array([ [ 1, 1, 1, 1, 1,  1],
                    [ 3, 2, 1, 1,-3, 25],
                    [ 0, 1, 2, 2, 6,-22],
                    [ 5, 4, 3, 3,-1, 27],
                    [ 2,-1, 0, 3, 2,  2]
                    ],dtype=float) '''


print_matrix('初始矩阵为：', matrix)
solve(matrix)
print_matrix('矩阵阶梯化为：', matrix)
judge(matrix)
rank = np.linalg.matrix_rank(matrix)
print(matrix)
print("矩阵的秩为：", rank)