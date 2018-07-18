import numpy as np
from lshash.lshash import LSHash
import random as rand
from q1.a.create_data import getMatrix
import matplotlib.pyplot as plt
import time

def getLSHashOutput(filename, hash_size, k):
    matrix = getMatrix(filename)
    list = []
    for i in range(matrix.shape[1]):
        list.append(matrix.iloc[i])
    total_num = len(matrix.iloc[0])
    lsh = LSHash(hash_size=int(hash_size * total_num), input_dim=len(matrix.iloc[:,0]))
    for i in range(total_num):
        lsh.index(input_point=matrix.iloc[:,i], extra_data=matrix.columns[i])
    out_num = rand.randint(0, total_num - 1)
    #有多种lshash函数，默认是euclidean
    m = lsh.query(query_point=matrix.iloc[:, out_num], num_results=k + 1, distance_func='euclidean')
    print("输入的vipno是" + str(matrix.columns[out_num]) + "\n其桶中的vipno有：")
    bucket = []
    for i in range(len(m)):
        print(m[i][0][1])
        tag = np.argwhere(matrix.columns == m[i][0][1])
        bucket.append(int(tag))
    return bucket

if __name__ == '__main__':
    hash_size = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    k = [1,2,3,4,5]
    r1 = rand.randint(0, 5)
    r2 = rand.randint(0,4)
    bucket = []
    time_list = []
    for i in hash_size:
        t_start = time.time()
        print('hash_size为' + str(i))
        bucket.append(len(getLSHashOutput('../../trade.csv', i, k[r2])))
        t_end = time.time()
        time_list.append(t_end - t_start)

    plt.bar(hash_size, bucket, width=0.02, facecolor="#9999ff", edgecolor="white")
    for x, y in zip(hash_size, bucket):
        plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom')
    plt.ylim(0, 6)
    plt.show()

    plt.bar(hash_size, time_list, width=0.02, facecolor="#9999ff", edgecolor="white")
    for x, y in zip(hash_size, time_list):
        plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom')
    plt.ylim(0, 6)
    plt.show()

