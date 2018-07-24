from q1.b.ls_hash import getLSHashOutput
from q2.a.kmeans import Kmeans

def verify(file_name):
    lsh_tags = getLSHashOutput(file_name, 0.01, 4)
    kmeans_tags = Kmeans(file_name)
    print('\nlsh的knn分类后的某一个桶中vipno所对应的矩阵索引值:')
    for i in lsh_tags:
        print(i)
    print()
    for i in lsh_tags:
        print('标签为%d的分类为：%d'% (i, kmeans_tags[i]))


if __name__ == '__main__':
    verify('../../trade.csv')