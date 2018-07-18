from q1.b.ls_hash import getLSHashOutput
from q3.a.dbscan import dbscan

def verify(file_name):
    lsh_tags = getLSHashOutput(file_name, 0.01, 4)
    dbscan_tags = dbscan(file_name)

    for i in lsh_tags:
        print('标签为%d的分类为：%d'% (i, dbscan_tags[i]))


if __name__ == '__main__':
    verify('../../trade.csv')