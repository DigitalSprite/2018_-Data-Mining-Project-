from q1.b.ls_hash import getLSHashOutput
from q4.a.gmm_kmeans import gmm_kmeans
from q4.b.gmm_dbscan import gmm_dbscan

def verify(file_name):
    lsh_tags = getLSHashOutput(file_name, 0.01, 4)
    gmm_tags = gmm_kmeans(file_name)
    for i in lsh_tags:
        print('Kmeans作为聚类结果：标签为%d的分类为：%d'% (i, gmm_tags[i]))
    gmm_tags = gmm_dbscan(file_name)
    for i in lsh_tags:
        print('DBSCAN作为聚类结果：标签为%d的分类为：%d'% (i, gmm_tags[i]))


if __name__ == '__main__':
    verify('../../trade.csv')