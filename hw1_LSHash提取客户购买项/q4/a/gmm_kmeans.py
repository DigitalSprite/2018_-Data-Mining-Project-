from sklearn.mixture import GaussianMixture
from q1.a.create_data import getMatrix
from sklearn.cluster import KMeans

def gmm_kmeans(file_name):
    data = getMatrix(file_name)
    list = []
    K_BEST = 2
    for i in range(data.shape[1]):
        list.append(data.iloc[:, i])

    # 算出Kmeans和Gaussian的labels对比正确率
    clf = GaussianMixture(n_components=K_BEST, covariance_type='full')
    clf.fit(list)
    gaussian_labels = clf.predict(list)
    kmeans_cluster = KMeans(n_clusters=K_BEST, random_state=10)
    kmeans_labels = kmeans_cluster.fit_predict(list)
    match_num = 0
    for i in range(data.shape[1]):
        if (gaussian_labels[i] == kmeans_labels[i]):
            match_num += 1
    ratio_kmeans = match_num / data.shape[1]
    print('与Kmeans相比，在K取' + str(K_BEST) + '，GMM的准确率是：' + str(ratio_kmeans * 100) + '%')
    return gaussian_labels




if __name__ == '__main__':
    gmm_kmeans('../../trade.csv')