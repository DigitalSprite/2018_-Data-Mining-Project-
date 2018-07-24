from sklearn.mixture import GaussianMixture
from q1.a.create_data import getMatrix
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import time

def gmm_dbscan(file_name, cova_type):
    data = getMatrix(file_name)
    list = []
    EPSK_BEST = 300
    for i in range(data.shape[1]):
        list.append(data.iloc[:, i])

    # 算出DBSCAN和Gaussian的labels的对比
    clf = GaussianMixture(n_components=1, covariance_type=cova_type)
    clf.fit(list)
    gaussian_labels = clf.predict(list)
    dbscan_cluster = DBSCAN(eps=EPSK_BEST, min_samples=10)
    dbscan_labels = dbscan_cluster.fit_predict(list)
    match_num = 0
    for i in range(data.shape[1]):
        if (dbscan_labels[i] == gaussian_labels[i]):
            match_num += 1
    ratio_dbscan = match_num / data.shape[1]
    print('与DBScan相比，在eps取' + str(EPSK_BEST) + '，GMM的准确率是：' + str(ratio_dbscan * 100) + '%')
    return gaussian_labels

if __name__ == '__main__':
    # gmm_dbscan('../../../reco_data/trade.csv', 'tied')
    # spherical, diagonal, tied or full性能比较
    cova = ['spherical', 'diag', 'tied', 'full']
    cova_index = [1,2,3,4]
    time_t = []
    for i in cova:
        t_start = time.time()
        gmm_dbscan('../../trade.csv', i)
        t_end = time.time()
        time_t.append(t_end - t_start)
    plt.title('time_DBSCAN')
    plt.bar(cova_index, time_t, width=0.5, facecolor="#9999ff", edgecolor="white")
    for x, y in zip(cova_index, time_t):
        plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')
    plt.ylim(0.0, 8)
    plt.show()