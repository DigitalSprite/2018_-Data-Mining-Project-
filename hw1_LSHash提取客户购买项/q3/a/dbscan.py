from q1.a.create_data import getMatrix
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import time

def dbscan(file_name):
    data = getMatrix(file_name)
    list = []
    for i in range(data.shape[1]):
        list.append(data.iloc[:, i])
    range_n_clusters = np.arange(25, len(list) - 1, 10)
    eps = []
    silhouette = []
    noise_list = []
    time_list = []
    for i in range_n_clusters:
        start_t = time.time()
        db = DBSCAN(eps=i, min_samples=10).fit(list)
        end_time = time.time()
        labels = db.labels_
        avg_score = silhouette_score(list, labels)
        eps.append(i)
        silhouette.append(avg_score)
        noise_list.append(len(labels[labels == -1]) / len(labels))
        time_list.append(end_time - start_t)
        # 柱状图

    plt.title('time_DBSCAN')
    plt.bar(eps, time_list, width=2, facecolor="#9999ff", edgecolor="white")
    for x, y in zip(eps, time_list):
        plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')
    plt.ylim(0.0, 1.0)
    plt.show()

    plt.bar(eps, noise_list, width=2, facecolor="#9999ff", edgecolor="white")
    for x, y in zip(eps, noise_list):
        plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')
    plt.ylim(0.0, 1.0)
    plt.show()

    plt.figure('eps - Silhouette')
    plt.plot(eps, silhouette)
    plt.show()
    return labels

if __name__ == '__main__':
    dbscan('../../trade.csv')