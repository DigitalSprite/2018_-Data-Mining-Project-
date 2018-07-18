from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import math
import numpy as np
import matplotlib.pyplot as plt
from q1.a.create_data import getMatrix
import time

def Kmeans(file_name):
    data = getMatrix(file_name)
    list = []
    for i in range(data.shape[1]):
        list.append(data.iloc[:, i])
    initial_k = int(math.sqrt(data.shape[1] / 2))
    range_n_clusters = np.arange(2, initial_k, 1)
    k = []
    silhouette = []
    time_list = []
    for n_clusters in range_n_clusters:
        t_start = time.time()
        if n_clusters == 0 or n_clusters == 1:
            continue
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(list)
        avg_score = silhouette_score(list, cluster_labels)
        k.append(n_clusters)
        silhouette.append(avg_score)
        t_end = time.time()
        time_list.append(float(t_end - t_start))
        # print('k为' + str(n_clusters) + '的时候，时间：' + str(t_end - t_start) + 's')

        #n_cluster = 10时的扇形图
        # if n_clusters == 10:
        #     plt.figure(figsize=(6, 9))
        #     labels = []
        #     sizes = []
        #     for i in range(n_clusters):
        #         labels.append('label' + str(i))
        #         sizes.append(len(cluster_labels[cluster_labels == i]) / len(cluster_labels))
        #     patches, l_text, p_text = plt.pie(sizes, labels=labels,
        #                                       labeldistance=1.1, autopct='%3.1f%%', shadow=False,
        #                                       startangle=90, pctdistance=0.6)
        #     for t in l_text:
        #         t.set_size = (30)
        #     for t in p_text:
        #         t.set_size = (20)
        #     plt.axis('equal')
        #     plt.legend()
        #     plt.show()
    #柱状图
    plt.bar(k, time_list, width=0.8,facecolor="#9999ff",edgecolor="white")
    for x, y in zip(k, time_list):
        plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')
    plt.ylim(0.0, 1.0)
    plt.show()

    #折线图
    plt.figure('k-silhouette')
    plt.xlabel('K values')
    plt.ylabel('silhouette values')
    plt.title('k-silhouette table')
    plt.plot(k, silhouette)
    plt.show()
    return cluster_labels

if __name__ == '__main__':
    Kmeans('../../trade.csv')