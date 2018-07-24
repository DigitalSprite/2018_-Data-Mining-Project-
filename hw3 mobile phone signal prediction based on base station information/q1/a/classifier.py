from q1.a.handle_data import HandleData
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    handle = HandleData()
    handle.handle_data_2g('data/data.csv')
    print('#############################################################################')
    print('the minimum latitude is %.3f' % round(handle.min_latitude, 3))
    print('the maximum latitude is %.3f' % round(handle.max_latitude, 3))
    print('the minimum longitude is %.3f' % round(handle.min_longitude, 3))
    print('the maximum longitude is %.3f' % round(handle.max_longitude, 3))
    print('the grid width number is : %d' % handle.grid_width_num)
    print('the grid height number is : %d' % handle.grid_height_num)
    data = handle.data
    labels = data.loc[:, 'Position']
    axis = data.loc[:, ['Longitude', 'Latitude']]
    data = data.drop(['Longitude', 'Latitude', 'Position', 'MRTime', 'IMSI'], axis=1)
    kf = KFold(n_splits=10, shuffle=True)
    # 统计7种分类器的误差
    distance = []
    precision = []
    recall = []
    f1 = []
    for i in range(7):
        distance.append([])
        precision.append([])
        recall.append([])
        f1.append([])
    for train_index, test_index in kf.split(data):
        print('#############################################################################')
        train_data = data.loc[train_index, :]
        train_label = labels.loc[train_index]
        test_data = data.loc[test_index, :]
        test_label = labels.loc[test_index]
        test_axis = axis.loc[test_index, :]
        # decision tree classifier
        clf =DecisionTreeClassifier(criterion='entropy')
        clf.fit(train_data, train_label)
        result = clf.predict(test_data)
        temp = []
        for i in range(len(result)):
            lon = test_axis.iloc[i].loc['Longitude']
            lat = test_axis.iloc[i].loc['Latitude']
            temp.append(handle.grid_to_distance(lat, lon, result[i]))
        distance[0].append(sorted(temp))
        precision[0].append(precision_score(test_label, result, average='macro'))
        recall[0].append(recall_score(test_label, result, average='macro'))
        f1[0].append(f1_score(test_label, result, average='macro'))
        print('Decision Tree:\tprecision score: %.3f\trecall score:%.3f\tf1 score:%.3f' % (precision[0][-1],
                                                                                           recall[0][-1], f1[0][-1]))
        # gaussian classifier
        clf = GaussianNB()
        clf.fit(train_data, train_label)
        result = clf.predict(test_data)
        temp = []
        for i in range(len(result)):
            lon = test_axis.iloc[i].loc['Longitude']
            lat = test_axis.iloc[i].loc['Latitude']
            temp.append(handle.grid_to_distance(lat, lon, result[i]))
        distance[1].append(sorted(temp))
        precision[1].append(precision_score(test_label, result, average='macro'))
        recall[1].append(recall_score(test_label, result, average='macro'))
        f1[1].append(f1_score(test_label, result, average='macro'))
        print('Gaussian:\t\tprecision score: %.3f\trecall score:%.3f\tf1 score:%.3f' % (precision[1][-1],
                                                                                           recall[1][-1], f1[1][-1]))
        # k neighbor
        clf = KNeighborsClassifier(n_neighbors=2)
        clf.fit(train_data, train_label)
        result = clf.predict(test_data)
        temp = []
        for i in range(len(result)):
            lon = test_axis.iloc[i].loc['Longitude']
            lat = test_axis.iloc[i].loc['Latitude']
            temp.append(handle.grid_to_distance(lat, lon, result[i]))
        distance[2].append(sorted(temp))
        precision[2].append(precision_score(test_label, result, average='macro'))
        recall[2].append(recall_score(test_label, result, average='macro'))
        f1[2].append(f1_score(test_label, result, average='macro'))
        print('k Neighbor:\t\tprecision score: %.3f\trecall score:%.3f\tf1 score:%.3f' % (precision[2][-1],
                                                                                           recall[2][-1], f1[2][-1]))
        # adaboost
        clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
        clf.fit(train_data, train_label)
        result = clf.predict(test_data)
        temp = []
        for i in range(len(result)):
            lon = test_axis.iloc[i].loc['Longitude']
            lat = test_axis.iloc[i].loc['Latitude']
            temp.append(handle.grid_to_distance(lat, lon, result[i]))
        distance[3].append(sorted(temp))
        precision[3].append(precision_score(test_label, result, average='macro'))
        recall[3].append(recall_score(test_label, result, average='macro'))
        f1[3].append(f1_score(test_label, result, average='macro'))
        print('AdaBoost:\t\tprecision score: %.3f\trecall score:%.3f\tf1 score:%.3f' % (precision[3][-1],
                                                                                           recall[3][-1], f1[3][-1]))
        # bagging
        clf = BaggingClassifier(n_estimators=10)
        clf.fit(train_data, train_label)
        result = clf.predict(test_data)
        temp = []
        for i in range(len(result)):
            lon = test_axis.iloc[i].loc['Longitude']
            lat = test_axis.iloc[i].loc['Latitude']
            temp.append(handle.grid_to_distance(lat, lon, result[i]))
        distance[4].append(sorted(temp))
        precision[4].append(precision_score(test_label, result, average='macro'))
        recall[4].append(recall_score(test_label, result, average='macro'))
        f1[4].append(f1_score(test_label, result, average='macro'))
        print('Bagging:\t\tprecision score: %.3f\trecall score:%.3f\tf1 score:%.3f' % (precision[4][-1],
                                                                                        recall[4][-1], f1[4][-1]))
        # random forest
        clf = RandomForestClassifier(n_estimators=40, max_depth=20)
        clf.fit(train_data, train_label)
        result = clf.predict(test_data)
        temp = []
        for i in range(len(result)):
            lon = test_axis.iloc[i].loc['Longitude']
            lat = test_axis.iloc[i].loc['Latitude']
            temp.append(handle.grid_to_distance(lat, lon, result[i]))
        distance[5].append(sorted(temp))
        precision[5].append(precision_score(test_label, result, average='macro'))
        recall[5].append(recall_score(test_label, result, average='macro'))
        f1[5].append(f1_score(test_label, result, average='macro'))
        print('Random Forest:\tprecision score: %.3f\trecall score:%.3f\tf1 score:%.3f' % (precision[5][-1],
                                                                                       recall[5][-1], f1[5][-1]))
        # # gradient boost
        # clf = GradientBoostingClassifier(n_estimators=10, min_samples_split=2)
        # clf.fit(train_data, train_label)
        # result = clf.predict(test_data)
        # temp = []
        # for i in range(len(result)):
        #     lon = test_axis.iloc[i].loc['Longitude']
        #     lat = test_axis.iloc[i].loc['Latitude']
        #     temp.append(handle.grid_to_distance(lat, lon, result[i]))
        # distance[6].append(sorted(temp))
        # precision[6].append(precision_score(test_label, result, average='macro'))
        # recall[6].append(recall_score(test_label, result, average='macro'))
        # f1[6].append(f1_score(test_label, result, average='macro'))
        # print('Gradient Boost:\tprecision score: %.3f\trecall score:%.3f\tf1 score:%.3f' % (precision[6][-1],
        #                                                                                    recall[6][-1], f1[6][-1]))

    avg_error = []
    new_distance = []
    for i in range(6):
        t = []
        temp = []
        for j in range(min([len(distance[i][k]) for k in range(10)])):
            s = 0
            for k in range(10):
                s += distance[i][k][j]
            s = s / 10
            t.append(s)
        for j in range(10):
            # temp.append(t[int(len(t) * (j+1) / 10) - 1])
            temp.append(sum(t[0:int(len(t) * (j + 1) / 10) - 1]) / int(len(t) * (j + 1) / 10))
        avg_error.append(temp)
        precision[i] = sum(precision[i]) / 10
        recall[i] = sum(recall[i]) / 10
        f1[i] = sum(f1[i]) / 10
    # CDF
    x_axis = [str(i+1) + '0%'  for i in range(10)]
    # plt.subplot(1, 2, 1)
    plt.plot(x_axis, avg_error[0], label='DecisionTree')
    plt.plot(x_axis, avg_error[1], label='GaussianNB')
    plt.plot(x_axis, avg_error[2], label='KNeighbor')
    plt.plot(x_axis, avg_error[3], label='AdaBoost')
    plt.plot(x_axis, avg_error[4], label='Bagging')
    plt.plot(x_axis, avg_error[5], label='Random Forest')
    plt.ylabel('error')
    plt.xlabel('the ratio of the smallest error of the classifier')
    plt.title('The CDF diagram of the classifier')
    plt.legend(loc='higher left')
    split_line = np.linspace(0, 500, 50)
    plt.plot(['50%' for i in range(50)], split_line, 'r--')
    plt.ylim([0, 500])
    plt.show()
    width = 0.2
    x_axis = ['DecisionTree', 'Gaussian', 'KNeighbor', 'AdaBoost', 'Bagging', 'Random Forest']
    plt.bar([i for i in range(1, 7, 1)], precision[0:-1], label='precision', tick_label=x_axis, width=width)
    plt.bar([i + width for i in range(1, 7, 1)], recall[0:-1], label='recall', width=width)
    plt.bar([i + width * 2 for i in range(1, 7, 1)], f1[0:-1], label='f1', width=width)
    plt.ylim([-0.05, 0.6])
    plt.xlabel('classifier')
    plt.ylabel('rank')
    plt.show()

