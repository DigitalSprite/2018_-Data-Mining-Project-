import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
from pyecharts import Line
from q1.a.handle_data import HandleData
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings('ignore')


# 首先对data进行分组分析
def group_data(file_name):
    data = pd.read_csv(file_name).drop(['MRTime'], axis=1)
    grouped_data = data.groupby('longitude_1')
    errors = []
    tag = 0
    # 存储每个基站的中位误差
    mid_errors = []
    for i, new_data in data.groupby(['longitude_1', 'latitude_1']):
        print('##############################################################')
        print('train with base: latitude: %f\t lonitude: %f' %(list(i)[0], list(i)[1]))
        tag += 1
        print('the data number is: %d' % (len(new_data.loc[:, 'longitude_1'])))
        # 对每一个基站按照经纬度进行分组
        labels = []
        for index, row in new_data.iterrows():
            # 更改每一条信息的基站标签
            # distance, width, height = calculate_distance(base_longitude, base_latitude, row[0], row[1])
            labels.append([row.loc['Longitude'] - row.loc['longitude_1'], row.loc['Latitude'] - row.loc['latitude_1']])
        # 得到训练以后的error值
        error = train(new_data, labels)
        errors += error
        mid_errors.append(sorted(error)[int(len(error) / 2)])
    # 返回基站中位误差 和 数据 和 所有点的误差值
    errors = sorted(errors)
    return mid_errors, data, errors


# 计算两点间的经纬度距离
def calculate_distance(longitude_1, latitude_1, longitude_2, latitude_2):
    width = (longitude_2 - longitude_1) * 2 * 6371 * math.pi * 1000 / 360
    height = (latitude_2 - latitude_1) * math.pi * 6371 * 1000 / 180
    return math.sqrt(width**2 + height**2), width, height


def train(data, label):
    train_data, test_data, train_label, test_label = train_test_split(data.drop(['Longitude', 'Latitude'], axis=1), label, random_state=0, test_size=0.2)
    regr = RandomForestRegressor()
    regr.fit(train_data, train_label)
    result = regr.predict(test_data)
    # print(result)
    tag = 0
    errors = []
    for i in list(test_data.index):
        longitude = test_data.loc[i,'longitude_1'] + result[tag][0]
        latitude = test_data.loc[i,'latitude_1'] + result[tag][1]
        distance, width, height = calculate_distance(longitude, latitude, data.loc[i,'Longitude'], data.loc[i,'Latitude'])
        errors.append(distance)
        tag += 1
    print('error: %f' % (sum(errors) / len(errors)))
    return errors


if __name__ == '__main__':
    mid_errors, data, errors = group_data('../a/data/data.csv')
    axis_label = [str(i+1)+'0%' for i in range(10)]
    # errors = [errors[int(len(errors) * 0.1 * (i + 1)) - 1] for i in range(10)]
    # print(errors)
    handle = HandleData()
    handle.handle_data_2g('../a/data/data.csv')
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
        clf = RandomForestClassifier(criterion='entropy')
        clf.fit(train_data, train_label)
        result = clf.predict(test_data)
        temp = []
        for i in range(len(result)):
            lon = test_axis.iloc[i].loc['Longitude']
            lat = test_axis.iloc[i].loc['Latitude']
            temp.append(handle.grid_to_distance(lat, lon, result[i]))
        distance.append(sorted(temp))
        precision[0].append(precision_score(test_label, result, average='macro'))
        recall[0].append(recall_score(test_label, result, average='macro'))
        f1[0].append(f1_score(test_label, result, average='macro'))
        print('Decision Tree:\tprecision score: %.3f\trecall score:%.3f\tf1 score:%.3f' % (precision[0][-1],
                                                                                      recall[0][-1], f1[0][-1]))
    raw_error = []
    for i in range(len(distance[0])):
        tag = True
        temp = 0
        for j in range(10):
            if i == len(distance[j]):
                tag = False
                break
            else:
                temp += distance[j][i]
        if tag is True:
            raw_error.append(temp/10)
    raw_error = [sum(raw_error[0:int(len(raw_error)*0.1*(i+1)) ]) / int(len(raw_error)*0.1*(i+1)) for i in range(10)]
    print(raw_error)
    mid = []
    index = [i for i in range(10)]
    for i in index:
        mid.append(sum(errors[0:int(len(errors) * 0.1 * (i+1) ) - 1]) / int(len(errors) * 0.1 * (i+1) - 1))
    plt.plot(axis_label, mid, label='c method')
    plt.plot(axis_label, raw_error, label='a method')
    plt.xlabel('verification time')
    plt.ylabel('error')
    plt.title('the relative error to the base')
    plt.legend(loc='upper left')
    plt.show()
