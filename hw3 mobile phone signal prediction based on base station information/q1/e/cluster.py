import q1.c.group as gp
import q1.d.compare as cp
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.ensemble import RandomForestRegressor
from pyecharts import Line, Scatter
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def handle_data(mid_errors, data):
    errors = pd.DataFrame(mid_errors, columns=['Error'])
    number = int(len(mid_errors) * 0.2)
    errors = errors.sort_values(by='Error')
    # 得到中值误差最坏的集合索引
    tail_index = list(errors.tail(number).index)
    group = data.groupby('longitude_1')
    tag = 0
    base_positions = {}
    data_list = []
    base_longitudes = []
    base_latitudes = []
    for base_longitude, new_data in group:
        new_group = new_data.groupby('latitude_1')
        for base_latitude, new_data in new_group:
            base_positions.update({tag:[base_longitude, base_latitude]})
            base_longitudes.append(base_longitude)
            base_latitudes.append(base_latitude)
            data_list.append(new_data)
            tag += 1
    # 生成点状图查看基站的分布
    # scatter = Scatter('base station point')
    # scatter.add('all',base_longitudes, base_latitudes, xaxis_max=max(base_longitudes), xaxis_min=min(base_longitudes),
    #             yaxis_min= min(base_latitudes), yaxis_max=max(base_latitudes), symbol_size=4 )
    # scatter.add('tail', [base_longitudes[i] for i in tail_index], [base_latitudes[i] for i in tail_index], xaxis_max=max(base_longitudes), xaxis_min=min(base_longitudes),
    #             yaxis_min= min(base_latitudes), yaxis_max=max(base_latitudes), symbol_size=4)
    # scatter.render('figure/point.html')
    # 使用KMeans聚类
    data_set = []
    for i in range(len(base_latitudes)):
        data_set.append([base_longitudes[i], base_latitudes[i]])
    pred = KMeans(init='k-means++').fit_predict(data_set)
    # 将差的数据集根据predict的标签做合并
    for i in tail_index:
        for j in range(len(pred)):
            if pred[j] == pred[i] and j != i:
                data_list[i] = pd.concat([data_list[i], data_list[j]])
    # 做交叉验证
    mid_errors = []
    for i in range(10):
        tag = 0
        for temp_data in data_list:
            labels = []
            for index, row in temp_data.iterrows():
                labels.append([row.loc['Longitude'] - row.loc['longitude_1'],
                               row.loc['Latitude'] - row.loc['latitude_1']])
            train_data, test_data, train_label, test_label = \
                train_test_split(temp_data, labels, random_state=np.random.randint(1,100), test_size=0.2)
            regr = RandomForestRegressor()
            regr.fit(train_data, train_label)
            result  = regr.predict(test_data)
            for loc in range(len(result)):
                if i == 0:
                    mid_errors.append([])
                # distance, w, h = calculate_distance(result[loc][0], result[loc][1], test_label.iloc[loc,0],
                #                                     test_label.iloc[loc,1])
                distance, width, height = calculate_distance(test_data.iloc[loc].loc['longitude_1'] + result[loc][0],
                                                             test_data.iloc[loc].loc['latitude_1'] + result[loc][1],
                                                             test_data.iloc[loc].loc['Longitude'],
                                                             test_data.iloc[loc].loc['Latitude'])
                mid_errors[tag].append(distance)
                tag += 1
    for err in range(len(mid_errors)):
        mid_errors[err] = sorted(mid_errors[err])[5]
    print(mid_errors)
    return mid_errors


def calculate_distance(longitude_1, latitude_1, longitude_2, latitude_2):
    width = (longitude_2 - longitude_1) * 2 * 6371 * math.pi * 1000 / 360
    height = (latitude_2 - latitude_1) * math.pi * 6371 * 1000 / 180
    return math.sqrt(width**2 + height**2), width, height


if __name__ == '__main__':
    mid_errors, data, errors = gp.group_data('../a/data/a.csv')
    new_mid_e = sorted(handle_data(mid_errors, data))
    new_mid_d = cp.handle_data(mid_errors, data)
    index = []
    e = []
    for i in range(1,10,1):
        index.append('{0}0%'.format(i))
        e.append(round(sorted(new_mid_e)[int(len(new_mid_e) * 0.1 * i)], 2))
    line = Line('Compare')
    # line.add('d', index, new_mid_d, is_label_show=True)
    # line.add('e', index, e, is_label_show=True)
    # line.render('figure/d&e.html')
    plt.plot(index, new_mid_d, label='d')
    plt.plot(index, e, label='e')
    plt.xlabel('ratio')
    plt.ylabel('error')
    plt.legend(loc='upper left')
    plt.title('compare d & e')
    plt.show()
