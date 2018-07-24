import q1.c.group as gp
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt


def handle_data(errors, data):
    random_state = np.random.RandomState()
    number = int(len(errors) * 0.2)
    errors = pd.DataFrame(errors, columns=['error'])
    errors = errors.sort_values(by='error')
    head_index = errors.head(number).index
    tail_index = sorted(errors.tail(number).index)
    grouped = data.groupby(by='longitude_1')
    # 将不同主基站的信息存储在数组里
    data_list = []
    for base_longitude, grouped_temp in grouped:
        grouped_temp = grouped_temp.groupby(by='latitude_1')
        for base_latitude , new_data in grouped_temp:
            data_list.append(new_data)
    # 补全那些不好的数据
    for i in range(number):
        data_list[tail_index[i]] = pd.concat([data_list[head_index[i]], data_list[tail_index[i]]])
    # 进行交叉验证
    new_errors = []
    for r in range(10):
        # 对每个部分单独进行回归检测
        temp = 0
        for item in data_list:
            labels = []
            for index, row in item.iterrows():
                labels.append([row.loc['Longitude'] - row.loc['longitude_1'],
                               row.loc['Latitude'] - row.loc['latitude_1']])
            train_data, test_data, train_label, test_label = train_test_split(item, labels,
                                                                              random_state=random_state.randint(1, 100),
                                                                              test_size=0.2)
            regr = RandomForestRegressor()
            regr.fit(train_data, train_label)
            result = regr.predict(test_data)
            tag = 0
            if r == 0:
                for i in range(len(result)):
                    new_errors.append([])
            for i in test_data.index:
                distance, width, height = calculate_distance(test_data.loc[i, 'longitude_1'] + result[tag][0],
                                                             test_data.loc[i, 'latitude_1'] +
                                                             result[tag][1], test_data.loc[i, 'Longitude'],
                                                             test_data.loc[i, 'Latitude'])
                new_errors[temp].append(distance)
                tag += 1
                temp += 1
    for i in range(len(new_errors)):
        new_errors[i] = sorted(new_errors[i])[5]
    mid_errors = []
    for i in range(1,10,1):
        mid_errors.append(round(sorted(new_errors)[int(len(new_errors) * 0.1 * i)],2))
    return mid_errors


# 计算两点间的经纬度距离
def calculate_distance(longitude_1, latitude_1, longitude_2, latitude_2):
    width = (longitude_2 - longitude_1) * 2 * 6371 * math.pi * 1000 / 360
    height = (latitude_2 - latitude_1) * math.pi * 6371 * 1000 / 180
    return math.sqrt(width**2 + height**2), width, height


if __name__ == '__main__':
    mid_errors, data, errors = gp.group_data('../a/data/a.csv')
    new_errors = []
    for i in range(1,10,1):
        new_errors.append(round(sorted(errors)[int(len(errors) * 0.1 * i)], 2))
    # print(tag)
    new_mid_errors = handle_data(mid_errors, data)
    axis_label = [str(i + 1) + '0%' for i in range(9)]
    # 将d,c两问的
    print(new_errors)
    plt.plot(axis_label, new_errors, label='c')
    plt.plot(axis_label, new_mid_errors, label='d')
    plt.xlabel('ratio')
    plt.ylabel('error')
    plt.title('compare')
    plt.legend(loc='upper left')
    plt.show()
