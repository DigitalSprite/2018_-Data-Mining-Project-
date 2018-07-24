from sklearn.tree import DecisionTreeClassifier
from q1.a.handle_data import HandleData
from sklearn.model_selection import KFold
import math
import matplotlib.pyplot as plt


def grid_to_distance(grid_tag_1, grid_tag_2):
    y = int(grid_tag_1 / 82)
    x = grid_tag_1 % 82
    if x != 0:
        y += 1
    y2 = int(grid_tag_2 / 82)
    x2 = grid_tag_2 % 82
    if x2 != 0:
        y2 += 1
    distance_x = 20 * x - 10
    distance_y = 20 * y - 10
    distance_x_2 = 20 * x2 - 10
    distance_y_2 = 20 * y2 - 10
    return math.sqrt((distance_x - distance_x_2) ** 2 + (distance_y - distance_y_2) ** 2)


def calculate_distance(longitude_1, latitude_1, longitude_2, latitude_2):
    width = (longitude_2 - longitude_1) * 2 * 6371 * math.pi * 1000 / 360
    height = (latitude_2 - latitude_1) * math.pi * 6371 / 180
    return math.sqrt(width**2 + height**2)


def optimize_test_label():
    handle = HandleData()
    handle.handle_data_2g('../a/data/data.csv')
    # handle.data.sort_values(by=['MRTime'])

    # 得到原始的数据
    raw_data = handle.data
    plt.scatter(raw_data.loc[:, 'Longitude'], raw_data.loc[:, 'Latitude'], s=3)
    plt.xlim([handle.min_longitude, handle.max_longitude])
    plt.ylim([handle.min_latitude, handle.max_latitude])
    plt.title('the primitive figure')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.show()
    data = handle.data
    for index, new_data in handle.data.groupby('IMSI'):
        new_data = new_data.sort_values(by='MRTime')
        # 前一个坐标经度
        last_longitude = 0
        # 前一个坐标纬度
        last_latitude = 0
        last_longitude_2 = 0
        last_latitude_2 = 0
        # 前一个时间
        last_time = 0
        # 标记前两个点的速度
        last_speed = 0
        tag = 0
        speed_x = 0
        speed_y = 0
        for index_2, temp_data in new_data.iterrows():
            if last_latitude == 0:
                last_latitude = float(temp_data.loc['Latitude'])
                last_longitude = float(temp_data.loc['Longitude'])
                last_time = float(temp_data.loc['MRTime'])
                continue
            longitude = float(temp_data.loc['Longitude'])
            latitude = float(temp_data.loc['Latitude'])
            time = float(temp_data.loc['MRTime'])
            speed = calculate_distance(latitude, longitude, last_latitude, last_longitude) / (time - last_time) * 1000
            # 如果有较大的偏差, 修改坐标值
            if speed > 5 and last_speed > 5:
                print(data.loc[tag, 'Longitude'])
                data.loc[tag, 'Longitude'] = last_longitude_2
                data.loc[tag, 'Latitude'] = last_latitude_2
                print(data.loc[tag, 'Longitude'])
            speed_x = (temp_data.loc['Longitude'] - last_longitude) / (time - last_time) * 1000
            speed_y = (temp_data.loc['Latitude'] - last_latitude) / (time - last_time) * 1000
            last_latitude_2 = last_latitude
            last_longitude_2 = last_longitude
            last_latitude = float(temp_data.loc['Latitude'])
            last_longitude = float(temp_data.loc['Longitude'])
            last_time = float(temp_data.loc['MRTime'])
            last_speed = speed
            tag = index_2

    plt.scatter(data.loc[:, 'Longitude'], data.loc[:, 'Latitude'], s=3)
    plt.xlim([handle.min_longitude, handle.max_longitude])
    plt.ylim([handle.min_latitude, handle.max_latitude])
    plt.title('the fixed figure')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.show()
    for i, raw in data.iterrows():
        data.loc[i, 'Position'] = int(handle.calculate_grid_position(data.loc[i, 'Longitude'], data.loc[i, 'Latitude']))
    label = data.loc[:, 'Position']
    raw_label = raw_data.loc[:, 'Position']
    data = data.drop(['Longitude', 'Latitude', 'Position', 'MRTime', 'IMSI'], axis=1)
    raw_data = raw_data.drop(['Longitude', 'Latitude', 'Position', 'MRTime', 'IMSI'], axis=1)
    kf = KFold(n_splits=10, shuffle=True)
    error1 = []
    error2 = []
    for train_index, test_index in kf.split(data):
        #
        train_data = data.loc[train_index, :]
        train_label = label.loc[train_index]
        test_data = data.loc[test_index, :]
        test_label = label.loc[test_index]
        # 使用分类器进行训练
        clf = DecisionTreeClassifier()
        clf.fit(train_data, train_label)
        result = clf.predict(test_data)
        error = []
        for i in range(len(result)):
            error.append(grid_to_distance(test_label.iloc[i], result[i]))
        error1.append(sorted(error))
        train_data = raw_data.loc[train_index, :]
        train_label = raw_label.loc[train_index]
        test_data = raw_data.loc[test_index, :]
        test_label = raw_label.loc[test_index]
        # 使用分类器进行训练
        clf = DecisionTreeClassifier()
        clf.fit(train_data, train_label)
        result = clf.predict(test_data)
        raw_error = []
        for i in range(len(result)):
            raw_error.append(grid_to_distance(test_label.iloc[i], result[i]))
        error2.append(sorted(raw_error))
    new_error_1 = []
    new_error_2 = []
    e1 = []
    e2 = []
    for i in range(len(error1[0])):
        temp = 0
        tag = True
        for j in range(10):
            if len(error1[j]) == i:
                tag = False
                break
            temp += error1[j][i]
        if tag is True:
            new_error_1.append(temp)
    for i in range(len(error2[0])):
        temp = 0
        tag = True
        for j in range(10):
            if len(error2[j]) == i:
                tag = False
                break
            temp += error2[j][i]
        if tag is True:
            new_error_2.append(temp)
    for i in range(10):
        e1.append(sum(new_error_1[0: int((i + 1) * len(new_error_1) / 10) - 1]) / (int((i + 1) * len(new_error_1) / 10) - 1))
        e2.append(
            sum(new_error_2[0:int((i + 1) * len(new_error_2) / 10) - 1]) / (int((i + 1) * len(new_error_2) / 10) - 1))
    x_axis = [str(i + 1) + '0%' for i in range(10)]
    print(e2)
    print(e1)
    plt.plot(x_axis, e1, label='fixed_data')
    plt.plot(x_axis, e2, label='raw_data')
    plt.legend(loc='upper left')
    plt.show()


optimize_test_label()