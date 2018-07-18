import pandas as pd
import math
from sklearn import preprocessing
import numpy as np


def handle_data(from_file = False):
    if from_file is False:
        data = pd.read_csv('data/data_2g.csv', encoding='utf-8').drop(['IMSI', 'Num_connected'], axis=1)
        data = data.fillna(0)
        gongcan = pd.read_csv('data/2g_gongcan.csv', encoding='utf-8')
        # print(gongcan)
        new_data = pd.DataFrame()
        # new_data = pd.concat([new_data, data])
        min_lon = min(data.loc[:, 'Longitude'])
        max_lon = max(data.loc[:, 'Longitude'])
        min_lat = min(data.loc[:, 'Latitude'])
        max_lat = max(data.loc[:, 'Latitude'])
        grid_length = int(get_distance(min_lat, min_lon, min_lat, max_lon) / 20) + 1
        print(grid_length)
        data['tag'] = 0
        for i in range(len(data.loc[:, 'Longitude'])):
            data.loc[i, 'tag'] = calculate_grid_position(min_lat, min_lon,
                                                         data.loc[i, 'Latitude'], data.loc[i, 'Longitude'], grid_length)
        for i in range(1, 7):
            temp_data = data.loc[:, ['RSSI_%d' % i, 'AsuLevel_%d' % i,
                                     'RNCID_%d' % i, 'CellID_%d' % i, 'MRTime']]
            temp_data = pd.merge(left=temp_data, right=gongcan, how='left', left_on=['RNCID_%d' % i, 'CellID_%d' % i],
                                 right_on=['RNCID', 'CellID']).drop(['RNCID', 'CellID'], axis=1)
            temp_data.columns = ['RSSI_%d' % i, 'AsuLevel_%d' % i, 'RNCID_%d' % i, 'CellID_%d' % i, 'MRTime_%d' % i,
                                 'Lat%d' % i, "Lon%d" % i]
            new_data = pd.concat([new_data, temp_data], join='outer', axis=1)
        index = []
        for i in range(1, 7):
            index += ['MRTime_%d' % i, 'Lon%d' % i, 'Lat%d' % i, 'RSSI_%d' % i, 'AsuLevel_%d' % i, 'RNCID_%d' % i,
                      'CellID_%d' % i]
        new_data = new_data.ix[:, index]
        new_data = new_data.fillna(0)
        new_data = (new_data - new_data.min()) / (new_data.max() - new_data.min())
        new_data = pd.concat([new_data, data.loc[:, 'tag']], join='outer', axis=1)
        new_data.to_csv('data/handled_data.csv', index=False)
    else:
        new_data = pd.read_csv('data/handled_data.csv')
    enc = preprocessing.OneHotEncoder()
    temp = np.reshape(new_data.loc[:, 'tag'], newshape=(-1, 1))
    enc = enc.fit_transform(temp).toarray()
    label = []
    for i in range(len(temp)):
        label.append(list(enc[temp[i][0]]))
    new_data = new_data.drop(['tag'], axis=1)
    new_data = new_data.values
    new_data = np.reshape(new_data, newshape=(-1, 6, 7))
    new_data = new_data.reshape([-1, 6, 7, 1])
    return new_data, label


def calculate_grid_position(lat1, lon1, lat2, lon2, width):
    x = int(get_distance(lat1, lon1, lat1, lon2) / 20) + 1
    y = int(get_distance(lat1, lon1, lat2, lon1) / 20) + 1
    return (y - 1) * width + x


def get_distance(lat1, lon1, lat2, lon2):
    lat1 = (math.pi / 180) * lat1
    lat2 = (math.pi / 180) * lat2
    lon1 = (math.pi / 180) * lon1
    lon2 = (math.pi / 180) * lon2
    d = math.acos(math.sin(lat1)* math.sin(lat2) +
                  math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)) * 6371 * 1000
    return d


handle_data(True)

# print(new_data.columns)
# print(new_data)

# print(new_data)
