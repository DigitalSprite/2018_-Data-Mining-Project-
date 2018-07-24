# coding=utf-8
import pandas as pd
import math
import numpy as np

'''
读取data_2g.csv文件，提取出来信号特征和其相应的position ID
如果RNCID 或者CellID分别为0或者-1，则数据无效
'''


class HandleData:

    def __init__(self):
        # 经度最大值
        self.max_longitude = 0
        # 经度最小值
        self.min_longitude = 0
        # 纬度最大值
        self.max_latitude = 0
        # 纬度最小值
        self.min_latitude = 0
        # 网格宽
        self.grid_width = 0
        # 网格高
        self.grid_height = 0
        # 网格横向容纳个数
        self.grid_width_num = 0
        # 网格纵向容纳个数
        self.grid_height_num = 0
        # data_2g的数据
        self.data = None
        # 公参
        self.gongcan = None
        # 训练数据
        self.train_data = None
        # 测试数据
        self.test_data = None
        # 训练标签
        self.train_label = None
        # 测试标签
        self.test_label = None
        # 数据长度
        self.length = 0

    '''
    处理2g数据的函数
    '''
    def handle_data_2g(self, filename=None):
        if filename is not None:
            self.data = pd.read_csv(filename)
        else:
            print('handling data...')
            # 读取文件，设置参数
            self.data = pd.read_csv('../../data/data_2g.csv', encoding='utf-8').loc[:, ['Longitude', 'Latitude',
                                                                                        'RNCID_1', 'CellID_1',
                                                                                        'AsuLevel_1', 'RSSI_1',
                                                                                        'RNCID_2', 'CellID_2',
                                                                                       'AsuLevel_2', 'RSSI_2',
                                                                                       'RNCID_3', 'CellID_3',
                                                                                       'AsuLevel_3', 'RSSI_3',
                                                                                       'RNCID_4', 'CellID_4',
                                                                                       'AsuLevel_4', 'RSSI_4',
                                                                                           'RNCID_5', 'CellID_5',
                                                                                           'AsuLevel_5', 'RSSI_5',
                                                                                           'RNCID_6', 'CellID_6',
                                                                                           'AsuLevel_6', 'RSSI_6',
                                                                                           'RNCID_7', 'CellID_7',
                                                                                           'AsuLevel_7', 'RSSI_7',
                                                                                           'MRTime', 'IMSI']]
        self.max_longitude = max(self.data.loc[:, 'Longitude'])
        self.min_longitude = min(self.data.loc[:, 'Longitude'])
        self.max_latitude = max(self.data.loc[:, 'Latitude'])
        self.min_latitude = min(self.data.loc[:, 'Latitude'])
        self.grid_width = self.get_distance(self.min_latitude, self.min_longitude, self.min_latitude, self.max_longitude)
        self.grid_height = self.get_distance(self.min_latitude, self.min_longitude, self.max_latitude, self.min_longitude)
        self.grid_width_num = int(self.grid_width / 20)+1
        self.grid_height_num = int(self.grid_height / 20)+1
        self.gongcan = pd.read_csv('../../data/2g_gongcan.csv', encoding='utf-8')
        self.length = len(self.data.loc[:, 'Longitude'])
        if filename is None:
            # 插入Position的列，来代替经纬度坐标
            self.data['Position'] = 0
            # 将基站的RNCID和CellID转化为基站所对应的经纬度值,做了左连接的操作
            for index in range(7):
                self.gongcan.columns = ['RNCID', 'CellID', 'latitude_' + str(index + 1), 'longitude_' + str(index + 1)]
                self.data = pd.merge(left=self.data, right=self.gongcan, how='left',
                                     left_on=['RNCID_' + str(index + 1), 'CellID_' + str(index + 1)],
                                     right_on=['RNCID', 'CellID']) \
                    .drop(['RNCID_' + str(index + 1), 'CellID_' + str(index + 1), 'RNCID', 'CellID'], axis=1)
            # 将人所在位置替换为栅格所在ID
            for index, raw in self.data.iterrows():
                self.data.loc[index, 'Position'] = int(self.calculate_grid_position(self.data.loc[index, 'Longitude'],
                                                                                    self.data.loc[index, 'Latitude']))
            self.data = self.data.fillna(0)
            self.data.to_csv('data/data.csv', index=False, sep=',')

    '''
    将经纬度值换算成栅格ID
    '''
    def calculate_grid_position(self, longitude, latitude):
        lat1 = self.min_latitude
        lon1 = self.min_longitude
        lat2 = latitude
        lon2 = longitude
        x = int(self.get_distance(lat1, lon1, lat1, lon2) / 20) + 1
        y = int(self.get_distance(lat1, lon1, lat2, lon1) / 20) + 1
        return (y - 1) * self.grid_width_num + x

    # 经纬度换算成距离
    def get_distance(self, lat1, lon1, lat2, lon2):
        lat1 = (math.pi / 180) * lat1
        lat2 = (math.pi / 180) * lat2
        lon1 = (math.pi / 180) * lon1
        lon2 = (math.pi / 180) * lon2
        d = math.acos(math.sin(lat1)* math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)) * 6371 * 1000
        return d

    def grid_to_distance(self, lat1, lon1, grid_tag):
        y = int(grid_tag / self.grid_width_num)
        x = grid_tag % self.grid_width_num
        if x != 0:
            y += 1
        distance_x = 20 * x - 10
        distance_y = 20 * y - 10
        distance_x_2 = self.get_distance(self.min_latitude, self.min_longitude, self.min_latitude, lon1)
        distance_y_2 = self.get_distance(self.min_latitude, self.min_longitude, lat1, self.min_longitude)
        return math.sqrt((distance_x - distance_x_2)**2 + (distance_y - distance_y_2)**2)

    '''
    计算测试出来的误差
    将栅格的中心位置作为预测的经纬度，和test_label中的真实的经纬度进行欧式距离的计算，算出误差值，单位是米
    '''
    def calculate_error(self, number, real_longitude, real_latitude):
        x = number % self.grid_width_num
        y = int(number / self.grid_width_num + 1)
        if x < self.grid_width_num:
            longitude = self.min_longitude + (20 * (x - 1) + 10) / (2 * math.pi * 6371.004 * 1000) * 360
        else:
            longitude = self.min_longitude + (20 * (self.grid_width_num - 1) +
                                              (self.grid_width - 20 * (self.grid_width_num - 1)) * 1/2) \
                                                / (2 * math.pi * 6371.004 * 1000) * 360
        if y < self.grid_height_num - 1:
            latitude = self.min_latitude + (y * 20 + 10) / (math.pi * 6371.004 * 1000) * 180
        else:
            latitude = self.min_latitude + (y * 20 + 0.5 * (self.grid_height - y * 20)) / ( math.pi * 6371.004 * 1000) * 180
        error = math.sqrt(np.square((longitude - real_longitude) / 360 * 2 * math.pi * 6371.004 * 1000) +
                          np.square(latitude - real_latitude) / 180 * math.pi * 6371.004 * 1000)
        return error
