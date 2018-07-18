from a.fpgrowth.fp_growth import find_itemset_by_frequency
import pandas as pd
from collections import defaultdict
import time
from bokeh.plotting import figure,output_file, show


class AI:

    def __init__(self):
        self._result = None
        self._time = 0

    def get_data(self, property, file=1):
        """
        get train data for fpgrowth algorithm
        :param support: selected from [2,4,8,16,32,64]
        :param file: 1 for trade.csv and 2 for trade_new.csv
        :return:
        """
        # get data from trade.csv and select vipno, property, sldat, uid and property as the handled data for training
        if file == 1:
            data = pd.read_csv('../data/trade.csv', encoding='utf-8', dtype=(str)).loc[:,
                   ['vipno', property, 'sldat', 'uid']].sort_values(
                by=['uid', 'sldat']).set_index('vipno').loc[:, [property, 'uid']]
        # get data from trade_new.csv and do the same operation as front
        elif file == 2:
            data = pd.read_csv('../data/trade_new.csv', encoding='utf-8', dtype=(str)).loc[:,
                   ['vipno', property, 'sldatime', 'uid']].sort_values(
                by=['uid', 'sldatime']).set_index('vipno').loc[:, [property, 'uid']]
            data.rename(columns={'sldatime': 'sldat'}, inplace=True)
        # drop multiple item in one transaction
        indexs = data.index.drop_duplicates()
        # create transaction as the train data
        transactions = pd.DataFrame(columns=[property, 'uid'], dtype=(str))
        for index in indexs:
            temp_data = data.loc[index]
            temp_data = temp_data.head(int(len(temp_data) * 0.6))
            transactions = pd.concat([temp_data, transactions])
        #select 'uid' as index
        new_transaction = transactions.set_index('uid')
        data = defaultdict(lambda: [])
        for item in new_transaction.iterrows():
            if item[1][property] not in data[item[0]]:
                data[item[0]].append(item[1][property])
        return list(data.values())

    def run_algorithm(self, property,support, file=1):
        '''
        the function to run the fp-growth algorithm
        :param property: pluno, bndno or dptno
        :param support: selected from [2,4,8,16,32,64]
        :param file: 1 for trade.csv and 2 for trade_new.csv
        :return:
        '''
        start = time.time()
        data = self.get_data(property, file)
        result = []
        for itemset, support in find_itemset_by_frequency(data, support):
            result.append((itemset, support))
        end = time.time()
        self._time = end - start
        self._result = result
        return result

    def show_info(self):
        for itemset, support in self._result:
            print(str(itemset) + ' : ' + str(support))
        print('consuming time: ' + str(self._time) + 's')

    def get_time(self):
        return self._time

if __name__ == '__main__':
    # create AI object and run the algorithm
    test = AI()
    threshold = [2,4,8,16,32,64]
    tag = ['pluno','bndno','dptno']

    ''' build time-threshold figure '''
    # result = []
    # for i in tag:
    #     temp = []
    #     for j in threshold:
    #         test.run_algorithm(i,j,1)
    #         temp.append(test.get_time())
    #     result.append(temp)
    # # test.show_info()
    # print(result)
    # output_file('time-threshold.html')
    # p = figure(
    #     tools='pan,box_zoom,reset,save',
    #     y_range=[1,1.5], title='ai figure',
    #     x_axis_label='threshold',y_axis_label='time(s)'
    # )
    # p.line(threshold, result[0], legend=tag[0], line_color='red')
    # p.line(threshold,result[1],legend=tag[1],line_color='orange')
    # p.line(threshold, result[2], legend=tag[2], line_color='green')
    # show(p)

    ''' build number-threshold figure'''
    result = []
    for i in tag:
        temp = []
        for j in threshold:
            r = test.run_algorithm(i, j, 1)
            temp.append(len(r))
        result.append(temp)

    # create diagram by using bokeh
    output_file('number-threshold.html')
    p = figure(
        tools='pan,box_zoom,reset,save',
        y_range=[0,4500], title='number-threshold',
        x_axis_label='threshold', y_axis_label='number'
    )
    print(result)
    p.line(threshold, result[0], legend=tag[0], line_color='red')
    p.line(threshold, result[1], legend=tag[1], line_color='orange')
    p.circle(threshold, result[1], legend=tag[1], line_color='orange', fill_color='white')
    p.line(threshold, result[2], legend=tag[2], line_color='green', line_dash='4 4')
    show(p)