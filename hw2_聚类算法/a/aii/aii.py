from a.fpgrowth.fp_growth import find_itemset_by_frequency
import pandas as pd
from collections import defaultdict
import time
from a.ai import AI
from bokeh.plotting import figure,output_file, show


class AII:

    def __init__(self):
        self._time = 0
        self._result = None

    def get_data(self, property, file=1):
        """
        get train data for fpgrowth algorithm
        :param support: selected from [2,4,8,16,32,64]
        :param file: 1 for trade.csv and 2 for trade_new.csv
        :return:
        """
        if file == 1:
            data = pd.read_csv('../data/trade.csv', encoding='utf-8', dtype=(str)).loc[:,
                   ['vipno', property, 'sldat']].sort_values(
                by=['vipno', 'sldat']).set_index('vipno').loc[:, [property]]
        elif file == 2:
            data = pd.read_csv('../data/trade_new.csv', encoding='utf-8', dtype=(str)).loc[:,
                   ['vipno', property, 'sldatime']].sort_values(
                by=['vipno', 'sldatime']).set_index('vipno').loc[:, [property]]
            data.rename(columns={'sldatime': 'sldat'}, inplace=True)
        else:
            raise ValueError("file name error!")
        indexs = data.index.drop_duplicates()
        transactions = defaultdict(lambda: [])
        temp_transactions = pd.DataFrame(columns=[property], dtype=(str))
        for index in indexs:
            temp_data = data.loc[index]
            temp_data = temp_data.head(int(len(temp_data) * 0.6))
            temp_transactions = pd.concat([temp_transactions, temp_data])
        for item in temp_transactions.iterrows():
            if item[1][property] not in transactions[item[0]]:
                transactions[item[0]].append(item[1][property])
        return list(transactions.values())

    def run_algorithm(self, property, support, file=1):
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
    test = AII()
    threshold = [2, 4, 6, 8, 10]
    tag = ['pluno', 'bndno', 'dptno']

    ''' build time-threshold figure '''
    result = []
    time_result=[]
    for i in tag:
        temp = []
        time_temp=[]
        for j in threshold:
            r = test.run_algorithm(i,j,1)
            temp.append(len(r))
            time_temp.append(test.get_time())
        result.append(temp)
        time_result.append(time_temp)
    # test.show_info()
    # print(result)
    # output_file('number-threshold-aii.html')
    # p = figure(
    #     tools='pan,box_zoom,reset,save',
    #     y_range=[0, 200000], title='number-threshold',
    #     x_axis_label='threshold', y_axis_label='number'
    # )
    # print(result)
    # p.line(threshold, result[0], legend=tag[0], line_color='red')
    # p.line(threshold, result[1], legend=tag[1], line_color='orange')
    # p.circle(threshold, result[1], legend=tag[1], line_color='orange', fill_color='white')
    # p.line(threshold, result[2], legend=tag[2], line_color='green', line_dash='4 4')
    # show(p)


    output_file('time-threshold-aii.html')
    p2 = figure(
        tools='pan,box_zoom,reset,save',
        y_range=[0, 60], title='time-threshold-aii',
        x_axis_label='threshold', y_axis_label='time'
    )
    print(time_result)
    p2.line(threshold, time_result[0], legend=tag[0], line_color='red')
    p2.line(threshold, time_result[1], legend=tag[1], line_color='orange')
    p2.circle(threshold, time_result[1], legend=tag[1], line_color='orange', fill_color='white')
    p2.line(threshold, time_result[2], legend=tag[2], line_color='green', line_dash='4 4')
    show(p2)