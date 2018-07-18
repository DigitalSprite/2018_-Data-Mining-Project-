import pandas as pd
from collections import defaultdict
import time
from b.prefix_span.prefixspan import PrefixSpan
from bokeh.plotting import figure,output_file, show


class BI:

    def __init__(self):
        self._result = None
        self._time = 0

    def get_data(self, property, file=1):
        if file == 1:
            data = pd.read_csv('../data/trade.csv', encoding='utf-8', dtype=(str)).loc[:,
                   ['vipno', property, 'sldat', 'uid']].sort_values(
                by=['uid', 'sldat']).set_index('vipno')
        elif file == 2:
            data = pd.read_csv('../data/trade_new.csv', encoding='utf-8', dtype=(str)).loc[:,
                   ['vipno', property, 'sldatime', 'uid']].sort_values(
                by=['uid', 'sldatime']).set_index('vipno')
            data.rename(columns={'sldatime': 'sldat'}, inplace=True)
        else:
            raise ValueError('Incorrect file name')
        indexs = data.index.drop_duplicates()
        transactions = pd.DataFrame(columns=[property, 'uid'], dtype=(str))
        for index in indexs:
            temp_data = data.loc[index]
            temp_data = temp_data.head(int(len(temp_data) * 0.6))
            transactions = pd.concat([temp_data, transactions])
        data = defaultdict(lambda: [])
        for item in transactions.iterrows():
            if [item[1][property]] not in data[item[0]]:
                data[item[0]].append([item[1][property]])
        return list(data.values())

    def run_algorithm(self, property, support, file):
        start = time.time()
        data = self.get_data(property, file)
        model = PrefixSpan.train(data, minSupport=support, maxPatternLength=5)
        result = model.freqSequences().collect()
        end = time.time()
        self._time = end - start
        self._result = result
        return result

    def show_info(self):
        for fs in self._result:
            print('{}: {}'.format(fs.sequence, fs.freq))
        print('consuming time: ' + str(self._time) + 's')

    def get_time(self):
        return self._time

if __name__ == '__main__':
    test = BI()
    # test.run_algorithm('pluno',64,1)
    # test.show_info()
    # algorithm = BI()

    threshold = [2, 4, 8, 16, 32, 64]
    tag = ['pluno', 'bndno', 'dptno']

    ''' build number-threshold figure'''
    result = []
    for i in tag:
        temp = []
        for j in threshold:
            r = test.run_algorithm(i, j, 2)
            temp.append(len(r))
        result.append(temp)
    # test.show_info()
    print(result)
    output_file('number-threshold-bi.html')
    p = figure(
        tools='pan,box_zoom,reset,save',
        y_range=[0, 80000], title='number-threshold',
        x_axis_label='minsup', y_axis_label='number'
    )
    print(result)
    p.line(threshold, result[0], legend=tag[0], line_color='red')
    p.line(threshold, result[1], legend=tag[1], line_color='orange')
    p.circle(threshold, result[1], legend=tag[1], line_color='orange', fill_color='white')
    p.line(threshold, result[2], legend=tag[2], line_color='green', line_dash='4 4')
    show(p)

