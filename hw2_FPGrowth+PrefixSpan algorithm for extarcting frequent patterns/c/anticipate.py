import pandas as pd
from collections import defaultdict
from a.ai.ai import AI
from a.aii.aii import AII
from b.bi.bi import BI
from b.bii.bii import BII
import time as tm
from bokeh.plotting import figure,output_file, show


class Anticipate:

    def __init__(self):
        self._transactions = None
        self._train_data = None
        self._test_data = None
        self._index = None
        self._accuracy = 0.0
        self._time = 0
        self._correct = 0
        self._total = 0


    def get_data(self, file):
        if file == 1:
            data = pd.read_csv('../data/trade.csv', encoding='utf-8', dtype=(str)).loc[:,
                   ['vipno', 'pluno', 'dptno', 'bndno', 'sldat', 'uid']].sort_values(
                by=['vipno', 'sldat']).set_index('vipno')
        elif file == 2:
            data = pd.read_csv('../data/trade_new.csv', encoding='utf-8', dtype=(str)).loc[:,
                   ['vipno', 'pluno', 'dptno', 'bndno', 'sldatime', 'uid']].sort_values(
                by=['vipno', 'sldatime']).set_index('vipno')
            data.rename(columns={'sldatime': 'sldat'}, inplace=True)
        else:
            raise ValueError('Incorrect file name')
        indexs = data.index.drop_duplicates()
        self._index = indexs
        self._transactions = data
        train_data = pd.DataFrame(columns=['pluno', 'dptno', 'bndno', 'sldat', 'uid'], dtype=(str))
        test_data = pd.DataFrame(columns=['pluno', 'dptno', 'bndno', 'sldat', 'uid'], dtype=(str))
        for index in indexs:
            temp_data = data.loc[index]
            temp_data1 = temp_data.tail(len(temp_data) - int(len(temp_data) * 0.6))
            test_data = pd.concat([temp_data1,test_data])
            temp_data2 = temp_data.head(int(len(temp_data) * 0.6))
            train_data = pd.concat([temp_data2, train_data])
        self._train_data = train_data
        self._test_data = test_data

    def get_train_data_by_vipno(self, vipno):
        return self._train_data.loc[vipno]

    def get_test_data_by_vipno(self, vipno):
        return self._test_data.loc[vipno]

    def get_index(self):
        return list(self._index)

    def get_test_by_property(self, vipno, property):
        return list(self.get_test_data_by_vipno(vipno).loc[:,property])

    def get_train_by_property(self, vipno, property):
        return list(self.get_train_data_by_vipno(vipno).loc[:,property])

    def anticipate(self, property, min_support, k, func, file):
        start = tm.time()
        if func == 'ai':
            algorithm = AI()
            result = algorithm.run_algorithm(property, min_support, file)
        elif func == 'aii':
            algorithm = AII()
            result = algorithm.run_algorithm(property, min_support, file)
        elif func == 'bi' or func == 'bii':
            if func == 'bi':
                algorithm = BI()
            else:
                algorithm = BII()
            list = algorithm.run_algorithm(property, min_support, file)
            result = []
            for i in list:
                temp = []
                for j in i.sequence:
                    for tp in j:
                        temp.append(tp)
                result.append((temp,i.freq))
        else:
            raise ValueError('no such func')
        print(result)
        correct = 0
        total = 0
        predict = []
        for vipno in self._index:
            try:
                vipno_data = anticipate.get_train_by_property(vipno, property)
            except:
                continue
            anticipate_result = defaultdict(lambda: 0)
            for itemset, time in result:
                for i in vipno_data:
                    if i in itemset:
                        for j in itemset:
                            if time > anticipate_result[j]:
                                anticipate_result[j] = time
                        break
            anticipate_result = sorted(anticipate_result, key=lambda k: anticipate_result[k], reverse=True)
            if len(anticipate_result) > k:
                anticipate_result = anticipate_result[:k-1]
            real_result = anticipate.get_test_by_property(vipno, 'pluno')
            tag = 0
            for i in anticipate_result:
                if i in real_result:
                    tag += 1
            if len(anticipate_result) == 0:
                print('testing vipno: ' + str(vipno) + '  the accuracy is: 0%')
                predict.append(0)
            else:
                print('testing vipno: ' + str(vipno) + '  the accuracy is: ' + str(round(100 * tag / len(anticipate_result),1)) + '%')
                predict.append(round(100 * tag / len(anticipate_result),1))
            correct += tag
            total += len(anticipate_result)

        self._accuracy = round(correct/total * 100,2)
        self._correct = correct
        self._total = total
        end = tm.time()
        self._time = round(end - start, 4)
        return predict

    def get_accuracy(self):
        return self._accuracy

    def get_total(self):
        return self._total

    def get_correct(self):
        return self._correct

    def get_time(self):
        return self._time

    def show_information(self):
        print('=========================================================')
        print('the total anticipated data is: ' + str(self._total))
        print('the correct anticipated data number is: ' + str(self._correct))
        print('the accuracy is: ' + str(self._accuracy) + '%')
        print('using time: ' + str(self._time) + 's')

    def get_index(self):
        return self._index

if __name__ == '__main__':
    anticipate = Anticipate()
    anticipate.get_data(2)

    sup_a = [2, 4, 8, 16, 32, 64]
    result = []
    for i in sup_a:
        anticipate.anticipate('pluno', 16, 10, 'ai', 2)
        result.append(anticipate.get_time())
    # print(result)
    # print(len(anticipate.get_index()))
    # index = []
    # for i in range(len(anticipate.get_index())):
    #     index.append(i)
    # choose fp growth algorithm whose itemset is classified by uid
    # anticipate.anticipate('pluno', 8, 6, 'bii',1)
    # sup_a = [2,4,8,16,32,64]
    # sup_b = [2,4,6,8,10]
    # func = ['ai','aii','bi','bii']
    # itemset_number = [2,4,6,8,10,12,14,16,18,20]
    # file = [1,2]

    # result = []
    # result_ai = []
    # result_aii = []
    # for k in itemset_number:
    #     anticipate.anticipate('pluno', 16,k,'aii',2)
    #     result.append(anticipate.get_accuracy())

    # for s in sup_a:
    #     anticipate.anticipate('pluno',s,10,'ai',2)
    #     result_ai.append(anticipate.get_accuracy())
    # for s in sup_a:
    #     anticipate.anticipate('pluno', s, 10, 'aii', 2)
    #     result_aii.append(anticipate.get_accuracy())
    # total_trade = []
    # correct_trade = []
    # total_new = []
    # correct_new = []
    # ''' test for minsup and property '''
    # for k in itemset_number:
    #     anticipate.anticipate('pluno', 16, k, 'aii', 1)
    #     total_trade.append(anticipate.get_total() / 20)
    #     correct_trade.append(anticipate.get_correct()/20)
    # anticipate.get_data(2)
    # for k in itemset_number:
    #     anticipate.anticipate('pluno', 16, k, 'aii', 2)
    #     total_new.append(anticipate.get_total()/20)
    #     correct_new.append(anticipate.get_correct()/20)
    anticipate.show_information()

    output_file('efficiency.html')
    p = figure(
        tools='pan,box_zoom,reset,save',
        y_range=[0, 5], title='predict',
        x_axis_label='min support', y_axis_label='seconds(s)',
        x_range=[0,70]
    )
    p.line(sup_a, result, legend='time consuming', line_color='orange', line_width=2)
    # p.line(sup_a, result_aii, legend='aii accuracy', line_color='orange', line_width=2)
    # p.line(itemset_number, correct_trade, legend='correct trade.csv', line_color = 'orange', line_width = 2, line_dash = '4 4')
    # p.line(itemset_number, total_new, legend='total trade_new.csv', line_color='#108749', line_width=2)
    # p.line(itemset_number, correct_new, legend='correct trade_new.csv', line_color='#6ddb00', line_width=2, line_dash='4 4')
    show(p)