import pandas as pd


def get_user(file):
    file = open('../ci/output/' + file)
    result = []
    for line in file:
        if line.split(',')[1].split('\n')[0] == 'Yes':
            result.append(line.split(',')[0])
    result = pd.DataFrame(result, columns=['u'])
    return result


def get_user_brand(file):
    file = open('../cii/output/' + file)
    result = []
    for line in file:
        temp = []
        if line.split(',')[2].split('\n')[0] == 'Yes':
            temp.append(line.split(',')[0])
            temp.append(line.split(',')[1])
            # temp.append(line.split(',')[2].split('\n')[0])
            result.append(temp)
    result = pd.DataFrame(result, columns=['u', 'b'])
    return result


def get_user_category(file):
    file = open('../ciii/output/' + file)
    result = []
    for line in file:
        temp = []
        if line.split(',')[2].split('\n')[0] == 'Yes':
            temp.append(line.split(',')[0])
            temp.append(line.split(',')[1])
            result.append(temp)
    result = pd.DataFrame(result, columns=['u', 'c'])
    return result


def get_user_amount(file):
    file = open('../civ/output/' + file)
    result = []
    for line in file:
        temp = []
        temp.append(line.split(',')[0])
        temp.append(line.split(',')[1].split('\n')[0])
        result.append(temp)
    result = pd.DataFrame(result, columns=['u', 'amt'])
    return result


if __name__ == '__main__':
    name_list = ['AdaBoost', 'Bagging', 'DecisionTree', 'GradientBoost',
                 'KNeighbors', 'RandomForest']

    raw_data = pd.read_csv('../../a/data/raw_data.csv').loc[:, ['bndno', 'pluno','dptno', 'amt', 'qty']]
    new_data = []
    for index, raw in raw_data.groupby(['bndno', 'pluno', 'dptno','amt', 'qty']):
        new_data.append(list(index))

    new_data = pd.DataFrame(new_data, columns=['b', 'i', 'c','amt_u', 'qty'])
    new_data = new_data[new_data['b'] == 15012.0]
    new_data = new_data[new_data['c'] == 15110]
    result_list = []
    for name in name_list:
        user = get_user('1552635_2ci_%s.txt' % name)
        user_brand = get_user_brand('1552635_2cii_%s.txt' % name)
        user_category = get_user_category('1552635_2ciii_%s.txt' % name)
        user_amt = get_user_amount('1552635_2civ_%s.txt' % name)
        data = pd.merge(user_brand, user, how='left')
        data = pd.merge(data, user_amt, how='left')
        data = pd.merge(data, user_category, how='inner')
        # data = pd.merge(new_data, new_data, how='outer', on=['b', 'c'])
        for index, row in data.iterrows():
            u = row.loc['u']
            b = float(row.loc['b'])
            c = float(row.loc['c'])
            # 计算欸个月用户购买的单价
            amt = float(row.loc['amt']) / 3
            li = new_data[new_data['b'] == b]
            li = li[li['amt_u'] < amt + 20]
            li = li[li['amt_u'] > amt - 20]
            for i, d in li.iterrows():
                result_list.append([u, b, c, d.loc['i']])
    result_list = pd.DataFrame(result_list, columns=['u', 'b', 'c', 'i'])
    file = open('output/predict.txt', 'w')
    for index, temp_data in result_list.groupby('u'):
        file.write(index + '::')
        for i in list(temp_data.loc[:, 'i']):
            file.write(str(int(i)) + ':' + str(int(len(temp_data[temp_data['i'] == i]) / 6) + 1) + ',')
        file.write('\n')