import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, \
    AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def get_union_feature():
    x = 0.16
    x = int(x)
    data = pd.read_csv('../../a/data/type1/count/data_u.csv')
    # data = pd.read_csv('../../a/data/type1/diversity/data_u_b.csv')
    # data = pd.merge(data, new_data, how='left')
    # new_data = pd.read_csv('../../a/data/type1/diversity/data_u_c.csv')
    # data = pd.merge(data, new_data, how='left')
    # new_data = pd.read_csv('../../a/data/type1/diversity/data_u_i.csv')
    # data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/data/type2/month_agg/count/data_u.csv')
    data = pd.merge(data, new_data, how='left')
    # new_data = pd.read_csv('../../a/data/type2/month_agg/diversity/data_u_b.csv')
    # data = pd.merge(data, new_data, how='left')
    # new_data = pd.read_csv('../../a/data/type2/month_agg/diversity/data_u_c.csv')
    # data = pd.merge(data, new_data, how='left')
    # new_data = pd.read_csv('../../a/data/type2/month_agg/diversity/data_u_i.csv')
    # data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/data/type2/brand_agg.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/data/type2/category_agg.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/data/type2/item_agg.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/data/raw_data.csv')
    new_data = new_data[new_data['time'] == 5]
    temp = []
    for index, temp_data in new_data.groupby('vipno'):
        temp.append([index, round(sum(temp_data.loc[:, 'amt']), 2)])
    new_data = pd.DataFrame(temp, columns=['u', 'amt'])
    data = pd.merge(data, new_data, how='left').fillna(0)
    data.to_csv('data/temp_data.csv', index=False)


def get_show_data(pre_data):
    new_data = []
    for i in range(len(pre_data[0])):
        temp = []
        for j in range(10):
            if i < len(pre_data[j]):
                temp.append(pre_data[j][i])
        new_data.append(round(sum(temp) / len(temp), 2))
    pre = []
    new_data = sorted(new_data)
    for i in range(10):
        pre.append(new_data[int(len(new_data) * (i + 1) / 10 - 1)])
    return pre


if __name__ == '__main__':
    get_union_feature()
    data = pd.read_csv('data/temp_data.csv', dtype={'amt': np.float64})
    labels = data.loc[:, 'amt']
    data = data.drop('amt', axis=1)
    kf = KFold(n_splits=10, shuffle=True)
    mean_tpr = [i for i in range(7)]
    mean_fpr = [i for i in range(7)]
    for i in range(7):
        mean_tpr[i] = 0.0
        mean_fpr[i] = np.linspace(0, 1, 50)
    decision_tree_pre = []
    k_neighbor_pre = []
    random_forest_pre = []
    bagging_pre = []
    adaboost_pre = []
    gradient_boost_pre = []
    for train_index, test_index in kf.split(data):
        train_data = data.loc[list(train_index), :]
        train_label = labels.loc[list(train_index)]
        test_data = data.loc[list(test_index), :]
        test_label = labels.loc[list(test_index)]
        print('#####################################################################################')
        clf_decision_tree =DecisionTreeRegressor()
        clf_decision_tree.fit(train_data, train_label)
        pre = clf_decision_tree.predict(test_data)
        judge = []
        for i in range(len(pre)):
            judge.append(abs(pre[i] - test_label.iloc[i]))
        print('decision tree average error: %f' % (sum(judge) / len(judge)))
        decision_tree_pre.append(judge)
        clf = KNeighborsRegressor()
        clf.fit(train_data, train_label)
        pre = clf.predict(test_data)
        judge = []
        for i in range(len(pre)):
            judge.append(abs(pre[i] - test_label.iloc[i]))
        print('k neighbor regression average error: %f' % (sum(judge) / len(judge)))
        k_neighbor_pre.append(judge)
        clf = RandomForestRegressor()
        clf.fit(train_data, train_label)
        pre = clf.predict(test_data)
        judge = []
        for i in range(len(pre)):
            judge.append(abs(pre[i] - test_label.iloc[i]))
        print('random forest regression average error: %f' % (sum(judge) / len(judge)))
        random_forest_pre.append(judge)
        clf = BaggingRegressor()
        clf.fit(train_data, train_label)
        pre = clf.predict(test_data)
        judge = []
        for i in range(len(pre)):
            judge.append(abs(pre[i] - test_label.iloc[i]))
        print('bagging regression average error: %f' % (sum(judge) / len(judge)))
        bagging_pre.append(judge)
        rg = AdaBoostRegressor()
        rg.fit(train_data, train_label)
        pre = rg.predict(test_data)
        judge = []
        for i in range(len(pre)):
            judge.append(abs(pre[i] - test_label.iloc[i]))
        print('adaboost regression average error: %f' % (sum(judge) / len(judge)))
        adaboost_pre.append(judge)
        rg = GradientBoostingRegressor()
        rg.fit(train_data, train_label)
        pre = rg.predict(test_data)
        judge = []
        for i in range(len(pre)):
            judge.append(abs(pre[i] - test_label.iloc[i]))
        print('gradient boost regression average error: %f' % (sum(judge) / len(judge)))
        gradient_boost_pre.append(judge)
    show_data = []
    y_line = np.linspace(0, 300, 50)
    x_line = []
    for i in range(50):
        x_line.append('50%')
    show_data.append(get_show_data(decision_tree_pre))
    show_data.append(get_show_data(k_neighbor_pre))
    show_data.append(get_show_data(random_forest_pre))
    show_data.append(get_show_data(bagging_pre))
    show_data.append(get_show_data(adaboost_pre))
    show_data.append(get_show_data(gradient_boost_pre))
    index = [str(i+1)+'0%' for i in range(10)]
    plt.plot(index, show_data[0], label='decision tree')
    plt.plot(index, show_data[1], label='k neighbor')
    plt.plot(index, show_data[2], label='random forest')
    plt.plot(index, show_data[3], label='bagging forest')
    plt.plot(index, show_data[4], label='adaboost')
    plt.plot(index, show_data[5], label='gradient boost')
    plt.plot(x_line, y_line, 'r--')
    plt.ylim([0, 300])
    plt.xlabel('the ratio of the most smallest error')
    plt.ylabel('error($)')
    plt.title('CDF Curve')
    plt.legend(loc='lower right')
    plt.show()
