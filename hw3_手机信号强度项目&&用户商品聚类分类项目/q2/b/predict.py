import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, \
    AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')


def get_union_feature():
    new_data = pd.read_csv('../a/data/raw_data.csv', dtype={'u': np.str})
    new_data = new_data[new_data['time'] > 5]
    data = []
    for index, temp_data in new_data.groupby(['vipno', 'pluno']):
        data.append(list(index))
    new_data = pd.DataFrame(data, columns=['u','i'])
    data_type1_count_data_u = pd.read_csv('../a/predict/type1/count/data_u.csv')
    new_data = pd.merge(new_data, data_type1_count_data_u, how='left')
    data_type1_count_data_u_i = pd.read_csv('../a/predict/type1/count/data_u_i.csv')
    new_data = pd.merge(new_data, data_type1_count_data_u_i, how='left')
    data_type1_count_data_i = pd.read_csv('../a/predict/type1/count/data_i.csv')
    new_data = pd.merge(new_data, data_type1_count_data_i, how='left')
    data_type1_diversity_data_ui = pd.read_csv('../a/predict/type1/diversity/data_u_i.csv')
    new_data = pd.merge(new_data, data_type1_diversity_data_ui, how='left')
    data_type1_peneration_data_i = pd.read_csv('../a/predict/type1/penetration/data_i.csv')
    new_data = pd.merge(new_data, data_type1_peneration_data_i, how='left')
    new_data = new_data.fillna(0)
    data_type2_month_agg_count_data_u_i = pd.read_csv('../a/predict/type2/month_agg/count/data_u_i.csv')
    new_data = pd.merge(new_data, data_type2_month_agg_count_data_u_i, how='left')
    data_type2_month_agg_count_data_u = pd.read_csv('../a/predict/type2/month_agg/count/data_u.csv')
    new_data = pd.merge(new_data, data_type2_month_agg_count_data_u, how='left', on='u')
    data_type2_month_agg_count_data_i = pd.read_csv('../a/predict/type2/month_agg/count/data_i.csv')
    new_data = pd.merge(new_data, data_type2_month_agg_count_data_i, how='left', on='i')
    data_type2_month_agg_diversity_data_u_i = pd.read_csv('../a/predict/type2/month_agg/diversity/data_u_i.csv')
    new_data = pd.merge(new_data, data_type2_month_agg_diversity_data_u_i, how='left')
    data_type2_month_agg_penetration_data_i = pd.read_csv('../a/predict/type2/month_agg/penetration/data_i.csv')
    new_data = pd.merge(new_data, data_type2_month_agg_penetration_data_i, how='left').fillna(0)
    data_type2_user_agg_i = pd.read_csv('../a/predict/type2/user_agg/user_agg_i.csv')
    new_data = pd.merge(new_data, data_type2_user_agg_i, how='left').fillna(0)
    data_item_agg = pd.read_csv('../a/predict/type2/item_agg.csv')
    new_data = pd.merge(new_data, data_item_agg, how='left')
    new_data.to_csv('predict_data.csv', index=False)


if __name__ == '__main__':
    data = pd.read_csv('data/dataset.csv', dtype={'tag': np.int})
    labels = data.loc[:, 'tag']
    data = data.drop('tag', axis=1)
    new_data = pd.read_csv('data/predict_data.csv')
    temp_data = pd.read_csv('../a/data/raw_data.csv')
    temp = []
    for i, raw_data in temp_data.groupby(['vipno', 'pluno']):
        temp.append(list(i))
    temp_data = pd.DataFrame(temp, columns=['u', 'i'])
    new_data = pd.merge(temp_data, new_data, how='left').fillna(0)
    # Gradient Boost
    clf = GradientBoostingClassifier(n_estimators=500, min_samples_split=2, learning_rate=0.01)
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2b_GradientBoost.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) + ',' + str(new_data.loc[i, 'i']) + ',' + tag + '\n')
    file.close()
    # Ada Boost
    clf = AdaBoostClassifier(n_estimators=20)
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2b_AdaBoost.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) + ',' + str(new_data.loc[i, 'i']) + ',' + tag + '\n')
    file.close()
    # Random Forest
    clf = RandomForestClassifier(n_estimators=40, max_depth=20)
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2b_RandomForest.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) + ',' + str(new_data.loc[i, 'i']) + ',' + tag + '\n')
    file.close()
    # Bagging
    clf = BaggingClassifier(n_estimators=10)
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2b_Bagging.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) + ',' + str(new_data.loc[i, 'i']) + ',' + tag + '\n')
    file.close()
    clf = DecisionTreeClassifier()
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2b_DecisionTree.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) + ',' + str(new_data.loc[i, 'i']) + ',' + tag + '\n')
    file.close()
    clf = GaussianNB()
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2b_GaussianNB.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) + ',' + str(new_data.loc[i, 'i']) + ',' + tag + '\n')
    file.close()
    clf = KNeighborsClassifier()
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2b_KNeighbors.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) + ',' + str(new_data.loc[i, 'i']) + ',' + tag + '\n')
    file.close()
