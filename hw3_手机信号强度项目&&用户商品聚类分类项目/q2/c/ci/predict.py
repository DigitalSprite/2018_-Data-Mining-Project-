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
    data = pd.read_csv('../../a/predict/type1/count/data_u.csv')
    new_data = pd.read_csv('../../a/predict/type1/diversity/data_u_b.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type1/diversity/data_u_c.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type1/diversity/data_u_i.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type2/month_agg/count/data_u.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type2/month_agg/diversity/data_u_b.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type2/month_agg/diversity/data_u_c.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type2/month_agg/diversity/data_u_i.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type2/brand_agg.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type2/category_agg.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type2/item_agg.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/data/raw_data.csv')
    new_data = new_data[new_data['time'] == 5].loc[:, 'vipno']
    new_data = pd.DataFrame(list(set(list(new_data))), columns=['u'])
    data = pd.merge(data, new_data, how='left')
    data.to_csv('data/predict_data.csv', index=False)


if __name__ == '__main__':
    get_union_feature()
    data = pd.read_csv('data/temp_data.csv', dtype={'tag': np.int})
    labels = data.loc[:, 'tag']
    data = data.drop('tag', axis=1)
    new_data = pd.read_csv('data/predict_data.csv')
    temp_data = pd.read_csv('../../a/data/raw_data.csv')
    temp = []
    for i, raw_data in temp_data.groupby('vipno'):
        temp.append(i)
    temp_data = pd.DataFrame(temp, columns=['u'])
    new_data = pd.merge(temp_data, new_data, how='left').fillna(0)
    # Gradient Boost
    clf = GradientBoostingClassifier(n_estimators=500, min_samples_split=2, learning_rate=0.01)
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2ci_GradientBoost.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) + ',' + tag + '\n')
    file.close()
    # Ada Boost
    clf = AdaBoostClassifier(n_estimators=20)
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2ci_AdaBoost.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) +  ',' + tag + '\n')
    file.close()
    # Random Forest
    clf = RandomForestClassifier(n_estimators=40, max_depth=20)
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2ci_RandomForest.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) + ','  + tag + '\n')
    file.close()
    # Bagging
    clf = BaggingClassifier(n_estimators=10)
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2ci_Bagging.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) + ',' + tag + '\n')
    file.close()
    clf = DecisionTreeClassifier()
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2ci_DecisionTree.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) +  ',' + tag + '\n')
    file.close()
    clf = GaussianNB()
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2ci_GaussianNB.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) + ','  + tag + '\n')
    file.close()
    clf = KNeighborsClassifier()
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2ci_KNeighbors.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u'])  + ',' + tag + '\n')
    file.close()
