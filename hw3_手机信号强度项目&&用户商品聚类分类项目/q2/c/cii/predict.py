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
    data = pd.read_csv('../../a/predict/type1/count/data_u_b.csv')
    new_data = pd.read_csv('../../a/predict/type1/count/data_u.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type1/count/data_b.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type1/diversity/data_u_b.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type1/penetration/data_b.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type2/month_agg/count/data_b.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type2/month_agg/count/data_u_b.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type2/month_agg/count/data_u.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type2/month_agg/diversity/data_u_b.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type2/month_agg/penetration/data_b.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type2/user_agg/user_agg_b.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type2/brand_agg.csv')
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
    for i, raw_data in temp_data.groupby(['vipno', 'bndno']):
        temp.append(list(i))
    temp_data = pd.DataFrame(temp, columns=['u', 'b'])
    new_data = pd.merge(temp_data, new_data, how='left').fillna(0)
    # Gradient Boost
    clf = GradientBoostingClassifier(n_estimators=500, min_samples_split=2, learning_rate=0.01)
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2cii_GradientBoost.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) + ',' + str(new_data.loc[i, 'b']) + ',' + tag + '\n')
    file.close()
    # Ada Boost
    clf = AdaBoostClassifier(n_estimators=20)
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2cii_AdaBoost.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) + ',' + str(new_data.loc[i, 'b']) + ',' + tag + '\n')
    file.close()
    # Random Forest
    clf = RandomForestClassifier(n_estimators=40, max_depth=20)
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2cii_RandomForest.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) + ',' + str(new_data.loc[i, 'b']) + ',' + tag + '\n')
    file.close()
    # Bagging
    clf = BaggingClassifier(n_estimators=10)
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2cii_Bagging.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) + ',' + str(new_data.loc[i, 'b']) + ',' + tag + '\n')
    file.close()
    clf = DecisionTreeClassifier()
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2cii_DecisionTree.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) + ',' + str(new_data.loc[i, 'b']) + ',' + tag + '\n')
    file.close()
    clf = GaussianNB()
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2cii_GaussianNB.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) + ',' + str(new_data.loc[i, 'b']) + ',' + tag + '\n')
    file.close()
    clf = KNeighborsClassifier()
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2cii_KNeighbors.txt', 'w')
    for i in range(len(result)):
        tag = ''
        if result[i] == 1:
            tag = 'Yes'
        else:
            tag = 'No'
        file.write(str(new_data.loc[i, 'u']) + ',' + str(new_data.loc[i, 'b']) + ',' + tag + '\n')
    file.close()
