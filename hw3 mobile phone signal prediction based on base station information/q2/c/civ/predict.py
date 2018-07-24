import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, \
    AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings('ignore')


def get_union_feature():
    data = pd.read_csv('../../a/predict/type1/count/data_u.csv')
    new_data = pd.read_csv('../../a/predict/type2/month_agg/count/data_u.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type2/brand_agg.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type2/category_agg.csv')
    data = pd.merge(data, new_data, how='left')
    new_data = pd.read_csv('../../a/predict/type2/item_agg.csv')
    data = pd.merge(data, new_data, how='left')
    data.to_csv('data/predict_data.csv', index=False)


if __name__ == '__main__':
    get_union_feature()
    data = pd.read_csv('data/temp_data.csv', dtype={'amt': np.float64})
    labels = data.loc[:, 'amt']
    data = data.drop('amt', axis=1)
    new_data = pd.read_csv('data/predict_data.csv')
    temp_data = pd.read_csv('../../a/data/raw_data.csv')
    temp = []
    for i, raw_data in temp_data.groupby('vipno'):
        temp.append(i)
    temp_data = pd.DataFrame(temp, columns=['u'])
    new_data = pd.merge(temp_data, new_data, how='left').fillna(0)
    # Gradient Boost
    clf = GradientBoostingRegressor(n_estimators=500, min_samples_split=2, learning_rate=0.01)
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2civ_GradientBoost.txt', 'w')
    for i in range(len(result)):
        file.write(str(new_data.loc[i, 'u']) + ',' + str(result[i]) + '\n')
    file.close()
    # Ada Boost
    clf = AdaBoostRegressor(n_estimators=20)
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2civ_AdaBoost.txt', 'w')
    for i in range(len(result)):
        file.write(str(new_data.loc[i, 'u']) +  ',' + str(result[i]) + '\n')
    file.close()
    # Random Forest
    clf = RandomForestRegressor(n_estimators=40, max_depth=20)
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2civ_RandomForest.txt', 'w')
    for i in range(len(result)):
        file.write(str(new_data.loc[i, 'u']) + ','  + str(result[i]) + '\n')
    file.close()
    # Bagging
    clf = BaggingRegressor(n_estimators=10)
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2civ_Bagging.txt', 'w')
    for i in range(len(result)):
        file.write(str(new_data.loc[i, 'u']) + ',' + str(result[i]) + '\n')
    file.close()
    clf = DecisionTreeRegressor()
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2civ_DecisionTree.txt', 'w')
    for i in range(len(result)):
        file.write(str(new_data.loc[i, 'u']) +  ',' + str(result[i]) + '\n')
    file.close()
    clf = KNeighborsRegressor()
    clf.fit(data, labels)
    result = clf.predict(new_data)
    file = open('output/1552635_2civ_KNeighbors.txt', 'w')
    for i in range(len(result)):
        file.write(str(new_data.loc[i, 'u'])  + ',' + str(result[i]) + '\n')
    file.close()
