import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, \
    AdaBoostClassifier, GradientBoostingClassifier
from scipy import interp
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def get_union_feature():
    new_data = pd.read_csv('../a/data/raw_data.csv', dtype={'u': np.str})
    new_data = new_data[new_data['time'] < 5]
    data = []
    for index, temp_data in new_data.groupby(['vipno', 'pluno']):
        data.append(list(index))
    new_data = pd.DataFrame(data, columns=['u','i'])
    data_type1_count_data_u = pd.read_csv('../a/data/type1/count/data_u.csv')
    new_data = pd.merge(new_data, data_type1_count_data_u, how='left')
    data_type1_count_data_u_i = pd.read_csv('../a/data/type1/count/data_u_i.csv')
    new_data = pd.merge(new_data, data_type1_count_data_u_i, how='left')
    data_type1_count_data_i = pd.read_csv('../a/data/type1/count/data_i.csv')
    new_data = pd.merge(new_data, data_type1_count_data_i, how='left')
    data_type1_diversity_data_ui = pd.read_csv('../a/data/type1/diversity/data_u_i.csv')
    new_data = pd.merge(new_data, data_type1_diversity_data_ui, how='left')
    data_type1_peneration_data_i = pd.read_csv('../a/data/type1/penetration/data_i.csv')
    new_data = pd.merge(new_data, data_type1_peneration_data_i, how='left')
    new_data = new_data.fillna(0)
    data_type2_month_agg_count_data_u_i = pd.read_csv('../a/data/type2/month_agg/count/data_u_i.csv')
    new_data = pd.merge(new_data, data_type2_month_agg_count_data_u_i, how='left')
    data_type2_month_agg_count_data_u = pd.read_csv('../a/data/type2/month_agg/count/data_u.csv')
    new_data = pd.merge(new_data, data_type2_month_agg_count_data_u, how='left', on='u')
    data_type2_month_agg_count_data_i = pd.read_csv('../a/data/type2/month_agg/count/data_i.csv')
    new_data = pd.merge(new_data, data_type2_month_agg_count_data_i, how='left', on='i')
    data_type2_month_agg_diversity_data_u_i = pd.read_csv('../a/data/type2/month_agg/diversity/data_u_i.csv')
    new_data = pd.merge(new_data, data_type2_month_agg_diversity_data_u_i, how='left')
    data_type2_month_agg_penetration_data_i = pd.read_csv('../a/data/type2/month_agg/penetration/data_i.csv')
    new_data = pd.merge(new_data, data_type2_month_agg_penetration_data_i, how='left').fillna(0)
    data_type2_user_agg_i = pd.read_csv('../a/data/type2/user_agg/user_agg_i.csv')
    new_data = pd.merge(new_data, data_type2_user_agg_i, how='left').fillna(0)
    data_item_agg = pd.read_csv('../a/data/type2/item_agg.csv')
    new_data = pd.merge(new_data, data_item_agg, how='left')
    new_data.to_csv('data/temp_data.csv', index=False)


def create_dataset():
    data_5 = pd.read_csv('../a/data/raw_data.csv', dtype={'u': np.str})
    data_5 = data_5[data_5['time'] == 5]
    data = []
    for index, temp_data in data_5.groupby(['vipno', 'pluno']):
        data.append(list(index))
    data_5 = pd.DataFrame(data, columns=['u', 'i'])
    data_5['tag'] = 1
    data_5.columns = ['u', 'i', 'tag']
    data = pd.read_csv('data/temp_data.csv')
    data = pd.merge(data, data_5, how='left', on=['u', 'i']).fillna(0)
    data.to_csv('data/dataset.csv', index=False)


if __name__ == '__main__':
    # 先组合提取出来的特征
    # get_union_feature()
    # 融合第五个月的数据进行label标注
    # create_dataset()
    data = pd.read_csv('data/dataset.csv', dtype={'tag': np.int})
    print('true sample: %d' % len(data[data['tag'] == 1]))
    print('false sample: %d' % len(data[data['tag'] == 0]))
    labels = data.loc[:, 'tag']
    data = data.drop('tag', axis=1)
    kf = KFold(n_splits=10, shuffle=True)
    mean_tpr = [i for i in range(7)]
    mean_fpr = [i for i in range(7)]
    for i in range(7):
        mean_tpr[i] = 0.0
        mean_fpr[i] = np.linspace(0,1,50)
    for train_index, test_index in kf.split(data):
        train_data = data.loc[list(train_index), :]
        train_label = labels.loc[list(train_index)]
        test_data = data.loc[list(test_index), :]
        test_label = labels.loc[list(test_index)]
        clf_decision_tree = DecisionTreeClassifier(criterion='entropy')
        clf_decision_tree.fit(train_data, train_label)
        prob = clf_decision_tree.predict_proba(test_data)
        result = clf_decision_tree.predict(test_data)
        fpr, tpr, threshold = roc_curve(test_label, prob[:, 1])
        mean_tpr[0] += interp(mean_fpr[0], fpr, tpr)  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
        mean_tpr[0][0] = 0.0  # 初始处为0
        precision = precision_score(test_label, result)
        recall = recall_score(test_label, result)
        f1 = f1_score(test_label, result)
        print('#####################################################################################')
        tag = 0
        for i in range(len(result)):
            if result[i] == test_label.iloc[i]:
                tag += 1
        print('precision: %.3f' % (tag / len(result)))
        print('decision tree:\t\t\tprecision: %f\trecall: %f\tf1 score: %f' %
              (round(precision, 2), round(recall,2), round(f1,2)))
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(train_data, train_label)
        prob = clf.predict_proba(test_data)
        result = clf.predict(test_data)
        fpr, tpr, threshold = roc_curve(test_label, prob[:, 1])
        mean_tpr[1] += interp(mean_fpr[1], fpr, tpr)
        mean_tpr[1][0] = 0.0  # 初始处为0
        precision = precision_score(test_label, result)
        recall = recall_score(test_label, result)
        f1 = f1_score(test_label, result)
        print('k neighbor:\t\t\t\tprecision: %f\trecall: %f\tf1 score: %f' %
              (round(precision, 2), round(recall, 2), round(f1, 2)))
        clf = GaussianNB()
        clf.fit(train_data, train_label)
        prob = clf.predict_proba(test_data)
        result = clf.predict(test_data)
        fpr, tpr, threshold = roc_curve(test_label, prob[:, 1])
        mean_tpr[2] += interp(mean_fpr[2], fpr, tpr)
        mean_tpr[2][0] = 0.0  # 初始处为0
        precision = precision_score(test_label, result)
        recall = recall_score(test_label, result)
        f1 = f1_score(test_label, result)
        print('Gaussian:\t\t\t\tprecision: %f\trecall: %f\tf1 score: %f' %
              (round(precision, 2), round(recall, 2), round(f1, 2)))
        clf = RandomForestClassifier(n_estimators=40, max_depth=20)
        clf.fit(train_data, train_label)
        prob = clf.predict_proba(test_data)
        result = clf.predict(test_data)
        fpr, tpr, threshold = roc_curve(test_label, prob[:, 1])
        mean_tpr[3] += interp(mean_fpr[3], fpr, tpr)
        mean_tpr[3][0] = 0.0  # 初始处为0
        precision = precision_score(test_label, result)
        recall = recall_score(test_label, result)
        f1 = f1_score(test_label, result)
        print('Random Forest:\t\t\tprecision: %f\trecall: %f\tf1 score: %f' %
              (round(precision, 2), round(recall, 2), round(f1, 2)))
        clf = BaggingClassifier(n_estimators=10)
        clf.fit(train_data, train_label)
        prob = clf.predict_proba(test_data)
        result = clf.predict(test_data)
        fpr, tpr, threshold = roc_curve(test_label, prob[:, 1])
        mean_tpr[4] += interp(mean_fpr[4], fpr, tpr)
        mean_tpr[4][0] = 0.0  # 初始处为0
        precision = precision_score(test_label, result)
        recall = recall_score(test_label, result)
        f1 = f1_score(test_label, result)
        print('Bagging Classifier:\t\tprecision: %f\trecall: %f\tf1 score: %f' %
              (round(precision, 2), round(recall, 2), round(f1, 2)))
        clf = AdaBoostClassifier(n_estimators=20)
        clf.fit(train_data, train_label)
        prob = clf.predict_proba(test_data)
        result = clf.predict(test_data)
        fpr, tpr, threshold = roc_curve(test_label, prob[:, 1])
        mean_tpr[5] += interp(mean_fpr[5], fpr, tpr)
        mean_tpr[5][0] = 0.0  # 初始处为0
        precision = precision_score(test_label, result)
        recall = recall_score(test_label, result)
        f1 = f1_score(test_label, result)
        print('Adaboost Classifier:\tprecision: %f\trecall: %f\tf1 score: %f' %
              (round(precision, 2), round(recall, 2), round(f1, 2)))
        clf = GradientBoostingClassifier(n_estimators=500, min_samples_split=2, learning_rate=0.01)
        clf.fit(train_data, train_label)
        prob = clf.predict_proba(test_data)
        result = clf.predict(test_data)
        fpr, tpr, threshold = roc_curve(test_label, prob[:, 1])
        mean_tpr[6] += interp(mean_fpr[6], fpr, tpr)
        mean_tpr[6][0] = 0.0  # 初始处为0
        precision = precision_score(test_label, result)
        recall = recall_score(test_label, result)
        f1 = f1_score(test_label, result)
        print('GradientBoost Classifier:\tprecision: %f\trecall: %f\tf1 score: %f' %
              (round(precision, 2), round(recall, 2), round(f1, 2)))

    print('#####################################################################################')
    mean_auc = [i for i in range(7)]
    for i in range(7):
        mean_tpr[i] /= 10  # 在mean_fpr100个点，每个点处插值插值多次取平均
        list(mean_tpr[i])[-1] = 1.0  # 坐标最后一个点为（1,1)
        mean_auc[i] = auc(mean_fpr[i], mean_tpr[i])
    # print('decision tree auc value: %f' % (round(mean_auc, 2)))
    split_line = np.linspace(0, 1, 50)
    plt.plot(split_line, split_line, 'r--')
    plt.plot(mean_fpr[0], mean_tpr[0], label='Decision Tree AUC: %f' % (round(mean_auc[0], 4)))
    plt.plot(mean_fpr[1], mean_tpr[1], label='K Neighbor AUC: %f' % (round(mean_auc[1], 4)))
    plt.plot(mean_fpr[2], mean_tpr[2], label='Gaussian AUC: %f' % (round(mean_auc[2], 4)))
    plt.plot(mean_fpr[3], mean_tpr[3], label='Random Forest AUC: %f' % (round(mean_auc[3], 4)))
    plt.plot(mean_fpr[4], mean_tpr[4], label='Bagging Classifier AUC: %f' % (round(mean_auc[4], 4)))
    plt.plot(mean_fpr[5], mean_tpr[5], label='Adaboost Classifier AUC: %f' % (round(mean_auc[5], 4)))
    plt.plot(mean_fpr[6], mean_tpr[6], label='GradientBoost Classifier AUC: %f' % (round(mean_auc[6], 4)))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('The ROC diagram of user_item_features')
    plt.legend(loc='lower right')
    plt.show()
