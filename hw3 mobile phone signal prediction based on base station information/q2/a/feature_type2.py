import pandas as pd
import numpy as np


def get_month_agg_count():
    print('\n##################################################')
    print('create data_agg_count_u grouped by vipno...')
    data = pd.read_csv('data/type1/count/data_u.csv', dtype={'u': np.str})
    count = []
    amount = []
    for index, row in data.iterrows():
        # 计算 count
        temp_count = [row.loc['u']]
        temp_count.append(max(list(row.loc[['u_count_2', 'u_count_3', 'u_count_4']])))
        temp_count.append(sorted(list(row.loc[['u_count_2', 'u_count_3', 'u_count_4']]))[1])
        temp_count.append(sum(list(row.loc[['u_count_2', 'u_count_3', 'u_count_4']])) / 3)
        temp_count.append(np.std(list(row.loc[['u_count_2', 'u_count_3', 'u_count_4']])))
        count.append(temp_count)
        # 计算amount
        temp_amount = []
        temp_amount.append(max(list(row.loc[['u_amount_2', 'u_amount_3', 'u_amount_4']])))
        temp_amount.append(sorted(list(row.loc[['u_amount_2', 'u_amount_3', 'u_amount_4']]))[1])
        temp_amount.append(sum(list(row.loc[['u_amount_2', 'u_amount_3', 'u_amount_4']])) / 3)
        temp_amount.append(np.std(list(row.loc[['u_amount_2', 'u_amount_3', 'u_amount_4']])))
        amount.append(temp_amount)
    new_data = [[a, a1, a2, a3, a4, a5, a6, a7, a8] for ([a, a1, a2, a3, a4], [a5, a6, a7, a8]) in zip(count, amount)]
    data_agg_count_u = pd.DataFrame(new_data, columns=['u', 'u_agg_count_max', 'u_agg_count_median',
                                                       'u_agg_count_mean', 'u_agg_count_std', 'u_agg_amount_max',
                                                       'u_agg_amount_median','u_agg_amount_mean', 'u_agg_amount_std'])
    data_agg_count_u.to_csv('data/type2/month_agg/count/data_u.csv', index=False)
    print('create data_agg_count_u grouped by vipno successfully')
    print('\n##################################################')
    print('create data_agg_count_u grouped by bndno...')
    data = pd.read_csv('data/type1/count/data_b.csv')
    count = []
    amount = []
    for index, row in data.iterrows():
        # 计算 count
        temp_count = [row.loc['b']]
        temp_count_data = list(row.loc[['b_count_2', 'b_count_3', 'b_count_4']])
        temp_count.append(max(temp_count_data))
        temp_count.append(sorted(temp_count_data)[1])
        temp_count.append(sum(temp_count_data) / 3)
        temp_count.append(np.std(temp_count_data))
        count.append(temp_count)
        # 计算amount
        temp_amount = []
        temp_amount_data = list(row.loc[['b_amount_2', 'b_amount_3', 'b_amount_4']])
        temp_amount.append(max(temp_amount_data))
        temp_amount.append(sorted(temp_amount_data)[1])
        temp_amount.append(sum(temp_amount_data) / 3)
        temp_amount.append(np.std(temp_amount_data))
        amount.append(temp_amount)
    new_data = [[a, a1, a2, a3, a4, a5, a6, a7, a8] for ([a, a1, a2, a3, a4], [a5, a6, a7, a8]) in zip(count, amount)]
    data_agg_count_b = pd.DataFrame(new_data, columns=['b', 'b_agg_count_max', 'b_agg_count_median',
                                                       'b_agg_count_mean', 'b_agg_count_std', 'b_agg_amount_max',
                                                       'b_agg_amount_median', 'b_agg_amount_mean', 'b_agg_amount_std'])
    data_agg_count_b.to_csv('data/type2/month_agg/count/data_b.csv', index=False)
    print('create data_agg_count_b grouped by bndno successfully')
    print('\n##################################################')
    print('create data_agg_count_c grouped by dptno...')
    data = pd.read_csv('data/type1/count/data_c.csv')
    count = []
    amount = []
    for index, row in data.iterrows():
        # 计算 count
        temp_count = [row.loc['c']]
        temp_count_data = list(row.loc[['c_count_2', 'c_count_3', 'c_count_4']])
        temp_count.append(max(temp_count_data))
        temp_count.append(sorted(temp_count_data)[1])
        temp_count.append(sum(temp_count_data) / 3)
        temp_count.append(np.std(temp_count_data))
        count.append(temp_count)
        # 计算amount
        temp_amount = []
        temp_amount_data = list(row.loc[['c_amount_2', 'c_amount_3', 'c_amount_4']])
        temp_amount.append(max(temp_amount_data))
        temp_amount.append(sorted(temp_amount_data)[1])
        temp_amount.append(sum(temp_amount_data) / 3)
        temp_amount.append(np.std(temp_amount_data))
        amount.append(temp_amount)
    new_data = [[a, a1, a2, a3, a4, a5, a6, a7, a8] for ([a, a1, a2, a3, a4], [a5, a6, a7, a8]) in zip(count, amount)]
    data_agg_count_c = pd.DataFrame(new_data, columns=['c', 'c_agg_count_max', 'c_agg_count_median',
                                                       'c_agg_count_mean', 'c_agg_count_std', 'c_agg_amount_max',
                                                       'c_agg_amount_median', 'c_agg_amount_mean', 'c_agg_amount_std'])
    data_agg_count_c.to_csv('data/type2/month_agg/count/data_c.csv', index=False)
    print('create data_agg_count_c grouped by dptno successfully')
    print('\n##################################################')
    print('create data_agg_count_i grouped by pluno...')
    data = pd.read_csv('data/type1/count/data_i.csv')
    count = []
    amount = []
    for index, row in data.iterrows():
        # 计算 count
        temp_count = [row.loc['i']]
        temp_count_data = list(row.loc[['i_count_2', 'i_count_3', 'i_count_4']])
        temp_count.append(max(temp_count_data))
        temp_count.append(sorted(temp_count_data)[1])
        temp_count.append(sum(temp_count_data) / 3)
        temp_count.append(np.std(temp_count_data))
        count.append(temp_count)
        # 计算amount
        temp_amount = []
        temp_amount_data = list(row.loc[['i_amount_2', 'i_amount_3', 'i_amount_4']])
        temp_amount.append(max(temp_amount_data))
        temp_amount.append(sorted(temp_amount_data)[1])
        temp_amount.append(sum(temp_amount_data) / 3)
        temp_amount.append(np.std(temp_amount_data))
        amount.append(temp_amount)
    new_data = [[a, a1, a2, a3, a4, a5, a6, a7, a8] for ([a, a1, a2, a3, a4], [a5, a6, a7, a8]) in zip(count, amount)]
    data_agg_count_i = pd.DataFrame(new_data, columns=['i', 'i_agg_count_max', 'i_agg_count_median',
                                                       'i_agg_count_mean', 'i_agg_count_std', 'i_agg_amount_max',
                                                       'i_agg_amount_median', 'i_agg_amount_mean', 'i_agg_amount_std'])
    data_agg_count_i.to_csv('data/type2/month_agg/count/data_i.csv', index=False)
    print('create data_agg_count_i grouped by pluno successfully')
    print('\n##################################################')
    print('create data_agg_count_u_b grouped by vipno and bndno...')
    data = pd.read_csv('data/type1/count/data_u_b.csv', dtype={'u': np.str})
    count = []
    amount = []
    for index, row in data.iterrows():
        # 计算 count
        temp_count = [row.loc['u'], row.loc['b']]
        temp_count_data = list(row.loc[['u_b_count_2', 'u_b_count_3', 'u_b_count_4']])
        temp_count.append(max(temp_count_data))
        temp_count.append(sorted(temp_count_data)[1])
        temp_count.append(sum(temp_count_data) / 3)
        temp_count.append(np.std(temp_count_data))
        count.append(temp_count)
        # 计算amount
        temp_amount = []
        temp_amount_data = list(row.loc[['u_b_amount_2', 'u_b_amount_3', 'u_b_amount_4']])
        temp_amount.append(max(temp_amount_data))
        temp_amount.append(sorted(temp_amount_data)[1])
        temp_amount.append(sum(temp_amount_data) / 3)
        temp_amount.append(np.std(temp_amount_data))
        amount.append(temp_amount)
    new_data = [[a, b, a1, a2, a3, a4, a5, a6, a7, a8] for ([a, b, a1, a2, a3, a4], [a5, a6, a7, a8]) in zip(count, amount)]
    data_agg_count_u_b = pd.DataFrame(new_data, columns=['u', 'b', 'u_b_agg_count_max', 'u_b_agg_count_median',
                                                       'u_b_agg_count_mean', 'u_b_agg_count_std', 'u_b_agg_amount_max',
                                                       'u_b_agg_amount_median', 'u_b_agg_amount_mean', 'u_b_agg_amount_std'])
    data_agg_count_u_b.to_csv('data/type2/month_agg/count/data_u_b.csv', index=False)
    print('create data_agg_count_u_b grouped by vipno and bndno successfully')
    print('\n##################################################')
    print('create data_agg_count_u_c grouped by vipno and dptno...')
    data = pd.read_csv('data/type1/count/data_u_c.csv', dtype={'u': np.str})
    count = []
    amount = []
    for index, row in data.iterrows():
        # 计算 count
        temp_count = [row.loc['u'], row.loc['c']]
        temp_count_data = list(row.loc[['u_c_count_2', 'u_c_count_3', 'u_c_count_4']])
        temp_count.append(max(temp_count_data))
        temp_count.append(sorted(temp_count_data)[1])
        temp_count.append(sum(temp_count_data) / 3)
        temp_count.append(np.std(temp_count_data))
        count.append(temp_count)
        # 计算amount
        temp_amount = []
        temp_amount_data = list(row.loc[['u_c_amount_2', 'u_c_amount_3', 'u_c_amount_4']])
        temp_amount.append(max(temp_amount_data))
        temp_amount.append(sorted(temp_amount_data)[1])
        temp_amount.append(sum(temp_amount_data) / 3)
        temp_amount.append(np.std(temp_amount_data))
        amount.append(temp_amount)
    new_data = [[a, b, a1, a2, a3, a4, a5, a6, a7, a8] for ([a, b, a1, a2, a3, a4], [a5, a6, a7, a8]) in
                zip(count, amount)]
    data_agg_count_u_c = pd.DataFrame(new_data, columns=['u', 'c', 'u_c_agg_count_max', 'u_c_agg_count_median',
                                                         'u_c_agg_count_mean', 'u_c_agg_count_std',
                                                         'u_c_agg_amount_max',
                                                         'u_c_agg_amount_median', 'u_c_agg_amount_mean',
                                                         'u_c_agg_amount_std'])
    data_agg_count_u_c.to_csv('data/type2/month_agg/count/data_u_c.csv', index=False)
    print('create data_agg_count_u_c grouped by vipno and dptno successfully')
    print('\n##################################################')
    print('create data_agg_count_u_i grouped by vipno and pluno...')
    data = pd.read_csv('data/type1/count/data_u_i.csv', dtype={'u': np.str})
    count = []
    amount = []
    for index, row in data.iterrows():
        # 计算 count
        temp_count = [row.loc['u'], row.loc['i']]
        temp_count_data = list(row.loc[['u_i_count_2', 'u_i_count_3', 'u_i_count_4']])
        temp_count.append(max(temp_count_data))
        temp_count.append(sorted(temp_count_data)[1])
        temp_count.append(sum(temp_count_data) / 3)
        temp_count.append(np.std(temp_count_data))
        count.append(temp_count)
        # 计算amount
        temp_amount = []
        temp_amount_data = list(row.loc[['u_i_amount_2', 'u_i_amount_3', 'u_i_amount_4']])
        temp_amount.append(max(temp_amount_data))
        temp_amount.append(sorted(temp_amount_data)[1])
        temp_amount.append(sum(temp_amount_data) / 3)
        temp_amount.append(np.std(temp_amount_data))
        amount.append(temp_amount)
    new_data = [[a, b, a1, a2, a3, a4, a5, a6, a7, a8] for ([a, b, a1, a2, a3, a4], [a5, a6, a7, a8]) in
                zip(count, amount)]
    data_agg_count_u_i = pd.DataFrame(new_data, columns=['u', 'i', 'u_i_agg_count_max', 'u_i_agg_count_median',
                                                         'u_i_agg_count_mean', 'u_i_agg_count_std',
                                                         'u_i_agg_amount_max', 'u_i_agg_amount_median',
                                                         'u_i_agg_amount_mean', 'u_i_agg_amount_std'])
    data_agg_count_u_i.to_csv('data/type2/month_agg/count/data_u_i.csv', index=False)
    print('create data_agg_count_u_i grouped by vipno and pluno successfully')
    print('\n##################################################')
    print('create data_agg_count_b_c grouped by bndno and dptno...')
    data = pd.read_csv('data/type1/count/data_b_c.csv')
    count = []
    amount = []
    for index, row in data.iterrows():
        # 计算 count
        temp_count = [row.loc['b'], row.loc['c']]
        temp_count_data = list(row.loc[['b_c_count_2', 'b_c_count_3', 'b_c_count_4']])
        temp_count.append(max(temp_count_data))
        temp_count.append(sorted(temp_count_data)[1])
        temp_count.append(sum(temp_count_data) / 3)
        temp_count.append(np.std(temp_count_data))
        count.append(temp_count)
        # 计算amount
        temp_amount = []
        temp_amount_data = list(row.loc[['b_c_amount_2', 'b_c_amount_3', 'b_c_amount_4']])
        temp_amount.append(max(temp_amount_data))
        temp_amount.append(sorted(temp_amount_data)[1])
        temp_amount.append(sum(temp_amount_data) / 3)
        temp_amount.append(np.std(temp_amount_data))
        amount.append(temp_amount)
    new_data = [[a, b, a1, a2, a3, a4, a5, a6, a7, a8] for ([a, b, a1, a2, a3, a4], [a5, a6, a7, a8]) in
                zip(count, amount)]
    data_agg_count_b_c = pd.DataFrame(new_data, columns=['b', 'c', 'b_c_agg_count_max', 'b_c_agg_count_median',
                                                         'b_c_agg_count_mean', 'b_c_agg_count_std',
                                                         'b_c_agg_amount_max', 'b_c_agg_amount_median',
                                                         'b_c_agg_amount_mean', 'b_c_agg_amount_std'])
    data_agg_count_b_c.to_csv('data/type2/month_agg/count/data_b_c.csv')
    print('create data_agg_count_b_c grouped by bndno and dptno successfully')


def get_month_agg_diversity():
    print('\n##################################################')
    print('create data_agg_diversity_pluno grouped by vipno..')
    data = pd.read_csv('data/type1/diversity/data_u_i.csv', dtype={'u': np.str})
    count = []
    for index, row in data.iterrows():
        # 计算 count
        temp_count = [row.loc['u']]
        temp_count_data = list(row.loc[['u_i_count_unique_2', 'u_i_count_unique_3', 'u_i_count_unique_4']])
        temp_count.append(max(temp_count_data))
        temp_count.append(sorted(temp_count_data)[1])
        temp_count.append(sum(temp_count_data) / 3)
        temp_count.append(np.std(temp_count_data))
        count.append(temp_count)
    new_data = pd.DataFrame(count, columns=['u', 'u_i_agg_count_unique_max', 'u_i_agg_count_unique_median',
                                                         'u_i_agg_count_unique_mean', 'u_i_agg_count_unique_std'])
    new_data.to_csv('data/type2/month_agg/diversity/data_u_i.csv', index=False)
    print('create data_agg_diversity_pluno grouped by vipno successfully')
    print('\n##################################################')
    print('create data_agg_diversity_bndno grouped by vipno..')
    data = pd.read_csv('data/type1/diversity/data_u_b.csv', dtype={'u': np.str})
    count = []
    for index, row in data.iterrows():
        # 计算 count
        temp_count = [row.loc['u']]
        temp_count_data = list(row.loc[['u_b_count_unique_2', 'u_b_count_unique_3', 'u_b_count_unique_4']])
        temp_count.append(max(temp_count_data))
        temp_count.append(sorted(temp_count_data)[1])
        temp_count.append(sum(temp_count_data) / 3)
        temp_count.append(np.std(temp_count_data))
        count.append(temp_count)
    new_data = pd.DataFrame(count, columns=['u', 'u_b_agg_count_unique_max', 'u_b_agg_count_unique_median',
                                            'u_b_agg_count_unique_mean', 'u_b_agg_count_unique_std'])
    new_data.to_csv('data/type2/month_agg/diversity/data_u_b.csv', index=False)
    print('create data_agg_diversity_bndno grouped by vipno successfully')
    print('\n##################################################')
    print('create data_agg_diversity_dptno grouped by vipno..')
    data = pd.read_csv('data/type1/diversity/data_u_c.csv', dtype={'u': np.str})
    count = []
    for index, row in data.iterrows():
        # 计算 count
        temp_count = [row.loc['u']]
        temp_count_data = list(row.loc[['u_c_count_unique_2', 'u_c_count_unique_3', 'u_c_count_unique_4']])
        temp_count.append(max(temp_count_data))
        temp_count.append(sorted(temp_count_data)[1])
        temp_count.append(sum(temp_count_data) / 3)
        temp_count.append(np.std(temp_count_data))
        count.append(temp_count)
    new_data = pd.DataFrame(count, columns=['u', 'u_c_agg_count_unique_max', 'u_c_agg_count_unique_median',
                                            'u_c_agg_count_unique_mean', 'u_c_agg_count_unique_std'])
    new_data.to_csv('data/type2/month_agg/diversity/data_u_c.csv', index=False)
    print('create data_agg_diversity_dptno grouped by vipno successfully')
    print('\n##################################################')
    print('create data_agg_diversity_pluno grouped by bndno..')
    data = pd.read_csv('data/type1/diversity/data_b_i.csv')
    count = []
    for index, row in data.iterrows():
        # 计算 count
        temp_count = [row.loc['b']]
        temp_count_data = list(row.loc[['b_i_count_unique_2', 'b_i_count_unique_3', 'b_i_count_unique_4']])
        temp_count.append(max(temp_count_data))
        temp_count.append(sorted(temp_count_data)[1])
        temp_count.append(sum(temp_count_data) / 3)
        temp_count.append(np.std(temp_count_data))
        count.append(temp_count)
    new_data = pd.DataFrame(count, columns=['b', 'b_i_agg_count_unique_max', 'b_i_agg_count_unique_median',
                                            'b_i_agg_count_unique_mean', 'b_i_agg_count_unique_std'])
    new_data.to_csv('data/type2/month_agg/diversity/data_b_i.csv', index=False)
    print('create data_agg_diversity_pluno grouped by bndno successfully')
    print('\n##################################################')
    print('create data_agg_diversity_pluno grouped by dptno..')
    data = pd.read_csv('data/type1/diversity/data_c_i.csv')
    count = []
    for index, row in data.iterrows():
        # 计算 count
        temp_count = [row.loc['c']]
        temp_count_data = list(row.loc[['c_i_count_unique_2', 'c_i_count_unique_3', 'c_i_count_unique_4']])
        temp_count.append(max(temp_count_data))
        temp_count.append(sorted(temp_count_data)[1])
        temp_count.append(sum(temp_count_data) / 3)
        temp_count.append(np.std(temp_count_data))
        count.append(temp_count)
    new_data = pd.DataFrame(count, columns=['c', 'c_i_agg_count_unique_max', 'c_i_agg_count_unique_median',
                                            'c_i_agg_count_unique_mean', 'c_i_agg_count_unique_std'])
    new_data.to_csv('data/type2/month_agg/diversity/data_c_i.csv', index=False)
    print('create data_agg_diversity_pluno grouped by dptno successfully')


def get_month_agg_penetration():
    print('\n##################################################')
    print('create data_agg_penetration_vipno grouped by bndno..')
    data = pd.read_csv('data/type1/penetration/data_b.csv')
    count = []
    for index, row in data.iterrows():
        # 计算 count
        temp_count = [row.loc['b']]
        temp_count_data = list(row.loc[['b_user_2', 'b_user_3', 'b_user_4']])
        temp_count.append(max(temp_count_data))
        temp_count.append(sorted(temp_count_data)[1])
        temp_count.append(round(sum(temp_count_data) / 3, 2))
        temp_count.append(round(np.std(temp_count_data),2))
        count.append(temp_count)
    new_data = pd.DataFrame(count, columns=['b', 'b_agg_user_max', 'b_agg_user_max_median',
                                            'b_agg_user_max_mean', 'b_agg_user_max_std'])
    new_data.to_csv('data/type2/month_agg/penetration/data_b.csv', index=False)
    print('create data_agg_diversity_vipno grouped by bndno successfully')
    print('\n##################################################')
    print('create data_agg_penetration_vipno grouped by dptno..')
    data = pd.read_csv('data/type1/penetration/data_c.csv')
    count = []
    for index, row in data.iterrows():
        # 计算 count
        temp_count = [row.loc['c']]
        temp_count_data = list(row.loc[['c_user_2', 'c_user_3', 'c_user_4']])
        temp_count.append(max(temp_count_data))
        temp_count.append(sorted(temp_count_data)[1])
        temp_count.append(round(sum(temp_count_data) / 3, 2))
        temp_count.append(round(np.std(temp_count_data),2))
        count.append(temp_count)
    new_data = pd.DataFrame(count, columns=['c', 'c_agg_user_max', 'c_agg_user_max_median',
                                            'c_agg_user_max_mean', 'c_agg_user_max_std'])
    new_data.to_csv('data/type2/month_agg/penetration/data_c.csv', index=False)
    print('create data_agg_diversity_vipno grouped by dptno successfully')
    print('\n##################################################')
    print('create data_agg_penetration_vipno grouped by pluno..')
    data = pd.read_csv('data/type1/penetration/data_i.csv')
    count = []
    for index, row in data.iterrows():
        # 计算 count
        temp_count = [row.loc['i']]
        temp_count_data = list(row.loc[['i_user_2', 'i_user_3', 'i_user_4']])
        temp_count.append(max(temp_count_data))
        temp_count.append(sorted(temp_count_data)[1])
        temp_count.append(round(sum(temp_count_data) / 3, 2))
        temp_count.append(round(np.std(temp_count_data), 2))
        count.append(temp_count)
    new_data = pd.DataFrame(count, columns=['i', 'i_agg_user_max', 'i_agg_user_max_median',
                                            'i_agg_user_max_mean', 'i_agg_user_max_std'])
    new_data.to_csv('data/type2/month_agg/penetration/data_i.csv', index=False)
    print('create data_agg_diversity_vipno grouped by pluno successfully')


def get_user_agg():
    data = pd.read_csv('data/raw_data.csv')
    data = data[data['time'] < 5]
    print('\n##################################################')
    print('create data_user_agg_vipno grouped by bndno..')
    new_data = []
    for index, temp_data in data.groupby('bndno'):
        temp = [index]
        times = []
        amount = []
        for index2, temp_data_2 in temp_data.groupby('vipno'):
            times.append(len(temp_data_2.iloc[:, 0]))
            amount.append(sum(temp_data_2.loc[:, 'amt']))
        temp.append(round(sum(times) / len(times), 2))
        temp.append(round(np.std(times), 2))
        temp.append(max(times))
        temp.append(sorted(times)[int(len(times)/2)])
        temp.append(round(sum(amount) / len(amount), 2))
        temp.append(round(np.std(amount), 2))
        temp.append(round(max(amount), 2))
        temp.append(round(sorted(amount)[int(len(amount) / 2)], 2))
        new_data.append(temp)
    new_data = pd.DataFrame(new_data, columns=['b', 'b_user_agg_count_mean', 'b_user_agg_count_std',
                                               'b_user_agg_count_max', 'b_user_agg_count_median',
                                               'b_user_agg_amount_mean', 'b_user_agg_amount_std',
                                               'b_user_agg_amount_max', 'b_user_agg_amount_median'])
    new_data.to_csv('data/type2/user_agg/user_agg_b.csv', index=False)
    print('create data_user_agg_vipno grouped by bndno successfully')
    print('\n##################################################')
    print('create data_user_agg_dptno grouped by vipno..')
    new_data = []
    for index, temp_data in data.groupby('dptno'):
        temp = [index]
        times = []
        amount = []
        for index2, temp_data_2 in temp_data.groupby('vipno'):
            times.append(len(temp_data_2.iloc[:, 0]))
            amount.append(sum(temp_data_2.loc[:, 'amt']))
        temp.append(round(sum(times) / len(times), 2))
        temp.append(round(np.std(times), 2))
        temp.append(max(times))
        temp.append(sorted(times)[int(len(times) / 2)])
        temp.append(round(sum(amount) / len(amount), 2))
        temp.append(round(np.std(amount), 2))
        temp.append(round(max(amount), 2))
        temp.append(round(sorted(amount)[int(len(amount) / 2)], 2))
        new_data.append(temp)
    new_data = pd.DataFrame(new_data, columns=['c', 'c_user_agg_count_mean', 'c_user_agg_count_std',
                                               'c_user_agg_count_max', 'c_user_agg_count_median',
                                               'c_user_agg_amount_mean', 'c_user_agg_amount_std',
                                               'c_user_agg_amount_max', 'c_user_agg_amount_median'])
    new_data.to_csv('data/type2/user_agg/user_agg_c.csv', index=False)
    print('create data_user_agg_vipno grouped by dptno successfully')
    print('\n##################################################')
    print('create data_user_agg_vipno grouped by pluno..')
    new_data = []
    for index, temp_data in data.groupby('pluno'):
        temp = [index]
        times = []
        amount = []
        for index2, temp_data_2 in temp_data.groupby('vipno'):
            times.append(len(temp_data_2.iloc[:, 0]))
            amount.append(sum(temp_data_2.loc[:, 'amt']))
        temp.append(round(sum(times) / len(times), 2))
        temp.append(round(np.std(times), 2))
        temp.append(max(times))
        temp.append(sorted(times)[int(len(times) / 2)])
        temp.append(round(sum(amount) / len(amount), 2))
        temp.append(round(np.std(amount), 2))
        temp.append(round(max(amount), 2))
        temp.append(round(sorted(amount)[int(len(amount) / 2)], 2))
        new_data.append(temp)
    new_data = pd.DataFrame(new_data, columns=['i', 'i_user_agg_count_mean', 'i_user_agg_count_std',
                                               'i_user_agg_count_max', 'i_user_agg_count_median',
                                               'i_user_agg_amount_mean', 'i_user_agg_amount_std',
                                               'i_user_agg_amount_max', 'i_user_agg_amount_median'])
    new_data.to_csv('data/type2/user_agg/user_agg_i.csv', index=False)
    print('create data_user_agg_vipno grouped by pluno successfully')


def get_multiple_agg():
    data = pd.read_csv('data/raw_data.csv')
    data = data[data['time'] < 5]
    print('\n##################################################')
    print('create data_user_agg_bndno grouped by vipno..')
    new_data = []
    for index, temp_data in data.groupby('vipno'):
        temp = [index]
        times = []
        amount = []
        for index2, temp_data_2 in temp_data.groupby('bndno'):
            times.append(len(temp_data_2.iloc[:, 0]))
            amount.append(sum(temp_data_2.loc[:, 'amt']))
        if len(times) == 0:
            continue
        temp.append(round(sum(times) / len(times), 2))
        temp.append(round(np.std(times), 2))
        temp.append(max(times))
        temp.append(sorted(times)[int(len(times) / 2)])
        temp.append(round(sum(amount) / len(amount), 2))
        temp.append(round(np.std(amount), 2))
        temp.append(round(max(amount), 2))
        temp.append(round(sorted(amount)[int(len(amount) / 2)], 2))
        new_data.append(temp)
    new_data = pd.DataFrame(new_data, columns=['u', 'u_b_user_agg_count_mean', 'u_b_user_agg_count_std',
                                               'u_b_user_agg_count_max', 'u_b_user_agg_count_median',
                                               'u_b_user_agg_amount_mean', 'u_b_user_agg_amount_std',
                                               'u_b_user_agg_amount_max', 'u_b_user_agg_amount_median'])
    new_data.to_csv('data/type2/brand_agg.csv', index=False)
    print('create data_user_agg_bndno grouped by vipno successfully')
    print('\n##################################################')
    print('create data_user_agg_dptno grouped by vipno..')
    new_data = []
    for index, temp_data in data.groupby('vipno'):
        temp = [index]
        times = []
        amount = []
        for index2, temp_data_2 in temp_data.groupby('dptno'):
            times.append(len(temp_data_2.iloc[:, 0]))
            amount.append(sum(temp_data_2.loc[:, 'amt']))
        if len(times) == 0:
            continue
        temp.append(round(sum(times) / len(times), 2))
        temp.append(round(np.std(times), 2))
        temp.append(max(times))
        temp.append(sorted(times)[int(len(times) / 2)])
        temp.append(round(sum(amount) / len(amount), 2))
        temp.append(round(np.std(amount), 2))
        temp.append(round(max(amount), 2))
        temp.append(round(sorted(amount)[int(len(amount) / 2)], 2))
        new_data.append(temp)
    new_data = pd.DataFrame(new_data, columns=['u', 'u_c_user_agg_count_mean', 'u_c_user_agg_count_std',
                                               'u_c_user_agg_count_max', 'u_c_user_agg_count_median',
                                               'u_c_user_agg_amount_mean', 'u_c_user_agg_amount_std',
                                               'u_c_user_agg_amount_max', 'u_c_user_agg_amount_median'])
    new_data.to_csv('data/type2/category_agg.csv', index=False)
    print('create data_user_agg_dptno grouped by vipno successfully')
    print('\n##################################################')
    print('create data_user_agg_pluno grouped by vipno..')
    new_data = []
    for index, temp_data in data.groupby('vipno'):
        temp = [index]
        times = []
        amount = []
        for index2, temp_data_2 in temp_data.groupby('pluno'):
            times.append(len(temp_data_2.iloc[:, 0]))
            amount.append(sum(temp_data_2.loc[:, 'amt']))
        if len(times) == 0:
            continue
        temp.append(round(sum(times) / len(times), 2))
        temp.append(round(np.std(times), 2))
        temp.append(max(times))
        temp.append(sorted(times)[int(len(times) / 2)])
        temp.append(round(sum(amount) / len(amount), 2))
        temp.append(round(np.std(amount), 2))
        temp.append(round(max(amount), 2))
        temp.append(round(sorted(amount)[int(len(amount) / 2)], 2))
        new_data.append(temp)
    new_data = pd.DataFrame(new_data, columns=['u', 'u_i_user_agg_count_mean', 'u_i_user_agg_count_std',
                                               'u_i_user_agg_count_max', 'u_i_user_agg_count_median',
                                               'u_i_user_agg_amount_mean', 'u_i_user_agg_amount_std',
                                               'u_i_user_agg_amount_max', 'u_i_user_agg_amount_median'])
    new_data.to_csv('data/type2/item_agg.csv', index=False)
    print('create data_user_agg_pluno grouped by vipno successfully')


get_multiple_agg()