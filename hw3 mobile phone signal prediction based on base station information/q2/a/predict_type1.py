import pandas as pd


def get_raw_data():
    print('creating raw data...')
    data = pd.read_csv('../../data/trade_new.csv').loc[:, ['sldatime', 'vipno', 'dptno', 'bndno', 'pluno', 'amt', 'qty']]
    update_times = []
    for index, row in data.iterrows():
        update_times.append([int(row.loc['sldatime'].split('-')[1])])
    update_times = pd.DataFrame(update_times, columns=['time'], index=data.index)
    # print(update_times)
    data = pd.concat([data.drop(['sldatime'], axis=1), update_times], axis=1)
    data.to_csv('data/raw_data.csv')
    print('create successfully and saved as raw_data.csv')
    return data


def get_count_data():
    data = pd.read_csv('data/raw_data.csv')
    print('\n##################################################')
    print('create count grouped by vipno...')
    data = data[data['time'] > 4]
    data_list = []
    for tt, data_new in data.groupby('vipno'):
        data_list.append(tt)
    data_u = pd.DataFrame(data_list, columns=['u'])
    count = []
    amount = []
    for vipno, temp_data in data.groupby('vipno'):
        temp_data = temp_data[temp_data['time'] > 4].sort_values(by='time')
        temp_count = [vipno, sum(temp_data.loc[:, 'qty'])]
        temp_amount = [vipno, sum(temp_data.loc[:, 'amt'])]
        for month in range(5, 8, 1):
            temp = temp_data[temp_data['time'] == month]
            temp_count.append(sum(temp.loc[:, 'qty']))
            temp_amount.append(sum(temp.loc[:, 'amt']))
        count.append(temp_count)
        amount.append(temp_amount)
    count_pd = pd.DataFrame(count, columns=['u', 'u_count_all', 'u_count_2', 'u_count_3', 'u_count_4'])
    amount_pd = pd.DataFrame(amount, columns=['u', 'u_amount_all', 'u_amount_2', 'u_amount_3', 'u_amount_4'])
    data_u = pd.merge(data_u, count_pd, how='left')
    data_u = pd.merge(data_u, amount_pd, how='left')
    data_u.to_csv('predict/type1/count/data_u.csv', index=False)
    print('creat count grouped by user successfully')
    print('\n##################################################')
    print('create count grouped by brand...')
    data_list = []
    for tt, data_new in data.groupby(['bndno']):
        data_list.append(tt)
    data_b = pd.DataFrame(data_list, columns=['b'])
    count = []
    amount = []
    for bndno, temp_data in data.groupby('bndno'):
        temp_data = temp_data[temp_data['time'] > 4].sort_values(by='time')
        temp_count = [bndno, sum(temp_data.loc[:, 'qty'])]
        temp_amount = [bndno, sum(temp_data.loc[:, 'amt'])]
        for month in range(5, 8, 1):
            temp = temp_data[temp_data['time'] == month]
            temp_count.append(sum(temp.loc[:, 'qty']))
            temp_amount.append(sum(temp.loc[:, 'amt']))
        count.append(temp_count)
        amount.append(temp_amount)
    count_pd = pd.DataFrame(count, columns=['b', 'b_count_all', 'b_count_2', 'b_count_3', 'b_count_4'])
    amount_pd = pd.DataFrame(amount, columns=['b', 'b_amount_all', 'b_amount_2', 'b_amount_3', 'b_amount_4'])
    data_b = pd.merge(data_b, count_pd, how='left')
    data_b = pd.merge(data_b, amount_pd, how='left')
    data_b.to_csv('predict/type1/count/data_b.csv', index=False)
    print('creat count grouped by brand successfully')
    print('\n##################################################')
    print('create count grouped by category...')
    data_list = []
    for tt, data_new in data.groupby('dptno'):
        data_list.append(tt)
    data_c = pd.DataFrame(data_list, columns=['c'])
    count = []
    amount = []
    for dptno, temp_data in data.groupby('dptno'):
        temp_data = temp_data[temp_data['time'] > 4].sort_values(by='time')
        temp_count = [dptno, sum(temp_data.loc[:, 'qty'])]
        temp_amount = [dptno, sum(temp_data.loc[:, 'amt'])]
        for month in range(5, 8, 1):
            temp = temp_data[temp_data['time'] == month]
            temp_count.append(sum(temp.loc[:, 'qty']))
            temp_amount.append(sum(temp.loc[:, 'amt']))
        count.append(temp_count)
        amount.append(temp_amount)
    count_pd = pd.DataFrame(count, columns=['c', 'c_count_all', 'c_count_2', 'c_count_3', 'c_count_4'])
    mount_pd = pd.DataFrame(amount, columns=['c', 'c_amount_all', 'c_amount_2', 'c_amount_3', 'c_amount_4'])
    data_c = pd.merge(data_c, count_pd, how='left')
    data_c = pd.merge(data_c, mount_pd, how='left')
    data_c.to_csv('predict/type1/count/data_c.csv', index=False)
    print('creat count grouped by category successfully')
    print('\n##################################################')
    print('create count grouped by item...')
    data_list = []
    for tt, data_new in data.groupby(['pluno']):
        data_list.append(tt)
    data_i = pd.DataFrame(data_list, columns=['i'])
    count = []
    amount = []
    for dptno, temp_data in data.groupby('pluno'):
        temp_data = temp_data[temp_data['time'] > 4].sort_values(by='time')
        temp_count = [dptno, sum(temp_data.loc[:, 'qty'])]
        temp_amount = [dptno, sum(temp_data.loc[:, 'amt'])]
        for month in range(5, 8, 1):
            temp = temp_data[temp_data['time'] == month]
            temp_count.append(sum(temp.loc[:, 'qty']))
            temp_amount.append(sum(temp.loc[:, 'amt']))
        count.append(temp_count)
        amount.append(temp_amount)
    count_pd = pd.DataFrame(count, columns=['i', 'i_count_all', 'i_count_2', 'i_count_3', 'i_count_4'])
    mount_pd = pd.DataFrame(amount, columns=['i', 'i_amount_all', 'i_amount_2', 'i_amount_3', 'i_amount_4'])
    data_i = pd.merge(data_i, count_pd, how='left')
    data_i = pd.merge(data_i, mount_pd, how='left')
    data_i.to_csv('predict/type1/count/data_i.csv', index=False)
    print('creat count grouped by item successfully')
    print('\n##################################################')
    print('create count grouped by user and brand...')
    data_list = []
    for tt, data_new in data.groupby(['vipno', 'bndno']):
        data_list.append(tt)
    data_u_b = pd.DataFrame(data_list, columns=['u', 'b'])
    count = []
    amount = []
    for (vipno, bndno), temp_data in data.groupby(['vipno', 'bndno']):
        temp_data = temp_data[temp_data['time'] > 4].sort_values(by='time')
        temp_count = [vipno, bndno, sum(temp_data.loc[:, 'qty'])]
        temp_amount = [vipno, bndno, sum(temp_data.loc[:, 'amt'])]
        for month in range(5, 8, 1):
            temp = temp_data[temp_data['time'] == month]
            temp_count.append(sum(temp.loc[:, 'qty']))
            temp_amount.append(sum(temp.loc[:, 'amt']))
        count.append(temp_count)
        amount.append(temp_amount)
    count_pd = pd.DataFrame(count, columns=['u', 'b', 'u_b_count_all', 'u_b_count_2', 'u_b_count_3', 'u_b_count_4'])
    mount_pd = pd.DataFrame(amount, columns=['u', 'b', 'u_b_amount_all', 'u_b_amount_2', 'u_b_amount_3', 'u_b_amount_4'])
    data_u_b = pd.merge(data_u_b, count_pd, how='left')
    data_u_b = pd.merge(data_u_b, mount_pd, how='left')
    data_u_b.to_csv('predict/type1/count/data_u_b.csv', index=False)
    print('creat count grouped by user and brand successfully')
    print('\n##################################################')
    print('create count grouped by user and category...')
    data_list = []
    for tt, data_new in data.groupby(['vipno', 'dptno']):
        data_list.append(tt)
    data_u_c = pd.DataFrame(data_list, columns=['u', 'c'])
    count = []
    amount = []
    for (vipno, bndno), temp_data in data.groupby(['vipno', 'dptno']):
        temp_data = temp_data[temp_data['time'] > 4].sort_values(by='time')
        temp_count = [vipno, bndno, sum(temp_data.loc[:, 'qty'])]
        temp_amount = [vipno, bndno, sum(temp_data.loc[:, 'amt'])]
        for month in range(5, 8, 1):
            temp = temp_data[temp_data['time'] == month]
            temp_count.append(sum(temp.loc[:, 'qty']))
            temp_amount.append(sum(temp.loc[:, 'amt']))
        count.append(temp_count)
        amount.append(temp_amount)
    count_pd = pd.DataFrame(count, columns=['u', 'c', 'u_c_count_all', 'u_c_count_2', 'u_c_count_3', 'u_c_count_4'])
    mount_pd = pd.DataFrame(amount,
                            columns=['u', 'c', 'u_c_amount_all', 'u_c_amount_2', 'u_c_amount_3', 'u_c_amount_4'])
    data_u_c = pd.merge(data_u_c, count_pd, how='left')
    data_u_c = pd.merge(data_u_c, mount_pd, how='left')
    data_u_c.to_csv('predict/type1/count/data_u_c.csv', index=False)
    print('creat count grouped by user and category successfully')
    print('\n##################################################')
    print('create count grouped by user and item...')
    data_list = []
    for tt, data_new in data.groupby(['vipno', 'pluno']):
        data_list.append(tt)
    data_u_i = pd.DataFrame(data_list, columns=['u', 'i'])
    count = []
    amount = []
    for (vipno, bndno), temp_data in data.groupby(['vipno', 'pluno']):
        temp_data = temp_data[temp_data['time'] > 4].sort_values(by='time')
        temp_count = [vipno, bndno, sum(temp_data.loc[:, 'qty'])]
        temp_amount = [vipno, bndno, sum(temp_data.loc[:, 'amt'])]
        for month in range(5, 8, 1):
            temp = temp_data[temp_data['time'] == month]
            temp_count.append(sum(temp.loc[:, 'qty']))
            temp_amount.append(sum(temp.loc[:, 'amt']))
        count.append(temp_count)
        amount.append(temp_amount)
    count_pd = pd.DataFrame(count, columns=['u', 'i', 'u_i_count_all', 'u_i_count_2', 'u_i_count_3', 'u_i_count_4'])
    mount_pd = pd.DataFrame(amount,
                            columns=['u', 'i', 'u_i_amount_all', 'u_i_amount_2', 'u_i_amount_3', 'u_i_amount_4'])
    data_u_i = pd.merge(data_u_i, count_pd, how='left')
    data_u_i = pd.merge(data_u_i, mount_pd, how='left')
    data_u_i.to_csv('predict/type1/count/data_u_i.csv', index=False)
    print('creat count grouped by user and item successfully')
    print('\n##################################################')
    print('create count grouped by brand and category...')
    data_list = []
    for tt, data_new in data.groupby(['bndno', 'dptno']):
        data_list.append(tt)
    data_b_c = pd.DataFrame(data_list, columns=['b', 'c'])
    count = []
    amount = []
    for (vipno, bndno), temp_data in data.groupby(['bndno', 'dptno']):
        temp_data = temp_data[temp_data['time'] > 4].sort_values(by='time')
        temp_count = [vipno, bndno, sum(temp_data.loc[:, 'qty'])]
        temp_amount = [vipno, bndno, sum(temp_data.loc[:, 'amt'])]
        for month in range(5, 8, 1):
            temp = temp_data[temp_data['time'] == month]
            temp_count.append(sum(temp.loc[:, 'qty']))
            temp_amount.append(sum(temp.loc[:, 'amt']))
        count.append(temp_count)
        amount.append(temp_amount)
    count_pd = pd.DataFrame(count, columns=['b', 'c', 'b_c_count_all', 'b_c_count_2', 'b_c_count_3', 'b_c_count_4'])
    mount_pd = pd.DataFrame(amount,
                            columns=['b', 'c', 'b_c_amount_all', 'b_c_amount_2', 'b_c_amount_3', 'b_c_amount_4'])
    data_b_c = pd.merge(data_b_c, count_pd, how='left')
    data_b_c = pd.merge(data_b_c, mount_pd, how='left')
    data_b_c.to_csv('predict/type1/count/data_b_c.csv', index=False)
    print('create count grouped by brand and category successfully')


def get_diversity_data():
    print('\n##################################################')
    print('create diversity_pluno grouped by vipno...')
    data = pd.read_csv('data/raw_data.csv')
    data = data[data['time'] > 4]
    count_i = []
    count_b = []
    count_c = []
    for index, temp_data in data.groupby('vipno'):
        temp = [index]
        temp.append(len(set(list((temp_data.loc[:, 'pluno'])))))
        for time in range(5, 8, 1):
            t = temp_data[temp_data['time'] == time]
            temp.append(len(set(list((t.loc[:, 'pluno'])))))
        count_i.append(temp)
    data_u_i = pd.DataFrame(count_i, columns=['u', 'u_i_count_unique_all', 'u_i_count_unique_2', 'u_i_count_unique_3',
                                              'u_i_count_unique_4'])
    data_u_i.to_csv('predict/type1/diversity/data_u_i.csv', index=False)
    print('create diversity_pluno grouped by vipno successfully')
    print('\n##################################################')
    print('create diversity_bndno grouped by vipno...')
    for index, temp_data in data.groupby('vipno'):
        temp = [index]
        temp.append(len(set(list((temp_data.loc[:, 'bndno'])))))
        for time in range(5, 8, 1):
            t = temp_data[temp_data['time'] == time]
            temp.append(len(set(list((t.loc[:, 'bndno'])))))
        count_b.append(temp)
    data_u_b = pd.DataFrame(count_b, columns=['u', 'u_b_count_unique_all', 'u_b_count_unique_2', 'u_b_count_unique_3',
                                              'u_b_count_unique_4'])
    data_u_b.to_csv('predict/type1/diversity/data_u_b.csv', index=False)
    print('create diversity_bndno grouped by vipno successfully')
    print('\n##################################################')
    print('create diversity_dptno grouped by vipno...')
    for index, temp_data in data.groupby('vipno'):
        temp = [index]
        temp.append(len(set(list((temp_data.loc[:, 'dptno'])))))
        for time in range(5,8,1):
            t = temp_data[temp_data['time'] == time]
            temp.append(len(set(list((t.loc[:, 'dptno'])))))
        count_c.append(temp)
    data_u_c = pd.DataFrame(count_c, columns=['u', 'u_c_count_unique_all', 'u_c_count_unique_2', 'u_c_count_unique_3',
                                              'u_c_count_unique_4'])
    data_u_c.to_csv('predict/type1/diversity/data_u_c.csv', index=False)
    print('create diversity_dptno grouped by vipno successfully')
    print('\n##################################################')
    print('create diversity_pluno grouped by bndno...')
    count_b_i = []
    for index, temp_data in data.groupby('bndno'):
        temp = [index]
        temp.append(len(set(list((temp_data.loc[:, 'pluno'])))))
        for time in range(5, 8, 1):
            t = temp_data[temp_data['time'] == time]
            temp.append(len(set(list((t.loc[:, 'pluno'])))))
        count_b_i.append(temp)
    data_b_i = pd.DataFrame(count_b_i,
                            columns=['b', 'b_i_count_unique_all', 'b_i_count_unique_2',
                                     'b_i_count_unique_3', 'b_i_count_unique_4'])
    data_b_i.to_csv('predict/type1/diversity/data_b_i.csv', index=False)
    print('create diversity_pluno grouped by bndno successfully')
    print('\n##################################################')
    print('create diversity_pluno grouped by dptno...')
    count_c_i = []
    for index, temp_data in data.groupby('dptno'):
        temp = [index]
        temp.append(len(set(list((temp_data.loc[:, 'pluno'])))))
        for time in range(5, 8, 1):
            t = temp_data[temp_data['time'] == time]
            temp.append(len(set(list((t.loc[:, 'pluno'])))))
        count_c_i.append(temp)
    data_c_i = pd.DataFrame(count_c_i,
                            columns=['c', 'c_i_count_unique_all', 'c_i_count_unique_2',
                                     'c_i_count_unique_3', 'c_i_count_unique_4'])
    data_c_i.to_csv('predict/type1/diversity/data_c_i.csv', index=False)
    print('create diversity_pluno grouped by dptno successfully')


def get_penetration_data():
    data = pd.read_csv('data/raw_data.csv')
    data = data[data['time'] > 4]
    print('\n##################################################')
    print('create penetration grouped by brand ...')
    count_b = []
    for brand, temp_data in data.groupby('bndno'):
        temp = [brand, len(set(list(temp_data.loc[:, 'vipno'])))]
        for time in range(5, 8, 1):
            t_data = temp_data[temp_data['time'] == time]
            temp.append(len(set(list(t_data.loc[:, 'vipno']))))
        count_b.append(temp)
    data_b = pd.DataFrame(count_b, columns=['b', 'b_user_all', 'b_user_2', 'b_user_3', 'b_user_4'])
    data_b.to_csv('predict/type1/penetration/data_b.csv', index=False)
    print('create penetration grouped by brand successfully')
    print('\n##################################################')
    print('create penetration grouped by category ...')
    count_c = []
    for category, temp_data in data.groupby('dptno'):
        temp = [category, len(set(list(temp_data.loc[:, 'vipno'])))]
        for time in range(5, 8, 1):
            t_data = temp_data[temp_data['time'] == time]
            temp.append(len(set(list(t_data.loc[:, 'vipno']))))
        count_c.append(temp)
    data_c = pd.DataFrame(count_c, columns=['c', 'c_user_all', 'c_user_2', 'c_user_3', 'c_user_4'])
    data_c.to_csv('predict/type1/penetration/data_c.csv', index=False)
    print('create penetration grouped by category successfully')
    print('\n##################################################')
    print('create penetration grouped by category ...')
    count_i = []
    for item, temp_data in data.groupby('pluno'):
        temp = [item, len(set(list(temp_data.loc[:, 'vipno'])))]
        for time in range(5, 8, 1):
            t_data = temp_data[temp_data['time'] == time]
            temp.append(len(set(list(t_data.loc[:, 'vipno']))))
        count_i.append(temp)
    data_i = pd.DataFrame(count_i, columns=['i', 'i_user_all', 'i_user_2', 'i_user_3', 'i_user_4'])
    data_i.to_csv('predict/type1/penetration/data_i.csv', index=False)
    print('create penetration grouped by category successfully')


get_penetration_data()

