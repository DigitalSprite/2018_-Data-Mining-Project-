import pandas as pd
import numpy as np

def getMatrix(file_name):
    df = pd.read_csv(file_name)
    line_index = set(df.iloc[:,4])
    row_index = set(df.iloc[:,6])
    new_df = pd.DataFrame(np.zeros((row_index.__len__(), line_index.__len__()), dtype=np.int), index=row_index, columns=line_index)
    temp = df.loc[:,['vipno','pluno','amt']]
    for i in range(temp.shape[0]):
        if i == 0:
            continue
        new_df.loc[temp.iloc[i, 1], temp.iloc[i, 0]] += int(temp.iloc[i, 2])
        if (temp.iloc[i, 2] - int(temp.iloc[i, 2]) == 0.5):
            new_df.loc[temp.iloc[i, 1], temp.iloc[i, 0]] += 1
    return new_df

if __name__ == '__main__':
    matrix = getMatrix("../../trade.csv")
    print("产生新矩阵" + str(matrix))