import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# 最大最小值规范化
x_train = np.array([[1., -1., 2.],
                    [2., 0., 0.],
                    [0., 1., -1.]])
min_max_scaler = preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
print(x_train_minmax)
# Z-score 规范化
x_scaled = preprocessing.scale(x_train)
print(x_scaled)
# 小数定标
print(x_train / 10)

test_dict={
    'id':[1, 2, 3, 4, 5, 6],
    'name':['Alice', 'Bob', 'Cindy', 'Eric', 'Helen', 'Grace'],
    'math':[90, 90, 99, 78, 97, 93],
    'sex':['F', 'M', 'F', 'M', 'M', 'M']
}
df = pd.DataFrame(test_dict)
print(df)
class_le = LabelEncoder()
df['sex'] = class_le.fit_transform(df['sex'].values)
print(df)

df = pd.DataFrame(test_dict)
enumerate(np.unique(df['sex']))
class_mapping = {label: idx for idx, label in [[1,'M'],[0,'F']]}
df['sex'] = df['sex'].map(class_mapping)
print(df)

df = pd.DataFrame(test_dict)
pf = pd.get_dummies(df[['sex']])
df = pd.concat([df, pf], axis=1)
df.drop(['sex'], axis=1, inplace=True)
print(df)