# -*- coding: utf-8 -*-
"""
Enron Fraud Dataset 安隆公司詐欺案資料集
@author: Martin lee
Date：2019/02/15

"""

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import  StandardScaler,MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import  metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
os.chdir('F:\\learning\\機器學習百日馬拉松活動\\期中考')

#============================check the data type and shape

data = pd.read_csv('data/train_data.csv')
data.shape
data.dtypes.reset_index()
data.head()


#確定只有 int64, float64, object 三種類型後對欄位名稱執行迴圈, 分別將欄位名稱存於三個 list 中
int_features = []
float_features = []
object_features = []
for dtype, feature in zip(data.dtypes, data.columns): #打包成同一個組別 http://www.runoob.com/python/python-func-zip.html
    if dtype == 'float64':
        float_features.append(feature)
    elif dtype == 'int64':
        int_features.append(feature)
    else:
        object_features.append(feature)
print(f'{len(int_features)} Integer Features : {int_features}\n')
print(f'{len(float_features)} Float Features : {float_features}\n')
print(f'{len(object_features)} Object Features : {object_features}')


#=============================EDA 特徵工程==============================
float_data = data[float_features].loc[:,data[float_features].isnull().sum()<len(data[float_features])*0.9] #claen the 70% NA
float_data.index = data[object_features].iloc[:,0]
float_data_columns = float_data.columns #for test data clean data

POI = data[object_features].iloc[:,2]
POI = LabelEncoder().fit_transform(POI)


# 空值補平均值
for i in float_data.columns:
    #float_data[i] = float_data[i].fillna(float_data[i].mean())
    float_data[i] = float_data[i].fillna(float_data[i].median())

'''
# 顯示 GrLivArea 與目標值的散佈圖
import seaborn as sns
import matplotlib.pyplot as plt
for i in float_data.columns:
    sns.regplot(x = float_data[i], y=POI)
    plt.show()
'''

#清除極端值
float_data_cleanoutlier = []
for i in float_data.columns:
    top = np.percentile(float_data[i],95)
    down = np.percentile(float_data[i],5)
    #float_data_cleanoutlier.append(float_data[i].clip(down, top))
    #keep_indexs = (float_data[i]> down) & (float_data[i]< top)
   # float_data_cleanoutlier.append(float_data[i][keep_indexs])
    float_data[i]=float_data[i].clip(down, top)

'''
import seaborn as sns
import matplotlib.pyplot as plt
for i in float_data.columns:
    sns.regplot(x = float_data[i], y=POI)
    plt.show()
'''

#**********************使用變數*********************************
use_feature =  ['bonus', 'exercised_stock_options', 'expenses', #, 'deferred_income'
       'from_messages', #, 'from_poi_to_this_person', 'from_this_person_to_poi'
       'long_term_incentive', 'other', 'salary', #, 'restricted_stock'
       'to_messages', #'shared_receipt_with_poi',, 'total_payments'
       'total_stock_value']

message_dff = float_data['from_poi_to_this_person']-float_data['from_this_person_to_poi']
message_from_to = float_data['to_messages']-float_data['from_messages']

df = pd.concat([float_data,pd.DataFrame(message_dff),pd.DataFrame(message_from_to)],ignore_index=True,axis = 1)
'''相關係數
df = pd.concat([pd.DataFrame(message_dff),pd.DataFrame(message_from_to)],ignore_index=True,axis = 1)


POI_corr = pd.DataFrame(POI)
POI_corr.index = float_data.index
corr_data = pd.concat([POI_corr,float_data,df],ignore_index=True,axis = 1)
corr_data.columns = ['POI','bonus', 'deferred_income', 'exercised_stock_options', 'expenses',
       'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi',
       'long_term_incentive', 'other', 'restricted_stock', 'salary',
       'shared_receipt_with_poi', 'to_messages', 'total_payments',
       'total_stock_value','message_dff','message_from_to']
corr = corr_data.corr()
corr.columns = ['POI','bonus', 'deferred_income', 'exercised_stock_options', 'expenses',
       'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi',
       'long_term_incentive', 'other', 'restricted_stock', 'salary',
       'shared_receipt_with_poi', 'to_messages', 'total_payments',
       'total_stock_value','message_dff','message_from_to']

corr.index = ['POI','bonus', 'deferred_income', 'exercised_stock_options', 'expenses',
       'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi',
       'long_term_incentive', 'other', 'restricted_stock', 'salary',
       'shared_receipt_with_poi', 'to_messages', 'total_payments',
       'total_stock_value','message_dff','message_from_to']


import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(corr)
plt.show()

df = corr_data.drop(['POI'] , axis=1)
high_list = list(corr[(corr['POI']>0.2)| (corr['POI']<-0.1)].index)
high_list.pop(-1)

df = df[high_list[1:len(high_list)]]
'''
#L1 score

from sklearn.linear_model import Lasso
L1_Reg = Lasso(alpha=0.001)
L1_Reg.fit(MinMaxScaler().fit_transform(df), POI)
L1_Reg.coef_

from itertools import compress
L1_mask = list((L1_Reg.coef_>0.15)|(L1_Reg.coef_<-0.15))
L1_list = list(compress(list(df), list(L1_mask)))
L1_list

df = df[L1_list]

#標準化
#df = StandardScaler().fit_transform(df)
df = MinMaxScaler().fit_transform(df)

#===============================ML=================================
x_train, x_test, y_train, y_test = train_test_split(df, POI, test_size=0.25, random_state=4)

# 建立模型
clf = GradientBoostingClassifier(random_state = 10)

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict_proba(x_test)


auc = metrics.roc_auc_score(y_test, y_pred[:,1])
print("AUC: ", auc)

#===========================CV===================================


if __name__=='__main__': #https://blog.csdn.net/u010004460/article/details/53889234
    # 設定要訓練的超參數組合
    #p_test3 = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750]}
    parameters = {
    "loss":["deviance"],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "n_estimators":[10,100]
    } 
    #tuning = GridSearchCV(estimator =GradientBoostingClassifier(max_depth=4, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10), 
    #        param_grid = p_test3, scoring='accuracy',n_jobs=-1,iid=False, cv=5)
    
    tuning =  GridSearchCV(clf, parameters, scoring="accuracy", verbose=1,iid=False) #, n_jobs=-1

    # 開始搜尋最佳參數
    grid_result = tuning.fit(x_train, y_train)

   
print("Best Accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

clf_bestparam = GradientBoostingClassifier(loss=grid_result.best_params_['loss'],
                                           learning_rate=grid_result.best_params_['learning_rate'],
                                           max_depth=grid_result.best_params_['max_depth'],
                                           n_estimators=grid_result.best_params_['n_estimators'],
                                           max_features=grid_result.best_params_['max_features'],
                                           random_state= 10
                                           )

'''
clf_bestparam = GradientBoostingClassifier(max_depth=4,
                                           min_samples_split=2, 
                                           min_samples_leaf=1, 
                                           subsample=1,
                                           max_features='sqrt',
                                           n_estimators=grid_result.best_params_['n_estimators'],
                                           learning_rate = grid_result.best_params_['learning_rate'])

'''

# 訓練模型
clf_bestparam.fit(x_train, y_train)

# 預測測試集
y_pred = clf_bestparam.predict_proba(x_test)

auc = metrics.roc_auc_score(y_test, y_pred[:,1])
print("AUC: ", auc)

#=========================use test data=================================
test_data = pd.read_csv('data/test_features.csv')
test_data_name = test_data['name']
test_data = test_data[float_data_columns]

# 空值補平均值
for i in test_data.columns:
    #test_data[i] = test_data[i].fillna(test_data[i].mean()).median()
    test_data[i] = test_data[i].fillna(test_data[i].median())



#清除極端值
test_data_cleanoutlier = []
for i in test_data.columns:
    top = np.percentile(test_data[i],95)
    down = np.percentile(test_data[i],5)
    #test_data_cleanoutlier.append(test_data[i].clip(down, top))
    #keep_indexs = (test_data[i]> down) & (test_data[i]< top)
   # test_data_cleanoutlier.append(test_data[i][keep_indexs])
    test_data[i]=test_data[i].clip(down, top)


message_dff = test_data['from_poi_to_this_person']-test_data['from_this_person_to_poi']
message_from_to = test_data['to_messages']-test_data['from_messages']

df = pd.concat([test_data,pd.DataFrame(message_dff),pd.DataFrame(message_from_to )],ignore_index=True,axis = 1)
df = df[L1_list]



#標準化
#df_answer = StandardScaler().fit_transform(df)
df_answer = MinMaxScaler().fit_transform(df)

answer = clf_bestparam.predict_proba(df_answer)[:,1]
clf_bestparam.predict(df_answer)
alldata = {'name' : test_data_name, 'poi':answer}
pd.DataFrame(alldata).to_csv('answer.csv',index = False)
