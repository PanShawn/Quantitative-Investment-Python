# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 20:55:57 2022

@author: dell
"""

import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

nfp_basic = pd.read_excel('NFP Statistics.xlsx', sheet_name='Basic')
nfp_est = pd.read_excel('NFP Statistics.xlsx', sheet_name='Estimation')   
nfp_ecos = pd.read_excel('NFP Statistics.xlsx', sheet_name='Economist')

# 提取特征值
nfp = pd.DataFrame()
nfp['Date'] = pd.to_datetime(nfp_basic['Release Date'])
nfp['Actual'] = nfp_basic['Actual']
nfp['Avg Est'] = nfp_est['Average Estimate'] 
nfp['SD'] = nfp_est['Standard Deviation']
# 自相关特征值
nfp['P1'] = nfp['Actual'].shift(1)
nfp['P2'] = nfp['Actual'].shift(2)

# 宏观经济指标：CPI\PPI\PMI\Retail Sales
# 已公布：注意公布时间
df_macro = pd.read_excel('Macro.xlsx')
nfp['CPIYOY'] = df_macro['CPI YOY Index'].shift(1).values
nfp['PPIYOY'] = df_macro['PPI YOY Index'].shift(1).values
nfp['PPI XYOY'] = df_macro['PPI XYOY Index'].shift(1).values
nfp['ISMPMI'] = df_macro['NAPMPMI Index'].values
nfp['RSTAMOM'] = df_macro['RSTAMOM Index'].shift(1).values

#nfp.dropna(axis = 0, inplace = True)


# 设置日期为索引
nfp.set_index('Date', drop = True, inplace = True)

# 添加特征值：前十名经济学家预测均值
nfp_ecos.dropna(subset = ['Rank'], axis = 0, inplace = True)
nfp_ecos['As of'] = pd.to_datetime(nfp_ecos['As of'])

nfp['ECOS'] = None

count = 0
last_dt = nfp.index[0] - datetime.timedelta(days=30)

# 经济学家预测中值
for dt in nfp.index:
    
    #nfp['ECOS'].loc[dt] = nfp_ecos[(nfp_ecos['As of'] <= dt) & (nfp_ecos['As of'] > last_dt)]['Estimate'].mean()
    nfp['ECOS'].loc[dt] = nfp_ecos[(nfp_ecos['As of'] <= dt) & (nfp_ecos['As of'] > last_dt)]['Estimate'].median()
    count += len(nfp_ecos.iloc[count:][nfp_ecos['As of'] < dt]['Estimate'])
    
    last_dt = dt
    
# 数据清洗：剔除异动数据:绝对值大于一个标准差
nfp_ = nfp.drop(nfp[abs(nfp['Actual']) > nfp['Actual'].std()].index)

nfp_.dropna(axis = 0, inplace = True)

X_ = nfp_[['Avg Est', 'SD', 'ECOS', 'CPIYOY', 'PPIYOY', 'PPI XYOY', 'ISMPMI',
       'RSTAMOM', 'P1', 'P2']]
y_ = nfp_['Actual']


import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers) # utilize the LSTM model in torch.nn 
        self.forwardCalculation = nn.Linear(hidden_size, output_size)
 
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x

if __name__ == '__main__':
    # create database
    '''
    data_len = 200
    t = np.linspace(0, 12*np.pi, data_len)
    sin_t = np.sin(t)
    cos_t = np.cos(t)

    dataset = np.zeros((data_len, 2))
    dataset[:,0] = sin_t
    dataset[:,1] = cos_t
    dataset = dataset.astype('float32')
    '''
    nfp_ = nfp_.head(200)
    
    data_len = len(nfp_)
    t = nfp_.index
    dataset = np.zeros((data_len, 2))
    dataset[:,0] = nfp_['ECOS']
    dataset[:,1] = nfp_['Actual']
    dataset = dataset.astype('float32')
    
    # plot part of the original dataset
    plt.figure()
    plt.plot(t[0:60], dataset[0:60,0], label='ECOS')
    plt.plot(t[0:60], dataset[0:60,1], label = 'NFP')
    #plt.plot([2.5, 2.5], [-1.3, 0.55], 'r--', label='t = 2.5') # t = 2.5
    #plt.plot([6.8, 6.8], [-1.3, 0.85], 'm--', label='t = 6.8') # t = 6.8
    plt.xlabel('t')
    #plt.ylim(-1000, 1000)
    plt.ylabel('ecos and nfp')
    plt.legend(loc='upper right')

    # choose dataset for training and testing
    train_data_ratio = 0.5 # Choose 80% of the data for testing
    train_data_len = int(data_len*train_data_ratio)
    train_x = dataset[:train_data_len, 0]
    train_y = dataset[:train_data_len, 1]
    INPUT_FEATURES_NUM = 1
    OUTPUT_FEATURES_NUM = 1
    t_for_training = t[:train_data_len]

    # test_x = train_x
    # test_y = train_y
    test_x = dataset[train_data_len:, 0]
    test_y = dataset[train_data_len:, 1]
    t_for_testing = t[train_data_len:]

    # ----------------- train -------------------
    train_x_tensor = train_x.reshape(-1, 5, INPUT_FEATURES_NUM) # set batch size to 5
    train_y_tensor = train_y.reshape(-1, 5, OUTPUT_FEATURES_NUM) # set batch size to 5
 
    # transfer data to pytorch tensor
    train_x_tensor = torch.from_numpy(train_x_tensor)
    train_y_tensor = torch.from_numpy(train_y_tensor)
    # test_x_tensor = torch.from_numpy(test_x)
 
    lstm_model = LstmRNN(INPUT_FEATURES_NUM, 16, output_size=OUTPUT_FEATURES_NUM, num_layers=1) # 16 hidden units
    print('LSTM model:', lstm_model)
    print('model.parameters:', lstm_model.parameters)
 
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)
 
    max_epochs = 100000
    for epoch in range(max_epochs):
        output = lstm_model(train_x_tensor)
        loss = loss_function(output, train_y_tensor)
 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
 
        if loss.item() < 1e-4:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch+1) % 100 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch+1, max_epochs, loss.item()))
 
    # prediction on training dataset
    predictive_y_for_training = lstm_model(train_x_tensor)
    predictive_y_for_training = predictive_y_for_training.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # torch.save(lstm_model.state_dict(), 'model_params.pkl') # save model parameters to files
 
    # ----------------- test -------------------
    # lstm_model.load_state_dict(torch.load('model_params.pkl'))  # load model parameters from files
    lstm_model = lstm_model.eval() # switch to testing model

    # prediction on test dataset
    test_x_tensor = test_x.reshape(-1, 5, INPUT_FEATURES_NUM) # set batch size to 5, the same value with the training set
    test_x_tensor = torch.from_numpy(test_x_tensor)
 
    predictive_y_for_testing = lstm_model(test_x_tensor)
    predictive_y_for_testing = predictive_y_for_testing.view(-1, OUTPUT_FEATURES_NUM).data.numpy()
    
    
    # 评估模型
    from sklearn import metrics

    MAE = metrics.mean_absolute_error(test_y, predictive_y_for_testing)
    MSE = metrics.mean_squared_error(test_y, predictive_y_for_testing)
    RMSE = np.sqrt(metrics.mean_squared_error(test_y, predictive_y_for_testing))

    MAE = metrics.mean_absolute_error(train_y, predictive_y_for_training)
    MSE = metrics.mean_squared_error(train_y, predictive_y_for_training)
    RMSE = np.sqrt(metrics.mean_squared_error(train_y, predictive_y_for_training))
    
    print("MAE:" + str(round(MAE, 4)))
    print("MSE:" + str(round(MSE, 4)))
    print("RMSE:" + str(round(RMSE,4)))

    r2 = round(r2_score(train_y, predictive_y_for_training), 2)
    print("R2:" + str(r2))
    # R2
    from sklearn.metrics import r2_score
    r2 = round(r2_score(test_y, predictive_y_for_testing), 2)
    print("R2:" + str(r2))
    
    
 
    # ----------------- plot -------------------
    plt.figure()
    plt.plot(t_for_training, train_x, 'limegreen', label='ecos_trn')
    plt.plot(t_for_training, train_y, 'royalblue', label='ref_nfp_trn')
    plt.plot(t_for_training, predictive_y_for_training, 'm--', label='pre_nfp_trn')

    plt.plot(t_for_testing, test_x, 'mediumspringgreen', label='ecos_tst')
    plt.plot(t_for_testing, test_y, 'cornflowerblue', label='ref_nfp_tst')
    plt.plot(t_for_testing, predictive_y_for_testing, 'm--', label='pre_nfp_tst')

    plt.plot([t[train_data_len], t[train_data_len]], [-500, 500], 'r--', label='separation line') # separation line

    plt.xlabel('t')
    plt.ylabel('ecos and nfp')
    plt.xlim(t[0], t[-1])
    #plt.ylim(-1.2, 4)
    plt.legend(loc='lower right')
    plt.text(2005, -400, "train", size = 15, alpha = 1.0)
    plt.text(2007, -400, "test", size = 15, alpha = 1.0)

    plt.show()










