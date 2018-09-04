# -*- coding: utf-8 -*-
"""
Created on Mon Oct 02 10:14:45 2017

@author: Dr Alina
"""
x = 'D:\COURSE programming with python for Data Science\DAT210x-master\Module3\Datasets\wheat.data'
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
df=pd.read_csv(x)
df = df.drop(['id'], axis= 1)


plt.imshow(df.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(df.columns))]
plt.xticks(tick_marks, df.columns, rotation='vertical')
plt.yticks(tick_marks, df.columns)

plt.show()
