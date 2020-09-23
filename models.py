#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@Description
    基础模型
@Author LiHao
@Date 2020/9/17
"""
import numpy as np
import os
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

# 导入数据
data, target = load_boston(True)
cols = data.shape[-1]
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
print('train size:%s and test size:%s' % (x_train.shape[0], x_test.shape[0]))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(cols,)))
model.summary()
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=500, batch_size=64)
print("ecaluate")
model.evaluate(x_test, y_test, verbose=1)
