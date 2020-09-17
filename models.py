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
import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(4,)))
model.summary()
model.compile(optimizer='adam', loss='mse')
model.fit()
