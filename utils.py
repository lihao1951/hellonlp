#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@Description
    工具包
@Author LiHao
@Date 2020/9/17
"""
import re
import numpy as np


def cosine(rep1, rep2):
    """
    修正后的余弦相似度
    :param rep1: numpy.array([*,*,*,...])
    :param rep2: numpy.array([*,*,*,...])
    :return: float
    """
    assert rep1.shape == rep2.shape
    cos = np.sum(np.multiply(rep1, rep2)) / (np.linalg.norm(rep1) * np.linalg.norm(rep2))
    return cos


def split_sentence(cont):
    """
    切分句子
    :param cont:
    :return:
    """
    sentences = []
    cont_split = re.split('[\n\.。！？\!\?;；]', cont)
    for s in cont_split:
        if s != '':
            sentences.append(s)
    return sentences


def softmax(array):
    """
    softmax
    :param array:
    :return:
    """
    shape = array.shape
    values = array.sum()
    for i in range(shape[0]):
        array[i][0] = array[i][0] / values
    return array


def pagerank(mat, alpha, epsilon=1e-4):
    """
    pagerank计算
    :param mat:
    :param alpha:
    :param epsilon:
    :return:
    """
    mat_shape = mat.shape
    assert mat_shape[0] == mat_shape[1]
    pr = np.divide(np.ones((mat_shape[0], 1), dtype=np.float), mat_shape[0])
    while True:
        new_pr = softmax(alpha * np.dot(mat, pr) + (1 - alpha) * pr)
        cha = new_pr - pr
        if np.sum(np.abs(cha)) <= epsilon:
            break
        pr = new_pr
    return pr
