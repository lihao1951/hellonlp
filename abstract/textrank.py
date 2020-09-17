#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@Description
    利用textrank方法来生成摘要
@Author LiHao
@Date 2020/9/9
"""
import os
import re
import sys

from gensim.models import KeyedVectors
import jieba
from utils import *
import numpy as np


class TextRank(object):
    def __init__(self, **kwargs):
        # 获取当前目录路径
        current_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        self.params = {
            'stopwords': os.path.join(current_path, 'stopwords.txt'),
            'userdicts': None,
            'embeddings': os.path.join(current_path, 'embeddings.txt'),
            'vocab': os.path.join(current_path, 'vocab.txt'),
            'topK': 3,
            'alpha': 0.5
        }
        # 若指定了新值，则用新值
        for key in kwargs.keys():
            value = kwargs.get(key)
            if key in self.params.keys():
                self.params[key] = value
            else:
                print('%s=%s is not in self params' % (key, value))
        # 导入用户自定义词典
        if self.params.get('userdicts'):
            jieba.load_userdict(self.params.get('userdicts'))
        # 导入通用词典
        self._load_vocab()
        # 导入embeddings
        self._load_embeddings()
        # 导入stopwords
        self._load_stopwords()

    def _load_stopwords(self):
        self._stopwords = {}
        with open(self.params.get('stopwords'), 'r', encoding='utf-8') as fv:
            for ix, line in enumerate(fv.readlines()):
                self._stopwords[line.strip()] = ix

    def _load_vocab(self):
        self._vocab = {}
        self._inv_vocab = {}
        self._vocab_weight = {}
        with open(self.params.get('vocab'), 'r', encoding='utf-8') as fv:
            for ix, line in enumerate(fv.readlines()):
                word, weight = line.strip().split(' ')
                self._vocab[word] = ix
                self._inv_vocab[ix] = word
                self._vocab_weight[word] = weight

    def _load_embeddings(self):
        self._embeddings = KeyedVectors.load_word2vec_format(self.params.get('embeddings'), binary=False)
        self._embeddings_size = self._embeddings.wv.syn0[0].shape[0]

    def _segment(self, content):
        """
        切分语句
        :param content:
        :return:
        """
        tokens = []
        words = jieba.cut(content)
        for word in words:
            if (word in self._vocab.keys()) and (word not in self._stopwords.keys()):
                tokens.append(word)
        return tokens

    def _get_word_embedding(self, word):
        vector = np.zeros(shape=self._embeddings_size, dtype=np.float)
        exist = True
        try:
            vector = self._embeddings.word_vec(word)
        except Exception as e:
            print('get word(%s) is wrong' % word)
            exist = False
        return vector, exist

    def _get_sentence_embedding(self, sentence):
        vector = np.zeros(shape=self._embeddings_size, dtype=np.float)
        count = 1e-5
        for word in self._segment(sentence):
            word_vec, exist = self._get_word_embedding(word)
            if exist:
                count += 1.0
                vector = np.add(vector, word_vec)
        return np.divide(vector, count)

    def similarity_between_sentences(self, s1, s2):
        v1 = self._get_sentence_embedding(s1)
        v2 = self._get_sentence_embedding(s2)
        return cosine(v1, v2)

    def extract_abstract(self, cont):
        sentences = split_sentence(cont)
        length = len(sentences)
        if length < self.params.get('topK'):
            return cont
        vector_sentences = []
        # 获取每个句子的句子向量
        for s in sentences:
            vector_sentences.append(self._get_sentence_embedding(s))
        # 初始化句子相似度矩阵
        simi_mat = np.zeros(shape=(length, length), dtype=np.float)
        for i in range(length):
            for j in range(length):
                simi_mat[i][j] = cosine(vector_sentences[i], vector_sentences[j])
        scores = pagerank(simi_mat, self.params.get('alpha'))
        index_scores = np.squeeze(np.argsort(scores, axis=0)).tolist()
        result = []
        for i in range(self.params.get('topK')):
            ix = index_scores[-1 - i]
            result.append(sentences[ix])
        return result
