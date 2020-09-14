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
import jieba


class TextRank(object):
    def __init__(self, **kwargs):
        # 获取当前目录路径
        current_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        self.params = {
            'stopwords': os.path.join(current_path, 'stopwords.txt'),
            'userdicts': None,
            'embeddings': os.path.join(current_path, 'embeddings.txt'),
            'vocab': os.path.join(current_path, 'vocab.txt'),
            'topK': 3
        }
        # 若指定了新值，则用新值
        for key in kwargs.keys():
            value = kwargs.get(key)
            self.params[key] = value
        # 导入用户自定义词典
        if self.params.get('userdicts', None):
            jieba.load_userdict(self.params.get('userdicts'))
        # 导入通用词典
        self._load_vocab()
        # 导入embeddings
        self._load_embeddings()

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
        from gensim.models import KeyedVectors
        self._embeddings = KeyedVectors.load_word2vec_format(self.params.get('embeddings'), binary=False)
        self._embeddings_size = self._embeddings.wv.syn0[0].shape[0]

    def _segment(self, content):
        words = jieba.cut(content)
        for word in words:
            print(word)

    def extract_sentence(self):
        ...


textrank = TextRank()
