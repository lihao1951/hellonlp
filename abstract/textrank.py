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
    sentences = []
    cont_split = re.split('[\n\.。！？\!\?;；]', cont)
    for s in cont_split:
        if s != '':
            sentences.append(s)
    return sentences


def softmax(array):
    shape = array.shape
    values = array.sum()
    for i in range(shape[0]):
        array[i][0] = array[i][0] / values
    return array


def pagerank(mat, alpha, epsilon=1e-4):
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
        vector = np.zeros(shape=200, dtype=np.float)
        exist = True
        try:
            vector = self._embeddings.word_vec(word)
        except Exception as e:
            print('get word(%s) is wrong' % word)
            exist = False
        return vector, exist

    def _get_sentence_embedding(self, sentence):
        vector = np.zeros(shape=200, dtype=np.float)
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
        scores = pagerank(simi_mat, 0.85)
        index_scores = np.squeeze(np.argsort(scores, axis=0)).tolist()
        result = []
        for i in range(self.params.get('topK')):
            ix = index_scores[-1 - i]
            result.append(sentences[ix])
        return result


textrank = TextRank(topK=2)
cont = '据福建日报消息，9月15日，福建省十三届人大四次会议采用无记名投票方式，补选王宁为福建省人民政府省长，李仰哲为福建省监察委员会主任。王宁，男，汉族，1961年4月生，湖南湘乡人（辽宁沈阳出生），1983年6月加入中国共产党，1983年8月参加工作，大学学历，高级工程师。现任中共十九届中央候补委员，省委副书记，省政府省长、党组书记。' \
       '1999年6月起任建设部建筑管理司（建筑市场管理司）助理巡视员（副司级，其间：1999年6月至2002年6月挂职任新疆自治区建设厅副厅长、党组成员）；2002年10月起任建设部建筑市场管理司副司长；2005年10月起任住房和城乡建设部稽查办公室副主任（主持工作，正司长级）；2008年8月起任住房和城乡建设部人事司司长；2013年6月起任住房和城乡' \
       '建设部副部长、党组成员；2015年12月起任福建省委常委、组织部部长，2016年4月兼任省委党校校长；2017年6月起任省委常委，福州市委书记兼福州新区党工委书记；2018年5月起任省委副书记，福州市委书记兼福州新区党工委书记；2020年6月起任省委副书记，省政府党组书记，福州市委书记兼福州新区党工委书记；'
s = textrank.extract_abstract(cont)
print(s)
