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
            'topK': 3,
            'alpha': 0.5
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


textrank = TextRank(topK=2,alpha=0.9)
cont = '中方已就昨天举行的中德欧领导人会晤发布了详细的新闻稿。关于人权问题，习近平主席强调，世界上没有放之四海而皆准的人权发展道路，人权保障没有最好，只有更好。各国首先应该做好自己的事情。相信欧方能够解决好自身存在的人权问题。中方不接受人权“教师爷”，反对搞“双重标准”。中方愿同欧方本着相互尊重的原则加强交流，共同进步。会上还讨论了欧盟内部存在的人权问题，如难民问题久拖不决、人道主义危机屡屡上演，一些欧盟成员国种族主义、极端主义、少数族裔问题抬头，反犹太、反穆斯林、反黑人等言论和恶性事件频频发生等等。欧方坦承自身存在的问题，希望同中方本着平等和尊重的原则开展对话，增进相互了解，妥善处理差异和分歧。习近平主席还阐明了中方在涉港、涉疆问题上的原则立场，指出涉港、涉疆问题的实质是维护中国国家主权、安全和统一，保护各族人民安居乐业的权利。中方坚决反对任何人、任何势力在中国制造不稳定、分裂和动乱，坚决反对任何国家干涉中国内政。在涉疆问题上，我们一直欢迎包括欧方在内的各国朋友去新疆走一走、看一看，去实地了解新疆的真实情况，而不是道听途说，偏信那些刻意编造的谎言。欧盟及成员国驻华使节提出希望访问新疆，中方已经同意并愿作出安排。现在球在欧方一边。同时我要说明一点，我们反对有罪推定式的调查。'
s = textrank.extract_abstract(cont)
print(s)
