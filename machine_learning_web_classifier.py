# python3
# -*- coding: utf-8 -*-
# @Time      2018/5/27 10:48
# @Author    Alina Wang
# @Email     recall52@163.com
# @Software: PyCharm

import pickle
from Analysis import create_word_bigram_scores, find_best_words, load_data, best_word_features, pos_features, neg_features, cut_data
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
import numpy


def store_classifier():
    word_scores = create_word_bigram_scores()
    best_words = find_best_words(word_scores, 9500)

    load_data()

    posFeatures = pos_features(best_word_features, best_words)
    negFeatures = neg_features(best_word_features, best_words)

    train, test = cut_data(posFeatures, negFeatures)

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(train)
    pickle.dump(LogisticRegression_classifier, open('classifier.pkl', 'wb'))

# 使用分类器分类，给出概率值
# 把文本变成特征表示
def transfer_text_to_moto():
    moto = pickle.load(open('moto_senti_seg.pkl', 'rb'))  # 载入文本数据

    def extract_features(data):
        feat = []
        word_scores = create_word_bigram_scores()
        best_words = find_best_words(word_scores, 9500)
        for i in data:
            feat.append(best_word_features(i, best_words))
            return feat

    moto_features = extract_features(moto)  # 把文本转化为特征表示的形式
    return moto_features

    # 对文本进行分类，给出概率值
def application():
    clf = pickle.load(open('classifier.pkl', 'rb'))  # 载入分类器

    pred = clf.classify_many(transfer_text_to_moto())  # 该方法是计算分类概率值的
    p_file = open('moto_ml_score.txt', 'w')  # 把结果写入文档
    for i in pred:
        p_file.write(i + '\n')
    p_file.close()

if __name__ == '__main__':
    # store_classifier()
    application()