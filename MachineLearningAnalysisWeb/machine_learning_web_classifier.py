# python3
# -*- coding: utf-8 -*-
# @Time      2018/5/27 10:48
# @Author    Alina Wang
# @Email     recall52@163.com
# @Software: PyCharm

from MachineLearningAnalysisWeb.Analysis import create_word_bigram_scores, find_best_words, load_data, best_word_features, pos_features, neg_features
import re
import jieba
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

def text_segment(text):
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～]+", "", text)
    list = jieba.cut(text, cut_all=False)
    return set(list)

def predict(classifier, train, test):
    train, train_target = list(zip(*train))

    dict = DictVectorizer()
    encoder = LabelEncoder()

    train = dict.fit_transform(train)
    train_target = encoder.fit_transform(train_target)

    test = dict.transform(test)

    classifier = classifier.fit(train, train_target)
    pred = classifier.predict(test)
    return pred

def get_ml_analysis(text):
    text_word_list = list(text_segment(text))

    word_scores = create_word_bigram_scores()
    best_words = find_best_words(word_scores, 9500)
    load_data()
    posFeatures = pos_features(best_word_features, best_words)  # 使用词和双词搭配作为特征
    negFeatures = neg_features(best_word_features, best_words)
    train = posFeatures + negFeatures

    text_word_list = best_word_features(text_word_list, best_words)
    pred = predict(LogisticRegression(), train, text_word_list)
    return pred

