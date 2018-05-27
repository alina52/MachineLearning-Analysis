# python3
# -*- coding: utf-8 -*-
# @Time      04/05/2018 10:53
# @Author    Alina Wang
# @Email     recall52@163.com
# @Software: PyCharm

import pickle
import matplotlib.pyplot as plt
import numpy as np
import itertools
from MachineLearningAnalysisWeb.Analysis import create_word_scores, create_bigram_scores, create_word_bigram_scores, find_best_words, load_data, best_word_features, pos_features, neg_features, cut_data
from MachineLearningAnalysisWeb.text_analysis_main import countResult
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
import sys
sys.setrecursionlimit(2000)

# 检验不同分类器和不同的特征选择的结果
# 使用训练集训练分类器
# 用分类器对开发测试集里面的数据进行分类，给出分类预测的标签
# 对比分类标签和人工标注的差异，计算出准确度


def predict(classifier, train, test):
    train, train_target = list(zip(*train))
    test, test_target = list(zip(*test))

    dict = DictVectorizer()
    encoder = LabelEncoder()

    train = dict.fit_transform(train)
    train_target = encoder.fit_transform(train_target)

    test = dict.transform(test)
    test_target = encoder.transform(test_target)

    # pred = classifier.classify_many(test)  # 对测试集的数据进行分类，给出预测的标签
    classifier = classifier.fit(train, train_target)
    pred = classifier.predict(test)
    # return accuracy_score(test_target, pred)  # 对比分类预测结果和人工标注的正确结果，给出分类器准确度
    return test_target, pred

def accuracy(test_target, pred):
    print('Accuracy is %.3f' % accuracy_score(test_target, pred))  # 对比分类预测结果和人工标注的正确结果，给出分类器准确度


def confustion_matrix(test_target, pred):
    title = "Confusion Matrix"
    targets = list(set(test_target))
    cm = confusion_matrix(test_target, pred, labels=targets)
    np.set_printoptions(precision=2)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap="Greys")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(targets))
    plt.xticks(tick_marks, targets, rotation=45)
    plt.yticks(tick_marks, targets)
    fmt = '.3f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def precision_recall_curve(test_target, pred):
    precision, recall, th = precision_recall_curve(test_target, pred)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')
    plt.show()

def machine_learning_evaluate(test_target, pred):
    targets = list(set(test_target))
    precision, recall, f, support = precision_recall_fscore_support(test_target, pred, average="binary")
    print("Machine Learning`s Precsion is %.3f" % precision + "\n" + "Recall is %.3f" % recall + "\n" + "f-score is %.3f" % f)
    confustion_matrix(test_target, pred)
    # precision_recall_curve(test_target,pred)

def dictionary_evaluate(tp, tn, fp, fn):
    tp_list = [1] * tp
    fp_list = [0] * fp
    tn_list = [0] * tn
    fn_list = [1] * fn

    test_target = [1] * (tp + fp) + [0] * (tn + fn)
    pred = tp_list + fp_list + tn_list + fn_list
    precision, recall, f, support = precision_recall_fscore_support(test_target, pred, average="binary")
    print("Based on dictionary precision is %f3" % precision + "\n" + "Recall is %3f" % recall + "\n" + "f-score is %3f" % f)
    confustion_matrix(test_target, pred)

def compare_machine_learning_classifier():
    # word_scores_1 = create_word_scores()
    word_scores_2 = create_bigram_scores()
    # word_scores_3 = create_word_bigram_scores()
    for k in np.arange(1000, 11000, 1000):
        # best_words_1 = find_best_words(word_scores_1, k)
        best_words_2 = find_best_words(word_scores_2,k)
        # best_words_3 = find_best_words(word_scores_3, 8500)
        load_data()
        posFeatures = pos_features(best_word_features, best_words_2)  # 使用词和双词搭配作为特征
        negFeatures = neg_features(best_word_features, best_words_2)
        train, test = cut_data(posFeatures, negFeatures)
        test_target, pred = predict(NuSVC(), train, test)

        print(k)
        accuracy(test_target, pred)

    # machine_learning_evaluate(test_target, pred)

        # print('BernoulliNB`s accuracy is %f' % accuracy(test_target, pred))
        # print('MultinomialNB`s accuracy is %f' % accuracy(test_target, pred))
        # print('LogisticRegression`s accuracy is %f' % accuracy(test_target, pred))
        # print('SVC`s accuracy is %f' % accuracy(test_target, pred))
        # print('LinearSVC`s accuracy is %f' % accuracy(test_target, pred))
        # print('NuSVC`s accuracy is %f' % accuracy(test_target, pred))

def compare_dictionary_classifier():
    processed_pos_file_count, processed_neg_file_count, neg_score_for_pos_input_count, pos_score_for_neg_input_count = countResult()
    tp = processed_pos_file_count - neg_score_for_pos_input_count
    tn = processed_neg_file_count - pos_score_for_neg_input_count
    fp = neg_score_for_pos_input_count
    fn = pos_score_for_neg_input_count

    dictionary_evaluate(tp, tn, fp, fn)





# def store_classifier():
#     word_scores = create_word_bigram_scores()
#     best_words = find_best_words(word_scores, 1400)
#
#     posFeatures = pos_features(best_word_features)
#     negFeatures = neg_features(best_word_features)
#
#     trainSet = posFeatures + negFeatures
#
#     MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
#     MultinomialNB_classifier.train(trainSet)
#     pickle.dump(MultinomialNB_classifier, open('classifier.pkl', 'wb'))
#
# # 使用分类器分类，给出概率值
# # 把文本变成特征表示
# def transfer_text_to_moto():
#     moto = pickle.load(open('moto_senti_seg.pkl', 'rb'))  # 载入文本数据
#
#     def extract_features(data):
#         feat = []
#         for i in data:
#             feat.append(best_word_features(i))
#         return feat
#
#     moto_features = extract_features(moto)  # 把文本转化为特征表示的形式
#     return moto_features
#
#     # 对文本进行分类，给出概率值
# def application():
#     clf = pickle.load(open('classifier.pkl', 'rb'))  # 载入分类器
#
#     pred = clf.batch_prob_classify(transfer_text_to_moto())  # 该方法是计算分类概率值的
#     p_file = open('moto_ml_score.txt', 'w')  # 把结果写入文档
#     for i in pred:
#         p_file.write(str(i.prob('pos')) + ' ' + str(i.prob('neg')) + '\n')
#     p_file.close()

if __name__ == '__main__':
    # compare_machine_learning_classifier()
    compare_dictionary_classifier()
