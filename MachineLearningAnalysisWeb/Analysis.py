import pickle
import itertools
import nltk
import sklearn
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.model_selection import train_test_split
import os
import sys


def bag_of_words(words):
    return dict([(word, True) for word in words])

# 把所有词和双词搭配一起当作特征
def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)

    return bag_of_words(words + bigrams)  # 所有词和（信息量大的）双词搭配一起作为特征


    # 计算整个语料每个词和双词搭配的信息量
def create_word_bigram_scores():
    os.chdir(sys.path[0]);
    pos_review_pkl_path = os.path.abspath('MachineLearningAnalysisWeb/pos_review.pkl')
    neg_review_pkl_path = os.path.abspath('MachineLearningAnalysisWeb/neg_review.pkl')
    posdata = pickle.load(open(pos_review_pkl_path, 'rb'))
    negdata = pickle.load(open(neg_review_pkl_path, 'rb'))

    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    bigram_finder = BigramCollocationFinder.from_words(posWords)
    posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 10000)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 10000)

    pos = posWords + posBigrams  # 词和双词搭配
    neg = negWords + negBigrams

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        word_fd[word] += 1
        cond_word_fd["pos"][word] += 1
    for word in neg:
        word_fd[word] += 1
        cond_word_fd["neg"][word] += 1

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

    # 根据信息量进行倒序排序，选择排名靠前的信息量的词
def find_best_words(word_scores, number):
    # 把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
    best_vals = sorted(word_scores.items(), key=lambda w_s: w_s[1], reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

    # 把选出的词作为特征，也就是信息量丰富的特征
def best_word_features(words, best_words):
    return dict([(word, True) for word in words if word in best_words])

pos_review = []  # 积极数据
neg_review = []  # 消极数据

# 分割数据及赋予类标签
    # 载入数据
def load_data():
    global pos_review, neg_review, review
    pos_review = pickle.load(open('MachineLearningAnalysisWeb/pos_review.pkl', 'rb'))
    neg_review = pickle.load(open('MachineLearningAnalysisWeb/neg_review.pkl', 'rb'))

    # 赋予类标签
        # 积极

def pos_features(feature_extraction_method, best_words):
    posFeatures = []
    for i in pos_review:
        posWords = [feature_extraction_method(i, best_words), 'pos']  # 为积极文本赋予"pos"
        posFeatures.append(posWords)
    return posFeatures

        # 消极

def neg_features(feature_extraction_method, best_words):
    negFeatures = []
    for j in neg_review:
        negWords = [feature_extraction_method(j, best_words), 'neg']  # 为消极文本赋予"neg"
        negFeatures.append(negWords)
    return negFeatures
