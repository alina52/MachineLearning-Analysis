import pickle
import itertools
import nltk
import sklearn
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures



from sklearn.model_selection import train_test_split



# 把所有词当作特征
def bag_of_words(words):
    return dict([(word, True) for word in words])

# 把双词搭配当作特征
def bigram(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)  # 把文本变成双词搭配的形式
    bigrams = bigram_finder.nbest(score_fn, n)  # 使用了卡方统计的方法，选择排名前1000的双词
    return bag_of_words(bigrams)

# 把所有词和双词搭配一起当作特征
def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)

    return bag_of_words(words + bigrams)  # 所有词和（信息量大的）双词搭配一起作为特征

# 特征选择
    # 计算出整个语料里面每个词的信息量
def create_word_scores():
    posWords = pickle.load(open('pos_review.pkl', 'rb'))
    negWords = pickle.load(open('neg_review.pkl', 'rb'))

    posWords = list(itertools.chain(*posWords))  # 把多维数组解链成一维数组
    negWords = list(itertools.chain(*negWords))  # 同理

    word_fd = FreqDist()  # 可统计所有词的词频
    cond_word_fd = ConditionalFreqDist()  # 可统计积极文本中的词频和消极文本中的词
    for word in posWords:
        word_fd[word] += 1
        cond_word_fd["pos"][word] += 1
    for word in negWords:
        word_fd[word] += 1
        cond_word_fd["neg"][word] += 1

    pos_word_count = cond_word_fd['pos'].N()  # 积极词的数量
    neg_word_count = cond_word_fd['neg'].N()  # 消极词的数量
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.items():
        # 计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        # 计算消极词的卡方统计量，这里也可以计算互信息等其它统计量
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        # 一个词的信息量等于积极卡方统计量加上消极卡方统计量
        word_scores[word] = pos_score + neg_score
    # 包括了每个词以及信息量
    return word_scores

def create_bigram_scores():
    posdata = pickle.load(open('pos_review.pkl', 'rb'))
    negdata = pickle.load(open('neg_review.pkl', 'rb'))

    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    bigram_finder = BigramCollocationFinder.from_words(posWords)
    posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 10000)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 10000)

    word_fd = FreqDist()  # 可统计所有词的词频
    cond_word_fd = ConditionalFreqDist()  # 可统计积极文本中的词频和消极文本中的词
    for word in posBigrams:
        word_fd[word] += 1
        cond_word_fd["pos"][word] += 1
    for word in negBigrams:
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

    # 计算整个语料每个词和双词搭配的信息量
def create_word_bigram_scores():
    posdata = pickle.load(open('pos_review.pkl', 'rb'))
    negdata = pickle.load(open('neg_review.pkl', 'rb'))

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

# def best_word_features(words, best_words):
#     ret = dict()
#     for word in words:
#         for bestWord in best_words:
#             if word == bestWord[0]:
#                 if bestWord[1] in words:
#                     ret[bestWord] = True
#             elif word == bestWord[1]:
#                 if bestWord[0] in words:
#                     ret[bestWord] = True
#     return ret
#     return dict([(word, True) for word in words if word in best_words])

pos_review = []  # 积极数据
neg_review = []  # 消极数据

# 分割数据及赋予类标签
    # 载入数据
def load_data():
    global pos_review, neg_review, review
    pos_review = pickle.load(open('pos_review.pkl', 'rb'))
    neg_review = pickle.load(open('neg_review.pkl', 'rb'))

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


    # 把特征化之后的数据数据分割为开发集和测试集

def cut_data(posFeatures, negFeatures):
    pos_train, pos_test = train_test_split(posFeatures, test_size=0.2, random_state=25)
    neg_train, neg_test = train_test_split(negFeatures, test_size=0.2, random_state=25)
    train = pos_train + neg_train
    test = pos_test + neg_test
    return train, test





