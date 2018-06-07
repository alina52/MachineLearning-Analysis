# python3
# -*- coding: utf-8 -*-
# @Time      04/05/2018 10:53
# @Author    Alina Wang
# @Email     recall52@163.com
# @Software: PyCharm

import re
import jieba.posseg

def get_jieba_word_list(simple_sentences):
    list = jieba.posseg.cut(simple_sentences)
    return list


def get_simple_sentences_gen(text):
    compound_sentences_gen = __get_compound_sentences_gen__(text)
    simple_sentences_gen = __get_simple_sentences_gen__(compound_sentences_gen)
    return simple_sentences_gen

def __get_compound_sentences_gen__(str):
    str = re.split(r'[。？！；.?!;“”．]', str)
    for s in str:
        if s != ''and s!= ' ':
            yield s.strip()

def __get_simple_sentences_gen__(compound_sentences):
    for s in compound_sentences:
        s = re.split(r'[,，、（）\s]',s)
        for str in s:
            if str != ' ' and str != '':
                yield str.strip()