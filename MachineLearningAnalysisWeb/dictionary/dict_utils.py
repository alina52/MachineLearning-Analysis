'''
Created on 2018-4-28

@author: Administrator
'''

import os
import sys
import xlrd

dict_mapping = {1: "dict_zhiwang", 2: "dict_tsinghua", 3: "dict_ntusd", 4: "dict_dalianligong", 5: "dict_extreme"}
all_type_word_dict = {}
extent_dict = {}
deny_word_set = set()
sense_word_kind_set = {"a", "ad", "an", "ag", "al", "d", "dg","n","l",'v','m','z','i','zg','nr'}

def load_dict():
    extent_dict.update(load_extent_dict());
    all_type_word_dict[1] = load_dict_by_type(1);
    all_type_word_dict[2] = load_dict_by_type(2);
    all_type_word_dict[3] = load_dict_by_type(3);
    all_type_word_dict[4] = load_dict_by_type(4);
    __init_deny_word_set__();

def __init_deny_word_set__():
    os.chdir(sys.path[0]);
    deny_word_file_path = os.path.abspath('MachineLearningAnalysisWeb/dictionary/data/dict_common/reversed.txt')
    with open(deny_word_file_path, encoding="utf-8") as f:
        for items in f:
            item = items.strip()
            deny_word_set.add(item)

def load_dict_by_type(dict_type):
    dict = {};
    os.chdir(sys.path[0]);
    dict_path = os.path.abspath('MachineLearningAnalysisWeb/dictionary/data/'+dict_mapping[dict_type]);
    if dict_type != 5:
        for dirs, sub_dirs, files in os.walk(dict_path):
            for file in files:
                score = 0

                if file.startswith('pos'):
                    score = 1
                else:
                    score = -1

                dict.update(__load_words_from_file_with_given_score__(os.path.join(dict_path, file), score))
    else:
        dict = load_extreme_dict()

    return dict

# def read_xlsx_file(path, file_name):
#     book = xlrd.open_workbook(path + file_name)
#     sh = book.sheet_by_name("Sheet1")
#     list = []
#     for i in range(1, sh.nrows):
#         list.append(sh.row_values(i))
#     return list
#
# def load_dalianligong_dict():
#     os.chdir(sys.path[0]);
#     file_path = os.path.abspath('MachineLearningAnalysisWeb/dictionary/data/dict_dalianligong/SenDic.xlsx');
#     dalianligong_dict = read_xlsx_file(file_path)
#     return dalianligong_dict;

def load_extreme_dict():
    os.chdir(sys.path[0]);
    file_path = os.path.abspath('MachineLearningAnalysisWeb/dictionary/data/dict_extreme/extreme.txt');
    extreme_dict = __load_scored_dict_from_file__(file_path)
    return extreme_dict;

def load_extent_dict():
    os.chdir(sys.path[0]);
    file_path = os.path.abspath('MachineLearningAnalysisWeb/dictionary/data/dict_common/extent.txt');
    extent_dict = __load_scored_dict_from_file__(file_path)
    return extent_dict;
    pass

def __load_scored_dict_from_file__(file_path):
    dict = {}
    file_object = open(file_path, encoding="utf-8")
    try:
        for line in file_object:
            data= line.strip().split(',')
            dict[data[0]] = data[1];
    finally:
        file_object.close()

    return dict;

def __load_words_from_file_with_given_score__(file_path, score):
    dict = {};
    file_object = open(file_path, encoding="utf-8")
    try:
        for word in file_object:
            dict[word.strip()] = score
    finally:
        file_object.close()

    return dict;