'''
Created on 2018-4-28

@author: Administrator
'''

import os
import sys

dict_mapping = {1: "dict_zhiwang", 2: "dict_tsinghua", 3: "dict_ntusd", 4: "dict_dalianligong", 5:"dic_extreme"}


def load_dict_by_type(dict_type):
    dict = {};
    os.chdir(sys.path[0]);
    dict_path = os.path.abspath('dictionary/data/'+dict_mapping[dict_type]);
    if dict_type !=4:
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
#
# def load_dalianligong_dict():
#     os.chdir(sys.path[0]);
#     file_path = os.path.abspath('dictionary/data/dict_dalianligong/');
#     dalianligong_dic = read_xlsx_file(file_path, "SenDic.xlsx")
#     return dalianligong_dic;


def load_extreme_dict():
    os.chdir(sys.path[0]);
    file_path = os.path.abspath('dictionary/data/dict_extreme/extreme.txt');
    extreme_dict = __load_scored_dict_from_file__(file_path)
    return extreme_dict;

def load_extent_dict():
    os.chdir(sys.path[0]);
    file_path = os.path.abspath('dictionary/data/dict_common/extent.txt');
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