'''

@author: Administrator
'''
from django.http import HttpResponse
from django.shortcuts import render_to_response
from MachineLearningAnalysisWeb.dictionary.dict_utils import *
from django.shortcuts import render
from django.views.decorators import csrf
from MachineLearningAnalysisWeb.segementation_utils import *
from functools import reduce

def text_analysis_form(request):
    
    return render_to_response('text_analysis_form.html')

def text_analysis(request):
    ctx ={}
    dict = all_type_word_dict[1];
    if request.POST:
        dict_type_arg = request.POST['dict_type']
        dict = all_type_word_dict[int(dict_type_arg)];
        text_doc = request.POST['text']
        score = __caculate_text_score__(text_doc, dict)
        ctx['result'] = score
    return render(request, 'text_analysis_result.html', ctx)

def __fill_with_word_info__(word, k, s, p = None):
    word_info = {};
    word_info['n'] = word                   #word
    word_info['k'] = k                      #kind
    word_info['s'] = s                      #score
    word_info['p'] = p                      #property
    return word_info

def __getScore__(dict, word):
    return dict.get(word, 0)

def __get_word_detail_info__(word, word_kind, word_dict):
    word_info = {};
    if word in deny_word_set:
        word_info = __fill_with_word_info__(word, 'deny', None, None)
    elif word_kind in sense_word_kind_set:
        score = __getScore__(extent_dict, word)
        if score != 0:
            word_info = __fill_with_word_info__(word, word_kind, score, None)
        else:
            score = __getScore__(word_dict, word)
            if score >0:
                word_info = __fill_with_word_info__(word, word_kind, score, 'pos')
            elif score < 0:
                word_info = __fill_with_word_info__(word, word_kind, score, 'neg')
            else:
                word_info = __fill_with_word_info__(word, word_kind, score)
    else:
        word_info = __fill_with_word_info__(word, word_kind, 0)

    return word_info

def __caculate_score_of_simple_sentence__(stack = [], ExtInNoAndSen = False):
    if ExtInNoAndSen:
        return reduce(lambda item1, item2: item1 * item2, stack) * -0.5
    else:
        return reduce(lambda item1, item2: item1 * item2, stack)
    
def get_simple_sentence_score(word_list=[{}]):
    if len(word_list) > 0:
        stack = []
        copystack = []
        GroupScore = 0
        NoWordFirst = False
        HaveSenWord = False
        ExtInNoAndSen = False
        if word_list[0].get("k") == 'no':
            NoWordFirst = True
            copystack.append('no')

        for item in word_list:
            if item.get('p') == 'pos' or item.get('p') == 'neg':
                HaveSenWord = True
                stack.append(item.get('s'))
            elif item.get('p') == 'ext':
                stack.append(item.get('s'))
                if NoWordFirst == True and HaveSenWord == False:
                    ExtInNoAndSen = True
            elif item.get('k') == 'c':
                pass
            elif item.get('k') == 'no':
                stack.append(-1)
        copystack.append(stack)
        if HaveSenWord:
            GroupScore = __caculate_score_of_simple_sentence__(stack, ExtInNoAndSen)
        return GroupScore, copystack
    return 0, None

def __caculate_text_score__(text_doc, word_dict):
    simple_sentences_gen = get_simple_sentences_gen(text_doc)
    _score_sum_ = 0;
    for simple_sentence in simple_sentences_gen:
        jieba_word_list = get_jieba_word_list(simple_sentence)
        word_info_list = []
        for word, kind in jieba_word_list:
            word_info = __get_word_detail_info__(word, kind, word_dict)
            word_info_list.append(word_info)

        score, stack = get_simple_sentence_score(word_info_list)
        _score_sum_ += score
    return _score_sum_
