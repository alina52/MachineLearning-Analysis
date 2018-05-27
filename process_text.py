import pickle
import jieba
import os
import re

pos_words = []
neg_words = []


FindPath = 'testData/pos/'
FileNames = os.listdir(FindPath)

for file_name in FileNames:
    full_file_name = os.path.join(FindPath, file_name)
    if 'utf8' in full_file_name:
        with open(full_file_name, 'r', encoding='utf-8') as pos_f:
            pos_text = pos_f.read()
            pos_text = ''.join(pos_text.split())
            # pos_text = re.sub(string.punctuation, "", pos_text)
            pos_text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～-]+", "", pos_text)
            pos_list = jieba.cut(pos_text, cut_all=False)
            pos_words.append(list(pos_list))


FindPath = 'testData/neg/'
FileNames = os.listdir(FindPath)
for file_name in FileNames:
    full_file_name = os.path.join(FindPath, file_name)
    if 'utf8' in full_file_name:
        with open(full_file_name, 'r', encoding='utf-8') as neg_f:
            neg_text = neg_f.read()
            neg_text = ''.join(neg_text.split())
            # neg_text = re.sub(string.punctuation, "", neg_text)
            neg_text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～]+", "", neg_text)
            neg_list = jieba.cut(neg_text, cut_all=False)
            neg_words.append(list(neg_list))

FindPath = 'testData/machine-learning-test/'
FileNames = os.listdir(FindPath)
for file_name in FileNames:
    full_file_name = os.path.join(FindPath, file_name)
    if 'utf8' in full_file_name:
        with open(full_file_name, 'r', encoding='utf-8') as neg_f:
            neg_text = neg_f.read()
            neg_text = ''.join(neg_text.split())
            # neg_text = re.sub(string.punctuation, "", neg_text)
            neg_text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～]+", "", neg_text)
            neg_list = jieba.cut(neg_text, cut_all=False)
            neg_words.append(list(neg_list))


output = open('pos_review.pkl', 'wb')
pickle.dump(pos_words, output)
output.close()

output = open('neg_review.pkl', 'wb')
pickle.dump(neg_words, output)
output.close()

output = open('moto_senti_seg.pkl', 'wb')
pickle.dump(pos_words, output)
output.close()
