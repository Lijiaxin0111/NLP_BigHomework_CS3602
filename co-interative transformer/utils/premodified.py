
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import difflib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_util import config

from functools import cmp_to_key
import numpy as np
from pypinyin import pinyin, lazy_pinyin
import os
import json

# -----不同的距离计算方法-----------

import Levenshtein
# 编辑距离
def levenshtein_distance(str1, str2):
    return Levenshtein.distance(str1, str2)




def jaccard_similarity(str1, str2):
    set1 = set(str1)
    set2 = set(str2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union
    return similarity

# -----------------

def compare_similarity(str1, str2,distance = "jac"):
    # 在这里选择使用余弦相似度进行比较，你也可以选择其他方法
    # print(str1,str2, ":",jaccard_similarity(str1, str2))

    if distance == "jac":
        # print("using jac")
        return jaccard_similarity(str1, str2)
    elif distance == 'lev':
        # print("using lev")
        return - levenshtein_distance(str1, str2)
    
    elif distance == "cos":
        # print("using cos")
        return cosine_similarity(str1, str2)

# 主函数，接收字符串列表和目标字符串，按照相似度由高到低排序输出
def strings_by_similarity(strings, target,distance = "jac", pinyin = True):
    # 使用cmp_to_key将比较函数转换为key函数
    # print(strings)


    if pinyin:

        vectorized_similarity = np.vectorize(lambda x: compare_similarity(get_pinyin(x), get_pinyin(target), distance ))
        # print('using pinyin')
    else:
        vectorized_similarity = np.vectorize(lambda x: compare_similarity((x), (target), distance ))
        # print('not using pinyin')

    similarity_idx = vectorized_similarity(np.array(strings)).argmax()
    most_possible_value = strings[similarity_idx]

    return most_possible_value




def get_pinyin(word):
    # 使用pinyin函数获取带声调的拼音列表
    pinyin_list = lazy_pinyin(word)
    # print(pinyin_list)
    # 将带声调的拼音列表转换为不带声调的拼音列表
    pinyin_without_tone = [''.join(item) for item in pinyin_list]


    return pinyin_without_tone

print("[PREPARE] Get the ontology")

data_root = config.data_path
with open(os.path.join(data_root, "ontology.json") , "r", encoding='utf-8') as ontofile:
    ontology_dict = json.load(ontofile)


possible_values_dict = {}

for key in ontology_dict["slots"].keys():
    possible_values = ontology_dict["slots"][key]
    if  type(possible_values) != type([]):
        possible_values = possible_values.split('/')[-1]
        with open(os.path.join( os.path.join(data_root, "lexicon") , possible_values ), encoding='utf-8') as txt_file:
            possible_values = txt_file.readlines()
            possible_values_dict[key] = [ (value.replace('\n',""))  for value in possible_values]
    else:
        
        possible_values_dict[key] = [ (value.replace('\n',""))  for value in possible_values]

    
            
print("[DONE] Process the ontology")

# print(possible_values_dict.keys())
# print(possible_values_dict)

def modified(slot,value,distance = "jac", pinyin = True):


    possible_values = possible_values_dict[slot]
    
    
    most_possible_value =  strings_by_similarity(possible_values, value, distance , pinyin)
    # if(value == "第一"):
        # print(most_possible_value)
        # print(sort_strings_by_similarity(possible_values, value))
    # compare_and_sort_edit(possible_values, value)

    return most_possible_value




def modified_preds(pred, distance = "jac", pinyin = True):
    new_pred = []
    cnt = 0
    for value_list in pred:
        cnt += 1
        new_value_list = []
        for asv in value_list:
            asv = asv.split("-")
            act = asv[0]
            slot = asv[1]
            value = asv[2]
            if value in possible_values_dict[slot]:
                new_value_list.append('-'.join(asv))
            else:
                asv[2] = modified(slot, value , distance , pinyin)
                new_value_list.append('-'.join(asv))
        new_pred.append(new_value_list)
        if cnt % 50 == 0:
            print(f"Modifying {cnt} / {len(pred)}...")
    return new_pred

def modified_pred(pred, distance = "jac", pinyin = True):
    new_pred = []
    for asv in pred:
        act = asv[0]
        slot = asv[1]
        value = asv[2]
        if value in possible_values_dict[slot]:
            new_pred.append('-'.join(asv))
        else:
            asv[2] = modified(slot, value , distance , pinyin)
            new_pred.append('-'.join(asv))
    return new_pred


                
# print(modified("终点目标","金沙人民", distance= "jac",pinyin= False))
# print(jaccard_similarity("好","好"))


# ---------------上面是同义词纠正------------------


# 下面的词向量求相似度不准 or 对于没有见过的词 效果不好
# # ----------------------测试相似词的直接匹配------------------------

# def train_word2vec():
#     corpus = []

#     for key in ontology_dict["slots"].keys():
#         possible_values = ontology_dict["slots"][key]
#         if  type(possible_values) != type([]):
#             possible_values = possible_values.split('/')[-1]
#             with open(os.path.join( os.path.join(data_root, "lexicon") , possible_values )) as txt_file:
#                 possible_values = txt_file.readlines()
#                 corpus = corpus + [ (value.replace('\n',""))  for value in possible_values]
        
#         else:
#             corpus = corpus + [ (value.replace('\n',""))  for value in possible_values]

#     tokenized_corpus = [list(jieba.cut(sentence)) for sentence in corpus]

#     # 训练Word2Vec模型
#     model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
#     print("[DONE] train the Word2Vec")
#     return model

# # 计算相似度
# def get_most_similar_2(word, model):
#     try:
#         word_vector = model.wv[word]
#         similarities = [(other_word, cosine_similarity([word_vector], [model.wv[other_word]])[0][0]) for other_word in model.wv.index_to_key]
#         similarities.sort(key=lambda x: x[1], reverse=True)
#         print(similarities)
#         return similarities
#     except KeyError:
#         return []


            
# modified_model = train_word2vec()


# def modified_2(slot,value):
    

#     return  get_most_similar_2(value, modified_model)[0]

# # 上面这个对于词表中没有出现过的词 不能计算相似度

# # ----------------------


# def get_corpus():
#     corpus = []

#     for key in ontology_dict["slots"].keys():
#         possible_values = ontology_dict["slots"][key]
#         if  type(possible_values) != type([]):
#             possible_values = possible_values.split('/')[-1]
#             with open(os.path.join( os.path.join(data_root, "lexicon") , possible_values )) as txt_file:
#                 possible_values = txt_file.readlines()
#                 corpus = corpus + [ (value.replace('\n',""))  for value in possible_values]
        
#         else:
#             corpus = corpus + [ (value.replace('\n',""))  for value in possible_values]

#     tokenized_corpus = [list(jieba.cut(sentence)) for sentence in corpus]
#     tokenized_corpus =  corpus
    

#     # 训练Word2Vec模型

#     print("[DONE] get the corpus")
#     return tokenized_corpus


# tokenized_corpus = get_corpus()

# def modified_3(slot,value):

#     most_similar_word = difflib.get_close_matches(value, tokenized_corpus[0], n=1, cutoff=0.8)[0]
#     return most_similar_word

# # print(modified_3("s","导航"))
