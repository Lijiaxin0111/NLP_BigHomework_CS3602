#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import json 
import os
import re

from pypinyin import pinyin, lazy_pinyin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import cmp_to_key


# -----不同的距离计算方法-----------

import Levenshtein
# 编辑距离
def levenshtein_distance(str1, str2):
    return Levenshtein.distance(str1, str2)

def cosine_similarity_measure(str1, str2):
    vectorizer = CountVectorizer().fit_transform([str1, str2])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return similarity

def jaccard_similarity(str1, str2):
    set1 = set(str1)
    set2 = set(str2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union
    return similarity

# -----------------

def compare_similarity(str1, str2):
    # 在这里选择使用余弦相似度进行比较，你也可以选择其他方法
    # print(str1,str2, ":",jaccard_similarity(str1, str2))
    
    
    return jaccard_similarity(str1, str2)

# 主函数，接收字符串列表和目标字符串，按照相似度由高到低排序输出
def sort_strings_by_similarity(strings, target):
    # 使用cmp_to_key将比较函数转换为key函数
    # print(strings)
    
  

    sorted_strings = sorted(strings, key=lambda x: compare_similarity(get_pinyin(x), get_pinyin(target)), reverse=True)
    
    return sorted_strings




def get_pinyin(word):
    # 使用pinyin函数获取带声调的拼音列表
    pinyin_list = lazy_pinyin(word)
    # print(pinyin_list)
    # 将带声调的拼音列表转换为不带声调的拼音列表
    pinyin_without_tone = [''.join(item) for item in pinyin_list]


    return pinyin_without_tone



data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)) , "data" )
with open(os.path.join(data_root, "ontology.json") , "r") as ontofile:
    ontology_dict = json.load(ontofile)

    

def modified(slot,value):

    possible_values = ontology_dict["slots"][slot.split('-')[1]]
    if  type(possible_values) != type([]):
        possible_values = possible_values.split('/')[-1]
        with open(os.path.join( os.path.join(data_root, "lexicon") , possible_values )) as txt_file:
            possible_values = txt_file.readlines()
            possible_values = [ value.replace('\n',"")  for value in possible_values]
    
    
    most_possible_value =  sort_strings_by_similarity(possible_values, value)[0]
    # compare_and_sort_edit(possible_values, value)

    return most_possible_value

# print(jaccard_similarity("好","坏"))
# print(jaccard_similarity("好","好"))
# print(sort_strings_by_similarity(["好","坏","怪","好人","好东西","号"], "好"))





class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0) # 把词索引号映射到 embedding 向量, 权重矩阵是 |V| * embed_size, padding_idx 作为pad的索引号
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids # bsize,  tag_mlen
        tag_mask = batch.tag_mask  # bsize,  tag_mlen
        input_ids = batch.input_ids # bsizse, mlen
        lengths = batch.lengths # batch里面每个句子的长度

        embed = self.word_embed(input_ids) #  embed: bsize,  tag_mlen, |V|     这里得看看input_ids的维度  idx -> embed
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True) # 把batch里面的所有数据embed 连接起来, (总字数, embed_size),为了方便RNN处理序列
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True) #
        hiddens = self.dropout_layer(rnn_out)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids) # 

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]
        predictions = []
        for i in range(batch_size):
            # 得到最大的那个idx
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            # ? utt
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    value = modified(value= value, slot=slot)
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                value = modified(value= value, slot=slot)
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:
            return predictions 
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)#  self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)


    # loss 每个单字预测的tag 分布与实际索引的crossEntropy
    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1)) 
            return prob, loss
        return (prob, )
