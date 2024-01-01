import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator
import random
import os
import re

# 训练的时候，


class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path):
        # 导入数据集中的预料
        dataset = json.load(open(data_path, "r",encoding="UTF-8"))
        examples = []
        # data 是一大轮的数据,里面可能还有多个回合的数据
        for di, data in enumerate(dataset):
            # utt 一回合的数据
            for ui, utt in enumerate(data):
                # 数据类:  Example() 
                #         属性 ex 字典对应key有utt_id , manual_transcript, asr_1best , semantic : [act, slot, value] d;
                #         属性 did 对应第几轮的第几回合的数据
                #         属性 utt : asr_1best
                #         属性 slot: 字典 : "act--slot" -> value
                #         属性 tags :对应asr_1best 长度的 tag序列
                #         属性  slotvalue: 这个数据的 act-slot-value列表
                #         属性 input_idx: 得到对应每个asr_1best里面的单字的词表索引列表
                #         属性 tag_id: 得到对应tags的索引号列表
                ex = cls(utt, f"{di}-{ui}")
                examples.append(ex)
        return examples

    def __init__(self, ex: dict, did):
        super(Example, self).__init__()
        self.ex = ex
        self.did = did
        

        self.utt = ex["asr_1best"]
        # self.utt = ex["manual_transcript"]
        self.slot = {}
        for label in ex["semantic"]:
            act_slot = f"{label[0]}-{label[1]}"
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        #初始化 tags
        self.tags = ["O"] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            # 找到asr中对应value的位置
            bidx = self.utt.find(value)
            if bidx != -1:
                # 把 asr对应value的位置标记为 I-act-slot
                # 起始位置标记为 B-act-slot
                self.tags[bidx: bidx + len(value)] = [f"I-{slot}"] * len(value)
                self.tags[bidx] = f"B-{slot}"
        self.slotvalue = [f"{slot}-{value}" for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]




