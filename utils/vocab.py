#coding=utf8
import os, json
PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'


class Vocab():
    # 词典: 单子->idx

    def __init__(self, padding=False, unk=False, min_freq=1, filepath=None):
        super(Vocab, self).__init__()
        self.word2id = dict()
        self.id2word = dict()
        if padding:
            # 这个PAD用来做对齐
            idx = len(self.word2id)
            self.word2id[PAD], self.id2word[idx] = idx, PAD
        if unk:
            # 这个UNK用来标记unkown
            idx = len(self.word2id)
            self.word2id[UNK], self.id2word[idx] = idx, UNK

        if filepath is not None:
            # 从file中导入字典
            self.from_train(filepath, min_freq=min_freq)

    def from_train(self, filepath, min_freq=1):
        with open(filepath, 'r',encoding='UTF-8') as f:
            trains = json.load(f)
        word_freq = {}
        for data in trains:
            for utt in data:
                # 每个待预测的句子
                text = utt['manual_transcript']
                for char in text:
                    # 记录每个字的频率 ? 为什么是单字呢
                    word_freq[char] = word_freq.get(char, 0) + 1
        for word in word_freq:
            if word_freq[word] >= min_freq:
                # 把频率大于1的存入到词表索引字典中
                idx = len(self.word2id)
                self.word2id[word], self.id2word[idx] = idx, word

    def __len__(self):
        return len(self.word2id)

    @property
    def vocab_size(self):
        return len(self.word2id)

    def __getitem__(self, key):
        return self.word2id.get(key, self.word2id[UNK])


class LabelVocab():

    def __init__(self, root):
        # tag标记的词表
        self.tag2idx, self.idx2tag = {}, {}

        self.tag2idx[PAD] = 0
        self.idx2tag[0] = PAD
        self.tag2idx['O'] = 1
        self.idx2tag[1] = 'O'
        self.from_filepath(root)

    def from_filepath(self, root):
        ontology = json.load(open(os.path.join(root, 'ontology.json'), 'r',encoding='UTF-8'))
        acts = ontology['acts']
        slots = ontology['slots']

        for act in acts:
            for slot in slots:
                for bi in ['B', 'I']:
                    # 基于ontology 生成所有的 tag对应的 B I
                    idx = len(self.tag2idx)
                    tag = f'{bi}-{act}-{slot}'
                    self.tag2idx[tag], self.idx2tag[idx] = idx, tag

    def convert_tag_to_idx(self, tag):
        return self.tag2idx[tag]

    def convert_idx_to_tag(self, idx):
        return self.idx2tag[idx]

    @property
    def num_tags(self):
        return len(self.tag2idx)
