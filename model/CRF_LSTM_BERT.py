#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torchcrf import CRF  # pytorch-crf包提供了一个CRF层的PyTorch版本实现

from transformers import AutoTokenizer, AutoModelForMaskedLM 
BERT_VOCAO_SIZE = 21128

class CRF_LSTM_BERT(nn.Module):

    def __init__(self, config):
        super(CRF_LSTM_BERT, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        # self.word_embed = nn.Embedding(BERT_VOCAO_SIZE, config.embed_size, padding_idx=0) # 把词索引号映射到 embedding 向量, 权重矩阵是 |V| * embed_size, padding_idx 作为pad的索引号
        # self.word_embed_transf =nn.Linear(BERT_VOCAO_SIZE, config.embed_size)

        self.rnn = getattr(nn, self.cell)(BERT_VOCAO_SIZE, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.hidden_to_tag_layer = nn.Linear(config.hidden_size, config.num_tags)   # 这里将隐藏层的输出映射到标签的数量
        self.crf = CRF(config.num_tags, batch_first=True)
        """ 
        CRF的forward的输入参数:
        input: 状态得分张量 (batch_size, seq_length, num_tags)
        targets: ground_truth 序列张量 (batch_size, seq_length)
        masks: 掩码张量，大小为(batch_size, seq_length) 

        CRF的decoder输入参数：
        input: 状态得分张量 (batch_size, seq_length, num_tags)
        masks: 掩码张量，大小为(batch_size, seq_length) 
        """


        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")  
        self.bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese",output_hidden_states=True,output_attentions=True)

        self.freeze_layer_num = 5

        # Freeze all layers except the last few
        for param in self.bert_model.base_model.parameters():
            param.requires_grad = False

        # Unfreeze the last N layers
        for param in self.bert_model.base_model.encoder.layer[-self.freeze_layer_num:].parameters():
            param.requires_grad = True

    def forward(self, batch):
        tag_ids = batch.tag_ids # bsize,  tag_mlen
        tag_mask = batch.tag_mask  # bsize,  tag_mlen
        lengths = batch.lengths # batch里面每个句子的长度
        utts = batch.utt

        
        inputs = self.tokenizer(utts, padding='max_length', truncation=True, max_length=tag_ids.shape[1],return_tensors="pt")  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    
   
        bert_out = self.bert_model(inputs["input_ids"]) #  通过bert把 输入的utt转变为embed
        all_hidden_states, all_attentions ,logits = bert_out['hidden_states'], bert_out['attentions'], bert_out["logits"]
        embed =  logits
    
        # embed = all_hidden_states[12]
        



        # embed = self.word_embed(input_ids) #  embed: bsize,  tag_mlen, |V|     这里得看看input_ids的维度  idx -> embed
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True) # 把batch里面的所有数据embed 连接起来, (总字数, embed_size),为了方便RNN处理序列
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True) 
        hiddens = self.dropout_layer(rnn_out)
        state = self.hidden_to_tag_layer(hiddens) 

        # print( self.crf.decode(state,tag_mask==1))

        if tag_ids == None:
            pred = self.crf.decode(state,tag_mask==1) # 解码对应的标签
            return pred
        else:

            tag_loss = -1 * (   self.crf(state,tag_ids, tag_mask == 1)) # 增加了scf层
            pred = self.crf.decode(state,tag_mask== 1) # 解码对应的标签
            return pred, tag_loss

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        
        predictions = []
        tag_list = output[0]
        for i in range(batch_size):
            # 得到最大的那个idx
            # pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred = tag_list[i]
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            # pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
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
