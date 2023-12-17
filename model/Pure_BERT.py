#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from transformers import AutoTokenizer, AutoModelForTokenClassification
from utils.initialization import *
BERT_VOCAO_SIZE = 768

class Pure_BERT(nn.Module):

    def __init__(self, config):
        super(Pure_BERT, self).__init__()
        self.config = config
        self.device =  set_torch_device(config.device)


        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(BERT_VOCAO_SIZE, config.num_tags, config.tag_pad_idx)

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.bert_model = AutoModelForTokenClassification.from_pretrained("bert-base-chinese",output_hidden_states=True,output_attentions=True).to(self.device)

        self.freeze_layer_num = config.freeze_layer_num

        # Freeze all layers except the last few
        for param in self.bert_model.base_model.parameters():
            param.requires_grad = False

        # Unfreeze the last N layers
        for param in self.bert_model.base_model.encoder.layer[-self.freeze_layer_num:].parameters():
            param.requires_grad = True


    def forward(self, batch):
        tag_ids = batch.tag_ids # bsize,  tag_mlen
        tag_mask = batch.tag_mask  # bsize,  tag_mlen
        input_ids = batch.input_ids # bsizse, mlen
        lengths = batch.lengths # batch里面每个句子的长度
        utts = batch.utt

        inputs = self.tokenizer(utts, padding='max_length', truncation=True, max_length=tag_mask.shape[1],return_tensors="pt").to(self.device)  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        # print("tokenizer need: ", time.time() - start)

    
        bert_out = self.bert_model(inputs["input_ids"], batch.tag_mask) #  通过bert把 输入的utt转变为embed
        # print("bert need: ", time.time() - start)
        all_hidden_states, all_attentions ,logits = bert_out['hidden_states'], bert_out['attentions'], bert_out["logits"]

        # embed =  logits
        hiddens_state_output = 12
        embed = all_hidden_states[hiddens_state_output]

#
        hiddens = self.dropout_layer(embed)
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
