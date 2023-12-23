#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torchcrf import CRF  # pytorch-crf包提供了一个CRF层的PyTorch版本实现

# 这里可以调整的超参：
# lr: 1e-5
class CRF_LSTM(nn.Module):

    def __init__(self, config):
        super(CRF_LSTM, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0) # 把词索引号映射到 embedding 向量, 权重矩阵是 |V| * embed_size, padding_idx 作为pad的索引号
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.hidden_to_tag_layer = nn.Linear(config.hidden_size, config.num_tags)   # 这里将隐藏层的输出映射到标签的数量
        self.crf = CRF(config.num_tags, batch_first=True)
        """ 
        input: 状态得分张量 (batch_size, seq_length, num_tags)
        targets: ground_truth 序列张量 (batch_size, seq_length)
        masks: 掩码张量，大小为(batch_size, seq_length)   
        """

    def forward(self, batch):
        tag_ids = batch.tag_ids # bsize,  tag_mlen
        tag_mask = batch.tag_mask  # bsize,  tag_mlen
        input_ids = batch.input_ids # bsizse, mlen
        lengths = batch.lengths # batch里面每个句子的长度


        embed = self.word_embed(input_ids) #  embed: bsize,  tag_mlen, |V|     这里得看看input_ids的维度  idx -> embed
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True) # 把batch里面的所有数据embed 连接起来, (总字数, embed_size),为了方便RNN处理序列
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True) 
        hiddens = self.dropout_layer(rnn_out)
        state = self.hidden_to_tag_layer(hiddens) 

        # print( self.crf.decode(state,tag_mask==1))

        if tag_ids == None:
            pred = self.crf.decode(state,tag_mask==1) # 解码对应的标签
            return (pred, )
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
            # if type(pred) == int:
            #     pred = [pred]
            # print("pred:",pred)
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


