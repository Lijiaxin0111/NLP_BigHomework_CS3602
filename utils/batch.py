#-*- coding:utf-8 -*-
import torch
import random
from utils.data_augment import data_augment_example
# from utils.data_augment_new import data_augment_example


# batch : utt -- asr的解析结果
#         lengths  -- 最长的input长度
#         input_ids  -- 把input_idx 对齐 (Bsize, max_lengths)
#         labels  --  act-slot-value列表  
#         tag_ids   -- 把tag_idx 对齐  (Bsize, max_tag_len)
#         tag_mesk  -- 把PD 的位置设置为0 

def from_example_list(args, ex_list, device='cpu', train=True, aug_ratio = 0):
    ex_list = [data_augment_example(ex, aug_ratio) for ex in ex_list]
    ex_list = sorted(ex_list, key=lambda x: len(x.input_idx), reverse=True)
    batch =  Batch(ex_list, device)
    pad_idx = args.pad_idx
    tag_pad_idx = args.tag_pad_idx

    batch.utt = [ex.utt for ex in ex_list] 
    input_lens = [len(ex.input_idx) for ex in ex_list]
    max_len = max(input_lens)
    input_ids = [ex.input_idx + [pad_idx] * (max_len - len(ex.input_idx)) for ex in ex_list]
    batch.input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    batch.lengths = input_lens
    batch.did = [ex.did for ex in ex_list]

    if train:
        batch.labels = [ex.slotvalue for ex in ex_list]
        tag_lens = [len(ex.tag_id) for ex in ex_list]
        max_tag_lens = max(tag_lens)
        tag_ids = [ex.tag_id + [tag_pad_idx] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]
        tag_mask = [[1] * len(ex.tag_id) + [0] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]
        batch.tag_ids = torch.tensor(tag_ids, dtype=torch.long, device=device)
        batch.tag_mask = torch.tensor(tag_mask, dtype=torch.float, device=device)
    else:
        batch.labels = None
        batch.tag_ids = None
        tag_mask = [[1] * len(ex.input_idx) + [0] * (max_len - len(ex.input_idx)) for ex in ex_list]
        batch.tag_mask = torch.tensor(tag_mask, dtype=torch.float, device=device)
        

    return batch





    

class Batch():

    def __init__(self, examples, device):
        super(Batch, self).__init__()

        self.examples = examples
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]