# -*- coding: utf-8 -*-

"""
@CreateTime :       2020/3/13 8:44
@Author     :       dcteng
@File       :       process.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2020/3/13 8:44
"""


import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
import os
import json
import time
import math
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
from matplotlib import pyplot as plt
from src.utils import computeF1Score, restore_order, save_slu_error_result, save_slu_result

class Processor(object):
    # TODO: divide original text into char-level text and word-level text, and provide alignment information between two level texts

    def __init__(self, dataset, model, batch_size):
        self.__dataset = dataset
        self.__model = model
        self.__batch_size = batch_size

        if torch.cuda.is_available():
            time_start = time.time()
            self.__model = self.__model.cuda()

            time_con = time.time() - time_start
            print("The model has been loaded into GPU and cost {:.6f} seconds.\n".format(time_con))

        self.__criterion = nn.NLLLoss()

        t_total = math.ceil(self.__dataset.num_training_samples / self.__batch_size) * self.__dataset.num_epoch
        if self.__dataset.use_bert_input:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.__model.named_parameters()], 'weight_decay': 0.0}
            ]
            self.__optimizer = AdamW(optimizer_grouped_parameters, lr=self.__dataset.learning_rate, eps=1e-8)
            self.__scheduler = get_linear_schedule_with_warmup(self.__optimizer, num_warmup_steps=0, num_training_steps=t_total)
        else:
            self.__optimizer = optim.Adam(self.__model.parameters(), lr=self.__dataset.learning_rate,
                                          weight_decay=self.__dataset.l2_penalty)
            self.__scheduler = get_linear_schedule_with_warmup(self.__optimizer, num_warmup_steps=0, num_training_steps=t_total)

    def train(self):
        best_dev_slot = 0.0
        best_dev_intent = 0.0
        best_dev_sent = 0.0
        best_dev_metric = 0.0

        dataloader = self.__dataset.batch_delivery('train')
        for epoch in range(0, self.__dataset.num_epoch):
            total_slot_loss, total_intent_loss = 0.0, 0.0

            time_start = time.time()
            self.__model.train()
            for batch in tqdm(dataloader, ncols=50) if not self.__dataset.no_progressbar else dataloader:
                # batch
                # 0: char_text_batch
                # 1: word_text_batch
                # 2: align_info_batch
                # 3: slot_batch
                # 4: intent_batch
                # 5: word_list_batch
                # 6: sent_seg_batch
                # 7: bert_input_id
                # 8: bert_attention_mask
                # 9: bert_token_type_id
                padded_char_text, padded_word_text, sorted_char_items, sorted_word_items, char_seq_lens, word_seq_lens, _ = \
                    self.__dataset.add_padding(
                        batch[0], batch[1],
                        char_items=[(batch[3], False), (batch[4], False), (batch[5], False), (batch[7], False), (batch[8], False), (batch[9], False)],
                        word_items=[(batch[2], False), (batch[6], False)]
                    )
                sorted_intent = list(Evaluator.expand_list(sorted_char_items[1]))

                if self.__dataset.use_bert_input:
                    char_text_var = [
                        torch.tensor(sorted_char_items[3], dtype=torch.long),
                        torch.tensor(sorted_char_items[4], dtype=torch.long),
                        torch.tensor(sorted_char_items[5], dtype=torch.long)
                    ]
                else:
                    char_text_var = torch.tensor(padded_char_text, dtype=torch.long)

                word_text_var = torch.tensor(padded_word_text, dtype=torch.long)
                slot_var = torch.tensor(list(Evaluator.expand_list(sorted_char_items[0])), dtype=torch.long)
                intent_var = torch.tensor(sorted_intent, dtype=torch.long)

                if torch.cuda.is_available():
                    if self.__dataset.use_bert_input:
                        char_text_var = [item.cuda() for item in char_text_var]
                    else:
                        char_text_var = char_text_var.cuda()

                    word_text_var = word_text_var.cuda()
                    slot_var = slot_var.cuda()
                    intent_var = intent_var.cuda()

                random_slot = random.random()
                slot_out, intent_out = \
                    self.__model(char_text_var, char_seq_lens,
                                 word_text_var, word_seq_lens, sorted_word_items[0], [sorted_char_items[2], sorted_word_items[1]],
                                 forced_slot=slot_var if random_slot < self.__dataset.slot_forcing_rate else None,
                                 golden_intent=intent_var if self.__dataset.golden_intent else None)

                slot_loss = self.__criterion(slot_out, slot_var)
                intent_loss = self.__criterion(intent_out, intent_var)
                batch_loss = slot_loss + intent_loss

                self.__optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.__model.parameters(), self.__dataset.max_grad_norm)
                self.__optimizer.step()
                self.__scheduler.step()

                try:
                    total_slot_loss += slot_loss.cpu().item()
                    total_intent_loss += intent_loss.cpu().item()
                except AttributeError:
                    total_slot_loss += slot_loss.cpu().data.numpy()[0]
                    total_intent_loss += intent_loss.cpu().data.numpy()[0]

            time_con = time.time() - time_start
            print('[Epoch {:2d}]: The total slot loss on train data is {:2.6f}, intent data is {:2.6f}, '
                  'cost about {:2.6} seconds.'.format(epoch, total_slot_loss, total_intent_loss, time_con))
            writer = SummaryWriter("logs/lr=1e-4")
            writer.add_scalar("Slot_loss", total_slot_loss, epoch)
            writer.add_scalar("Intent_loss", total_intent_loss, epoch)
            """plt.plot(total_slot_loss, label ='total_slot_loss')#画出val_acc数据
            plt.title('total_slot_loss')#设置标题
            plt.legend()#画出图例
            plt.show()"""
            time_start = time.time()
            dev_slot_acc, dev_f1_score, dev_precision, dev_recall, dev_acc = self.estimate(if_dev=True, test_batch=self.__batch_size * 2)
            writer.add_scalar("acc", dev_slot_acc, epoch)
            writer.add_scalar("fscore", dev_f1_score, epoch)
            writer.add_scalar("precision", dev_precision, epoch)
            writer.add_scalar("recall", dev_recall, epoch)
            writer.add_scalar("intent_acc", dev_acc, epoch)
            writer.close()
            # dev_metric = dev_f1_score
            dev_metric = dev_slot_acc #if not self.__dataset.golden_intent else dev_f1_score

            if dev_metric > best_dev_metric:
                slot_acc, test_f1, test_precision, test_recall, test_acc = self.estimate(if_dev=False, test_batch=self.__batch_size * 2)

                best_dev_metric = dev_metric
                best_dev_slot = dev_f1_score
                best_dev_precision = dev_precision
                best_dev_recall = dev_recall
                best_dev_acc = dev_slot_acc
                best_dev_intent = dev_acc

                print('\nTest result: slot acc score: {:.6f}, slot f1 score: {:.6f}, precision: {:.6f}, recall: {:.6f}, intent acc score: {:.6f}, '
                      .format(slot_acc, test_f1, test_precision, test_recall, test_acc))

                model_save_dir = os.path.join(self.__dataset.save_dir, "model")
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)

                torch.save(self.__model, os.path.join(model_save_dir, "model.pkl"))
                torch.save(self.__dataset, os.path.join(model_save_dir, 'dataset.pkl'))
                """plt.plot(dev_f1_score, label ='dev_f1_score')#画出val_acc数据
                plt.title('dev_f1_score')#设置标题
                plt.legend()#画出图例
                plt.show()"""
                time_con = time.time() - time_start
                print('[Epoch {:2d}]: In validation process, the slot acc is {:2.6f}, '
                      'the slot f1 score is {:2.6f}, the precision is {:2.6f}, the recall is {:2.6f}, the intent acc is {:2.6f} '
                      'cost about {:2.6f} seconds.\n'.format(epoch, dev_slot_acc, dev_f1_score, dev_precision, dev_recall, dev_acc, time_con))

    def estimate(self, if_dev, test_batch=100):
        """
        Estimate the performance of model on dev or test dataset.
        """
        if if_dev:
            pred_slot, real_slot, pred_intent, real_intent = self.prediction(
                self.__model, self.__dataset, "dev", test_batch
            )
        else:
            pred_slot, real_slot, pred_intent, real_intent = self.prediction(
                self.__model, self.__dataset, "test", test_batch
            )

        slot_f1, precision, recall = computeF1Score(pred_slot, real_slot)
        slot_acc = Evaluator.slot_accuracy(pred_slot, real_slot, pred_intent, real_intent)
        intent_acc = Evaluator.accuracy(pred_intent, real_intent)
        #sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)

        return slot_acc, slot_f1, precision, recall, intent_acc

    @staticmethod
    def validate(model, dataset, batch_size):
        """
        validation will write mistaken samples to files and make scores.
        """
        pred_slot, real_slot, pred_intent, real_intent = Processor.prediction(model, dataset, "test", batch_size)
        def output_trans(pred_slot, pred_intent, asr):
            ans = []
            st_pos = 0
            ed_pos = 0
            tot = 0
            slot = "NULL"
            for i in range(len(asr)):
                if(pred_slot[i][0] == 'B'):
                    st_pos = i
                    ed_pos = -1
                    slot = pred_slot[i][2:]
                else:
                    if (len(pred_slot[i]) >= 2):
                        if ((pred_slot[i][0] == 'I') and (pred_slot[i][1] != '-')):
                            if (ed_pos == -1):
                                ans.append(pred_intent+ '-'+slot+ '-'+asr[st_pos: i+1])
                                ed_pos = 0
                            else:
                                ans.append(pred_intent+ '-'+pred_slot[i][2:]+ '-'+asr[i: i+1])
            if (ed_pos == -1):
                ans.append(pred_intent+ '-'+slot+ '-'+asr[st_pos:])
            print(asr)
            print(pred_slot)
            print(pred_intent)
            print(ans)
            return ans

        with open('data/Project_data/test_out.json','w',encoding='utf-8') as file_out:
            with open('data/Project_data/test.json','r', encoding='utf-8') as file:
                data = json.load(file)
                cnt1 = 0
                cnt2 = 0
                hit = 0
                tot = 0
                data_out = []
                my_pred = []
                my_slot = []
                for items in data:
                    data_out.append([])
                    for entry in items:
                        entry_out = {}
                        entry_out["utt_id"] = entry["utt_id"]
                        #entry_out["manual_transcript"] = entry["manual_transcript"]
                        entry_out["asr_1best"] = entry["asr_1best"]
                        if (len(entry_out["asr_1best"]) < len(pred_slot[cnt1])):
                            entry_out["asr_1best"] = entry_out["asr_1best"] + ("#" * (len(pred_slot[cnt1]) - len(entry_out["asr_1best"])))
                        entry_out["semantics"] = []
                        entry_out["pred"] = output_trans(pred_slot[cnt1], pred_intent[cnt1], entry_out["asr_1best"])
                        tmp = []
                        for i, triple in enumerate(entry["semantic"]):
                            tmp.append(triple[0]+ '-'+triple[1]+ '-'+triple[2])
                        print(entry_out["pred"])
                        print(tmp)
                        #print("before",tmp)
                        #if (len(entry["semantic"]) < len(entry_out["pred"])):
                        #    for i in range(len(entry_out["pred"]) - len(entry["semantic"])):
                        #        tmp.append([])
                        #print("after", tmp)
                        my_pred.append(entry_out["pred"])
                        my_slot.append(tmp)
                        #for i, pred in enumerate(entry_out["pred"]):
                        #    tot += 1
                        #    if (len(entry["semantic"]) > i):
                        #        hit += set(pred) == set(entry["semantic"][i])
                        cnt1 += 1
                        data_out[cnt2].append(entry_out)
                    cnt2 += 1
                json.dump(data_out, file_out, indent=2, ensure_ascii=False)

        slot_f1, precision, recall = computeF1Score(my_pred, my_slot)
        #slot_acc = Evaluator.slot_accuracy(pred_slot, real_slot, pred_intent, real_intent)
        slot_acc = Evaluator.slot_accuracy(my_pred, my_slot)
        intent_acc = Evaluator.accuracy(pred_intent, real_intent)
        #sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)

        return slot_acc, slot_f1, precision, recall, intent_acc

    @staticmethod
    def prediction(model, dataset, mode, batch_size):
        model.eval()

        if mode == "dev":
            dataloader = dataset.batch_delivery('dev', batch_size=batch_size, shuffle=False, is_digital=False)
        elif mode == "test":
            dataloader = dataset.batch_delivery('test', batch_size=batch_size, shuffle=False, is_digital=False)
        else:
            raise Exception("Argument error! mode belongs to {\"dev\", \"test\"}.")

        char_text, word_text = [], []
        pred_slot, real_slot = [], []
        pred_intent, real_intent = [], []

        for batch in tqdm(dataloader, ncols=50) if not dataset.no_progressbar else dataloader:
            # batch
            # 0: char_text_batch
            # 1: word_text_batch
            # 2: align_info_batch
            # 3: slot_batch
            # 4: intent_batch
            # 5: word_list_batch
            # 6: sent_seg_batch
            # 7: bert_input_id
            # 8: bert_attention_mask
            # 9: bert_token_type_id
            padded_char_text, padded_word_text, sorted_char_items, sorted_word_items, char_seq_lens, word_seq_lens, sorted_index = \
                dataset.add_padding(
                    batch[0], batch[1],
                    char_items=[(batch[3], False), (batch[4], False), (batch[5], False), (batch[7], False),
                                (batch[8], False), (batch[9], False)],
                    word_items=[(batch[2], False), (batch[6], False)],
                    digital=False
                )
            sorted_intent = list(Evaluator.expand_list(sorted_char_items[1]))

            char_text.extend(restore_order(padded_char_text, sorted_index))
            word_text.extend(restore_order(padded_word_text, sorted_index))
            real_slot.extend(restore_order(sorted_char_items[0], sorted_index))
            real_intent.extend(restore_order(sorted_intent, sorted_index))

            if dataset.use_bert_input:
                var_char_text = [
                    torch.tensor(sorted_char_items[3], dtype=torch.long),
                    torch.tensor(sorted_char_items[4], dtype=torch.long),
                    torch.tensor(sorted_char_items[5], dtype=torch.long)
                ]
            else:
                digit_char_text = dataset.char_alphabet.get_index(padded_char_text)
                var_char_text = torch.tensor(digit_char_text, dtype=torch.long)

            digit_word_text = dataset.word_alphabet.get_index(padded_word_text)
            digit_intent = dataset.intent_alphabet.get_index(sorted_intent)
            var_word_text = torch.tensor(digit_word_text, dtype=torch.long)
            var_intent = torch.tensor(digit_intent, dtype=torch.long)

            if torch.cuda.is_available():
                if dataset.use_bert_input:
                    var_char_text = [item.cuda() for item in var_char_text]
                else:
                    var_char_text = var_char_text.cuda()

                var_word_text = var_word_text.cuda()
                var_intent = var_intent.cuda()

            slot_idx, intent_idx = model(var_char_text, char_seq_lens,
                                         var_word_text, word_seq_lens, sorted_word_items[0], [sorted_char_items[2], sorted_word_items[1]],
                                         n_predicts=1, golden_intent=var_intent if dataset.golden_intent else None)

            nested_slot = Evaluator.nested_list([list(Evaluator.expand_list(slot_idx))], char_seq_lens)[0]
            pred_slot.extend(restore_order(dataset.slot_alphabet.get_instance(nested_slot), sorted_index))
            pred_intent.extend(
                restore_order(dataset.intent_alphabet.get_instance(list(Evaluator.expand_list(intent_idx))),
                              sorted_index))
            #for i in range(len(pred_slot)):
             #       pred = []
              #      for j in range(len(pred_slot[i])):
               #         pred.append(idx2slot[pred_slot[i][j].item()])
                #    predictions[f'{dids[i][0]}-{dids[i][1]}'] = pred[1:-1]
        # save error result
        save_slu_error_result(pred_slot, real_slot, pred_intent, real_intent, [char_text, word_text], dataset.save_dir, mode)
        # save slu result
        save_slu_result(pred_slot, real_slot, pred_intent, real_intent, [char_text, word_text], dataset.save_dir, mode)

        return pred_slot, real_slot, pred_intent, real_intent


class Evaluator(object):

    @staticmethod
    def semantic_acc(pred_slot, real_slot, pred_intent, real_intent):
        """
        Compute the accuracy based on the whole predictions of
        given sentence, including slot and intent.
        """

        total_count, correct_count = 0.0, 0.0
        for p_slot, r_slot, p_intent, r_intent in zip(pred_slot, real_slot, pred_intent, real_intent):

            if p_slot == r_slot and p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count
    
    @staticmethod
    def slot_accuracy(pred_list, real_list):
        """
        Get accuracy measured by predictions and ground-trues.
        """
        corr, total = 0, 0
        for i, pred in enumerate(pred_list):
            total += 1
            corr += set(pred) == set(real_list[i])
        print(total)
        return 100 * corr / total
        """pred_array = np.array(list(Evaluator.expand_list(pred_list)))
        real_array = np.array(list(Evaluator.expand_list(real_list)))
        return (pred_array == real_array).sum() * 1.0 / len(pred_array)"""


    @staticmethod
    def accuracy(pred_list, real_list):
        """
        Get accuracy measured by predictions and ground-trues.
        """

        pred_array = np.array(list(Evaluator.expand_list(pred_list)))
        real_array = np.array(list(Evaluator.expand_list(real_list)))
        return (pred_array == real_array).sum() * 1.0 / len(pred_array)

    @staticmethod
    def max_freq_predict(sample):
        predict = []
        for items in sample:
            predict.append(Counter(items).most_common(1)[0][0])
        return predict

    @staticmethod
    def expand_list(nested_list):
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Evaluator.expand_list(item):
                    yield sub_item
            else:
                yield item

    @staticmethod
    def nested_list(items, seq_lens):
        num_items = len(items)
        trans_items = [[] for _ in range(0, num_items)]

        count = 0
        for jdx in range(0, len(seq_lens)):
            for idx in range(0, num_items):
                trans_items[idx].append(items[idx][count:count + seq_lens[jdx]])
            count += seq_lens[jdx]

        return trans_items
