# -*- coding: utf-8 -*-
import os, sys, time, gc, json
import random
import warnings

from data_util.data_process import *
from tqdm import tqdm, trange
from data_util.Metrics import IntentMetrics, SlotMetrics,semantic_acc
from model.joint_model_trans import Joint_model
from model.Radam import RAdam

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD, NON
from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings('ignore')
if config.use_gpu and torch.cuda.is_available():
    device = torch.device("cuda", torch.cuda.current_device())
    use_cuda = True
else:
    device = torch.device("cpu")
    use_cuda = False

import time
timestamp = time.time()
expr_name = f"slu_dca_lr_{config.lr}_weight_{config.loss_weight}_{timestamp}"
if config.mod == 'train':
    writer = SummaryWriter(os.path.join("logs",expr_name))

train_path = os.path.join(config.data_path, 'train.json')
dev_path = os.path.join(config.data_path, 'development.json')
Example.configuration(config.data_path, train_path=train_path, word2vec_path=os.path.join(config.data_path, 'word2vec-768.txt'))
train_dataset = Example.load_dataset(train_path, mod='train')
dev_dataset = Example.load_dataset(dev_path, mod='dev')

vocab_size = Example.word_vocab.vocab_size
pad_idx = Example.word_vocab[PAD]
num_tags = Example.label_vocab.num_tags
tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)
itag_non_idx = Example.label_vocab.convert_itag_to_idx(NON)


def set_seed():
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if not config.use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)


def dev(model, dev_loader):
    idx2slot = Example.label_vocab.idx2tag
    model.eval()
    eval_loss_intent = 0
    eval_loss_slot = 0
    pred_intents = []
    true_intents = []
    pred_slots = []
    true_slots = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dev_loader, desc="Evaluating")):
            # inputs, char_lists, slot_labels, intent_labels, masks, = batch
            inputs, slot_labels, intent_labels, masks, dids = batch
            if use_cuda:
                # inputs, char_lists, masks, intent_labels, slot_labels = \
                #     inputs.cuda(), char_lists.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda()
                inputs,  masks, intent_labels, slot_labels = \
                    inputs.cuda(),  masks.cuda(), intent_labels.cuda(), slot_labels.cuda()
            # logits_intent, logits_slot = model.forward_logit((inputs, char_lists), masks)
            logits_intent, logits_slot = model.forward_logit(inputs, masks)
            loss_intent, loss_slot = model.loss1(logits_intent, logits_slot, intent_labels, slot_labels, masks)

            pred_intent, pred_slot = model.pred_intent_slot(logits_intent, logits_slot, masks)
            pred_intents.extend(pred_intent.cpu().numpy().tolist())
            true_intents.extend(intent_labels.cpu().numpy().tolist())
            eval_loss_intent += loss_intent.item()
            eval_loss_slot += loss_slot.item()
            slot_labels = slot_labels.cpu().numpy().tolist()

            for i in range(len(pred_slot)):
                pred = []
                true = []
                for j in range(len(pred_slot[i])):
                    pred.append(idx2slot[pred_slot[i][j].item()])
                    true.append(idx2slot[slot_labels[i][j]])
                pred_slots.append(pred[1:-1])
                true_slots.append(true[1:-1])
    # slot f1, p, r
    slot_metrics = SlotMetrics(true_slots, pred_slots)
    slot_f1, slot_p, slot_r = slot_metrics.get_slot_metrics()
    # intent f1, p, r
    Metrics_intent = IntentMetrics(pred_intents, true_intents)
    intent_acc = Metrics_intent.accuracy
    data_nums = len(dev_loader.dataset)
    ave_loss_intent = eval_loss_intent * config.batch_size / data_nums
    ave_loss_slot = eval_loss_slot * config.batch_size / data_nums

    sent_acc = semantic_acc(pred_slots, true_slots, pred_intents, true_intents)

     
    print('\nEvaluation - intent_loss: {:.6f} slot_loss: {:.6f} acc: {:.4f}% '
          'slot f1: {:.4f} sent acc: {:.4f} \n'.format(ave_loss_intent, ave_loss_slot,
                                                       intent_acc, slot_f1, sent_acc))
    model.train()

    return [intent_acc, slot_p, slot_r, slot_f1, sent_acc, ave_loss_intent, ave_loss_slot]


def run_train(train_dataset, dev_dataset):

    print("load config and dict")
    embedding_file = open(config.data_path + "word2vec-768.txt", "r", encoding="utf-8")
    embeddings = [emb.strip() for emb in embedding_file] # 读取emb文件
    embedding_word, vocab = process_emb(embeddings, emb_dim=config.emb_dim) 
    # embbedinng_word每一行是一个词的768dim-embedding，vocab是一个word2id的字典

    idx2intent = Example.label_vocab.idx2itag
    intent2idx = Example.label_vocab.itag2idx # 两个字典: intent_label 与 index 的转换
    idx2slot = Example.label_vocab.idx2tag
    slot2idx = Example.label_vocab.tag2idx # 两个字典: slot_BIO 与 index 的转换
    n_slot_tag = len(idx2slot.items())
    n_intent_class = len(idx2intent.items())

    train_loader = get_loader(train_dataset, max_length=config.max_len, intent2idx=intent2idx, slot2idx=slot2idx,
                               vocab=vocab, is_train=True)
    dev_loader = get_loader(dev_dataset, max_length=config.max_len, intent2idx=intent2idx, slot2idx=slot2idx,
                               vocab=vocab, is_train=False)
    model = Joint_model(config, config.hidden_dim, config.batch_size, config.max_len, n_intent_class, n_slot_tag, embedding_word)

    if use_cuda:
        model.cuda()
    model.train()
    optimizer = RAdam(model.parameters(), lr=config.lr, weight_decay=0.000001)
    best_slot_f1 = [0.0, 0.0, 0.0]
    best_intent_acc = [0.0, 0.0, 0.0]
    best_sent_acc = [0.0, 0.0, 0.0]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40, 70], gamma=config.lr_scheduler_gama, last_epoch=-1)

    for epoch in trange(config.epoch, desc="Epoch"):
        print(scheduler.get_lr())
        step = 0
        epoch_loss_slot = 0
        epoch_loss_intent = 0
        epoch_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc="batch_nums")):
            step += 1
            model.zero_grad()
            # inputs, char_lists, slot_labels, intent_labels, masks, = batch
            inputs, slot_labels, intent_labels, masks, dids = batch
            if use_cuda:
                # inputs, char_lists, masks, intent_labels, slot_labels = \
                #     inputs.cuda(), char_lists.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda()
                inputs, masks, intent_labels, slot_labels = \
                    inputs.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda()
            # logits_intent, logits_slot = model.forward_logit((inputs, char_lists), masks)
            logits_intent, logits_slot = model.forward_logit(inputs, masks)
            loss_intent, loss_slot, = model.loss1(logits_intent, logits_slot, intent_labels, slot_labels, masks)

            if epoch < 40:
                loss = loss_slot + loss_intent
            else:
                loss = (1 - config.loss_weight) * loss_intent + config.loss_weight * loss_slot
            loss.backward()
            epoch_loss_intent += loss_intent
            epoch_loss_slot += loss_slot
            epoch_loss += loss
            optimizer.step()

            if step % 100 == 0:
                print("loss domain:", loss.item())
                print('epoch: {}|    step: {} |    loss: {}'.format(epoch, step, loss.item()))

        devret = dev(model, dev_loader)
        writer.add_scalar("epoch_loss", epoch_loss / step, epoch)
        writer.add_scalar("epoch_loss_intent", epoch_loss_intent / step, epoch)
        writer.add_scalar("epoch_loss_slot", epoch_loss_slot / step, epoch)
        writer.add_scalar("dev_intent_acc", devret[0], epoch)
        writer.add_scalar("dev_slot_p", devret[1], epoch)
        writer.add_scalar("dev_slot_r", devret[2], epoch)
        writer.add_scalar("dev_slot_fscore",  devret[3], epoch)
        writer.add_scalar("dev_slot_loss",  devret[6], epoch) 

        slot_f1, intent_acc, sent_acc = devret[0], devret[3], devret[4]
        if slot_f1 > best_slot_f1[1] :
            best_slot_f1 = [sent_acc, slot_f1, intent_acc, epoch]
            torch.save(model, config.model_save_dir + 'best_slot_f1_' + config.model_path)
        if intent_acc > best_intent_acc[2]:
            torch.save(model, config.model_save_dir + 'best_intent_acc_' + config.model_path)
            best_intent_acc = [sent_acc, slot_f1, intent_acc, epoch]
        if sent_acc > best_sent_acc[0]:
            torch.save(model, config.model_save_dir + 'best_sent_acc_' + config.model_path)
            best_sent_acc = [sent_acc, slot_f1, intent_acc, epoch]
        scheduler.step()
    print("best_slot_f1:", best_slot_f1)
    print("best_intent_acc:", best_intent_acc)
    print("best_sent_acc:", best_sent_acc)


def run_test(model):
    # load dict
    idx2intent = Example.label_vocab.idx2itag
    intent2idx = Example.label_vocab.itag2idx # 两个字典: intent_label 与 index 的转换
    idx2slot = Example.label_vocab.idx2tag
    slot2idx = Example.label_vocab.tag2idx # 两个字典: slot_BIO 与 index 的转换

    embedding_file = open(config.data_path + "word2vec-768.txt", "r", encoding="utf-8")
    embeddings = [emb.strip() for emb in embedding_file] # 读取emb文件
    embedding_word, vocab = process_emb(embeddings, emb_dim=config.emb_dim) 
    # embbedinng_word每一行是一个词的768dim-embedding，vocab是一个word2id的字典

    test_loader = get_loader(dev_dataset, max_length=config.max_len, intent2idx=intent2idx, slot2idx=slot2idx,
                              vocab=vocab, is_train=False)
    model = torch.load(config.model_save_dir + 'best_slot_f1_' + config.model_path, map_location=device)
    model.eval()
    pred_intents = []
    true_intents = []
    pred_slots = []
    true_slots = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # inputs, char_lists, slot_labels, intent_labels, masks, = batch
            inputs, slot_labels, intent_labels, masks, dids = batch
            if use_cuda:
                # inputs, char_lists, masks, intent_labels, slot_labels = inputs.cuda(), char_lists.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda()
                inputs, masks, intent_labels, slot_labels = inputs.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda()

            # logits_intent, logits_slot = model.forward_logit((inputs, char_lists), masks)
            logits_intent, logits_slot = model.forward_logit(inputs, masks)
            pred_intent, pred_slot = model.pred_intent_slot(logits_intent, logits_slot, masks)
            pred_intents.extend(pred_intent.cpu().numpy().tolist())
            true_intents.extend(intent_labels.cpu().numpy().tolist())

            slot_labels = slot_labels.cpu().numpy().tolist()
            for i in range(len(pred_slot)):
                pred = []
                true = []
                for j in range(len(pred_slot[i])):
                    pred.append(idx2slot[pred_slot[i][j].item()])
                    true.append(idx2slot[slot_labels[i][j]])
                pred_slots.append(pred[1:-1])
                true_slots.append(true[1:-1])
    slot_metrics = SlotMetrics(true_slots, pred_slots)
    slot_f1, _, _ = slot_metrics.get_slot_metrics()

    Metrics_intent = IntentMetrics(pred_intents, true_intents)
    print(Metrics_intent.classification_report)
    intent_acc = Metrics_intent.accuracy
    sent_acc = semantic_acc(pred_slots, true_slots, pred_intents, true_intents)
    print('\nEvaluation -  acc: {:.4f} ' 'slot f1: {:.4f} sent_acc: {:.4f}  \n'.format(intent_acc, slot_f1, sent_acc))

    return sent_acc

def destruct(label):
    if label == "O":
        return ("O", "", "")
    else:
        bio = label.split("-")[0]
        act = label.split("-")[1].split(".")[0]
        slot = label.split("-")[1].split(".")[1]
        return (bio, act, slot)

def decode(text, labels):
    pred = []
    triples = [destruct(label) for label in labels]
    for i in range(len(triples)):
        if triples[i][0] == "O":
            continue
        if triples[i][0] == "B":
            pred.append([triples[i][1], triples[i][2], text[i]])
        if triples[i][0] == "I":
            pred[-1][2] += text[i]
    return pred

def predict(model):
    print("load config and dict")
    embedding_file = open(config.data_path + "word2vec-768.txt", "r", encoding="utf-8")
    embeddings = [emb.strip() for emb in embedding_file] # 读取emb文件
    embedding_word, vocab = process_emb(embeddings, emb_dim=config.emb_dim) 
    # embbedinng_word每一行是一个词的768dim-embedding，vocab是一个word2id的字典

    idx2intent = Example.label_vocab.idx2itag
    intent2idx = Example.label_vocab.itag2idx # 两个字典: intent_label 与 index 的转换
    idx2slot = Example.label_vocab.idx2tag
    slot2idx = Example.label_vocab.tag2idx # 两个字典: slot_BIO 与 index 的转换
    n_slot_tag = len(idx2slot.items())
    n_intent_class = len(idx2intent.items())

    model.eval()
    test_path = os.path.join(config.data_path, 'test_unlabelled.json')
    test_dataset = Example.load_dataset(test_path, mod='test')
    test_loader = get_loader(test_dataset, max_length=config.max_len, intent2idx=intent2idx, slot2idx=slot2idx,
                              vocab=vocab, is_train=False)
    predictions = {}

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Predicting")):
            # inputs, char_lists, slot_labels, intent_labels, masks, = batch
            inputs, slot_labels, intent_labels, masks, dids = batch
            if use_cuda:
                # inputs, char_lists, masks, intent_labels, slot_labels = inputs.cuda(), char_lists.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda()
                inputs, masks, intent_labels, slot_labels, dids = inputs.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda(), dids.cuda()

            # logits_intent, logits_slot = model.forward_logit((inputs, char_lists), masks)
            logits_intent, logits_slot = model.forward_logit(inputs, masks)
            pred_intent, pred_slot = model.pred_intent_slot(logits_intent, logits_slot, masks)

            for i in range(len(pred_slot)):
                pred = []
                for j in range(len(pred_slot[i])):
                    pred.append(idx2slot[pred_slot[i][j].item()])
                predictions[f'{dids[i][0]}-{dids[i][1]}'] = pred[1:-1]

    test_json = json.load(open(test_path, 'r', encoding='utf-8'))
    ptr = 0
    for ei, example in enumerate(test_json):
        for ui, utt in enumerate(example):
            utt['pred'] = decode(utt['asr_1best'], predictions[f"{ei}-{ui}"])
            ptr += 1
    json.dump(test_json, open(os.path.join(config.data_path, 'prediction.json'), 'w',encoding='utf-8'), indent=4, ensure_ascii=False)



if __name__ == "__main__":
    if config.mod == "train":
        #trian model
        set_seed()
        run_train(train_dataset, dev_dataset)
    else:
        # test and predict
        model = torch.load(config.model_save_dir + 'best_slot_f1_' + config.model_path, map_location=device)
        run_test(model)
        predict(model)


