"""
分析测试集中，slot的值不出现在asr中的情况所占的比例
"""


import sys, os, time, gc, json
from torch.optim import Adam

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from model.slu_baseline_tagging import SLUTagging
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import tqdm

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)

print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
train_dataset = Example.load_dataset(train_path) 
dev_dataset = Example.load_dataset(dev_path) # ? 测试数据集
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.vocab_size = Example.word_vocab.vocab_size 
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)



expr_name = f"Analy_impossible_case_ratio"
print("[EXPRI] ", expr_name)
dataset = dev_dataset
cnt = 0
impossible_case = []


for i in range(0, len(dataset)):
    ex = dataset[i].ex
    # print(ex)
    for  label in ex['semantic']:
        if label[2] not in ex["asr_1best"]:
            cnt += 1
            impossible_case.append([ex])
            break
    # print("DONE")
print(impossible_case)

print("impossible_case_ratio: ", cnt / len(dataset))


