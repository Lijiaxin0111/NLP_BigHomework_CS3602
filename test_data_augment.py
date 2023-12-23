import sys, os, time, gc, json
from torch.optim import Adam

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.data_augment import data_augment_example
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






utt0 =    {
            "utt_id": 2,
            "manual_transcript": "第五个",
            "asr_1best": "第五个",
            "semantic": [
                [
                    "inform",
                    "序列号",
                    "第五个"
                ]
            ]
        }

utt1 =         {
            "utt_id": 2,
            "manual_transcript": "下一页",
            "asr_1best": "下一页",
            "semantic": [
                [
                    "inform",
                    "页码",
                    "下一页"
                ]
            ]
        }

utt2 =        {
            "utt_id": 1,
            "manual_transcript": "请你导航给我导的平阴平阴新一中躲避拥堵",
            "asr_1best": "请你导航我倒的平阴平阴新一中躲避拥堵",
            "semantic": [
                [
                    "inform",
                    "操作",
                    "导航"
                ],
                [
                    "inform",
                    "操作",
                    "导"
                ],
                [
                    "inform",
                    "终点名称",
                    "平阴平阴新一中"
                ],
                [
                    "inform",
                    "路线偏好",
                    "躲避拥堵"
                ]
            ]
        }

# print( utt1["asr_1best"])
test_example = Example(utt2, "1-1")
print(test_example.utt)
print(test_example.tags)

data_aug = data_augment_example(test_example)
print(data_aug.utt)

print(data_aug.tags)



# print(data_augment_example(test_example).tags)