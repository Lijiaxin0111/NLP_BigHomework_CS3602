import os

root_dir = os.path.expanduser("~")
# path
data_path = "./data/slu/"
model_save_dir = "./ckpt/"
model_path = "slu_model.bin"

# model hyperparameters
hidden_dim = 128
emb_dim = 768
emb_dorpout = 0.8
lstm_dropout = 0.5
attention_dropout = 0.1
num_attention_heads = 8

# hyperparameters
max_len = 32
lr_scheduler_gama = 0.5
batch_size= 32
epoch = 100
seed = 9
lr=0.001
eps = 1e-12
loss_weight = 0.6
use_gpu = False

mod = "test"

