# NLP_BigHomework

## 实验进度

---




# 数据增强
## 1217
### 1217 ljx: 
 - 测试数据增强,  以及实现没有dev集数据泄露的数据增强
 - 完成上述模型和 baseline在不同lr下的性能比较
 - 性能对比详见`Aug_ratio 测试.md`

# CRF_LSTM_BERT   CRF_LSTM Pure_BERT
## 1217
### 1217 ljx: 
 - 实现 CRF_LSTM_BERT   CRF_LSTM Pure_BERT
 - 完成上述模型和 baseline在不同lr下的性能比较



  
# Dev Log
## 1217
### Learning ASR-Robust Contextualized Embeddings for Spoken Language Understanding
* https://arxiv.org/abs/1909.10861
* https://github.com/MiuLab/SpokenVec
* 通过Word Confusion Network(WCN)来处理谐音/同音
* 通过预训练语言模型，得到有上下文信息的embedding
* 将经过上述两个步骤之后的ASR robust contextualized word embedding输入给SLU模块的BiLSTM，进而完成SLU任务（论文中是Intent Detection，不过我觉得这个任务是Intent还是SlotFilling关系不大）
* 关于代码：有一个我下载不下来的库（似乎与slu环境的python3.11不兼容），自己实现的话可能有些麻烦，但还谈不上完全做不了
* 启发：处理上下文，可以在前面添加一个预训练网络，用contextualized word embedding代替固定的embedding；处理ASR同音/谐音（这在中文里尤其明显），可以用WCN，WCN的训练方法文中也有涉及，也有其他文献

### Injecting Word Information with Multi-Level Word Adapter for Chinese Spoken Language Understanding
* https://arxiv.org/abs/2010.03903
* https://github.com/AaronTengDeChuan/MLWA-Chinese-SLU
* 使用了两个独立的channel，一个是char channel以每个中文字符为单位，另一个是word channel以分词结果中的中文分词为单位。
* 对于char channel，首先通过self-attention encoder得到中文字符的编码，这个模块由自注意力模块和BiLSTM组成，用于获取上下文信息和字符序列信息，得到ec
* 对于word channel，同样通过self-attention encoder，得到ew。
* 让ec，ew分别通过MLP Attention模块并经过word adapter以此预测Intent。
* 用单向LSTM作为slot-filling的decoder，处理ec，同时将预测的Intent作为输入，得到hc。
* 用双向LSTM作为slot-filling的decoder，处理ew，同时将预测的Intent作为输入，得到hw。
* 让hc和hw经过word adapter后用于槽值填充。
* 其中word adpter是一个简单的神经网络。
* 目前已经配置好了环境，代码还没有弄明白该咋用，使用两个channel来充分利用中文上下文的方法或许值得借鉴。
