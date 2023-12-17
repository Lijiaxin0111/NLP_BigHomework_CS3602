# NLP_BigHomework

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

