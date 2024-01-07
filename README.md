# README



## 文件结构

`scripts`  :

- 含有 LSTM_CRF 、 Pure_BERT、 BERT_LSTM_CRF以及带有数据增强和后纠错的训练框架代码
- `analyze`: 数据分析代码
- `test_hyperparameter` 含有测试训练超参的脚本

`model`: 含有LSTM_CRF 、 Pure_BERT、 BERT_LSTM_CRF的模型实现

`co-interative transformer`: 含有Co-interactive transformer的模型实现以及训练测试代码

`MLWA-Chinese-SLU-baseline-main`：  含有MLWA的模型实现以及训练测试代码

`utils`  :含有 `data_augment.py` 数据增强模块   `premodified.py` 后纠错模块

`train_split.json`: 数据增强测试时，抽取出的来自train set的value值

`appendix`： 含有未在实验报告展示的实验测试结果图

`data\prediction.json` 为输出`test.json`的预测value值文件 

> 由于所给的test样例较为简单，我们只给出了co-interative transformer的预测结果，其他的预测结果基本相同



除了上述文件，还对所给框架中的`args`、 `baseline.py` 等做了参数补充相关的简单修改，均在代码中加以注释

其他文件来自初始所给的文件框架





## 训练好的模型

请在https://jbox.sjtu.edu.cn/l/G1MhQU 中的`ckpt`文件夹中下载，内含

- 最好性能的多个不同模型的参数
- 测试不同weight的co-interative transformer模型



## 数据分析

```
python scripts/analyze/data_analyze.py 
```



## 数据增强

### 获取train set中的value

```
# 运行命令 
  python scripts/analyze/slot_split_data.py 
```





## 模型测试

### 测试不同学习率下的baseline\CRF_LSTM\Pure_BERT\CRF_LSTM_BERT

```
# 在scripts/test_hyperparameter/test_lr.py中修改参数
# [CHANGE] 修改为希望测试的模型
  model = test_model[3]
```

```
# 运行命令
  python scripts/test_hyperparameter/test_lr.py
```

> :bulb: 如果想要进行训练，把下面的command中的 --testing去掉即可



### 测试不同aug ratio下进行数据增强的baseline\CRF_LSTM\Pure_BERT\CRF_LSTM_BERT

```
# 在scripts/test_hyperparameter/test_aug_ratio.py中修改参数
# [CHANGE] 修改为希望测试的模型
  model = test_model[3]
```

```
# 运行命令
  python scripts/test_hyperparameter/test_aug_ratio.py
```

> :bulb:如果想要进行训练，把下面的command中的 --testing去掉即可



### 测试不同相似度计算方式下进行后纠错的CRF_LSTM\Pure_BERT

```
# 在scripts/test_hyperparameter/test_dis_pinyin.py中修改参数
# [CHANGE] 修改为希望测试的模型
  model = test_model[0]
```

```
# 运行命令
  python scripts/test_hyperparameter/test_dis_pinyin.py
```





### 测试Co-Interactive Model

方法见`co-interative tranformer/README.md`



### 测试MLWA

方法见`MLWA-Chinese-SLU-baseline-main/README.md`
