# README



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

### 

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
# 环境配置
pip install pypinyin
```

```
# 在scripts/test_hyperparameter/test_dis_pinyin.py中修改参数
# [CHANGE] 修改为希望测试的模型
  model = test_model[0]
```

```
# 运行命令
  python scripts/test_hyperparameter/test_dis_pinyin.py
```





