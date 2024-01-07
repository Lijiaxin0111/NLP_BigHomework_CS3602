# README for Co-Interactive Model

## 数据集

数据存放在 `./data/slu`

## 参数设置

在 `./data_util/config.py` 下设置数据集路径及模型参数

在训练时，设置`mod = "train"`；在测试时，设置`mod = "test"`

是否使用后纠错：设置 `modify = True / False`

## 运行

```python my_main_joint.py```

模型保存在 `./ckpt/` 下。当前该文件夹下分别保存了lr=1e-3, loss_weight=0.2，采用和未采用后纠错的两个训练好的模型。

TensorBoard日志保存在 `./logs/` 下

当前配置下的模型是性能最佳的