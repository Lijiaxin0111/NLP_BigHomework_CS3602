env: python 3.7.6
> - torch1.10.0+cu11.1
> - transformers==2.4.1
> - numpy==1.18.1
> - tqdm==4.42.1
> - seqeval==0.0.12
> - ordered_set
> - matplotlib
> - tensorboardX
> - jieba

## 数据集
数据存放在./data/Project_data中

## 参数设置
在./train.py中可以设置数据集路径，模型存储路径以及模型参数
<br/>
在./test.py中可以设置数据集路径，模型路径
## 运行
train: run
> python train.py -ced 128 -wed 128 -ehd 512

test: run
> python test.py -ced 128 -wed 128 -ehd 512
<br/>
当前的Tensorboard日志保存在./logs中
