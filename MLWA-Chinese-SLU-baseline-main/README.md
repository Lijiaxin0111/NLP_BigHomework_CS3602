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

train: run
> python train.py -ced 128 -wed 128 -ehd 512

已保存的模型中，debug对应的是学习率lr=1e-3
参数设置，可以对照train.py中的参数列表选择是否使用Bert，调整学习率，embedding维数等。\\
test: remove the train process and run
> python train.py -ced 128 -wed 128 -ehd 512
