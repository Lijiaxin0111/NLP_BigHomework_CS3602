import subprocess

test_model = ["CRF_LSTM_premodified.py  --lr 1e-3 --aug_ratio 0.3 ","Pure_BERT_premodified.py --lr 1e-5 --aug_ratio 0.3"]

test_pinyin = [True,False]
test_dis = ["jac", 'lev']

model = test_model[0]
for pinyin in test_pinyin :
    for dis in test_dis:
        print("model:" ,model, "pinyin: ", pinyin , 'test_dis: ', dis)
        if pinyin: 
            command = f"/ssdisk4/condaenvs/jiaxin/envs/oakink/bin/python scripts/{model} --dis {dis} --pinyin"
        else:
            command = f"/ssdisk4/condaenvs/jiaxin/envs/oakink/bin/python scripts/{model} --dis {dis}"
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"命令运行失败：{e}")