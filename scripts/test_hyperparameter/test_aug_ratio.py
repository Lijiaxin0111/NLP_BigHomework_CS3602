
import subprocess

test_model = ["slu_baseline.py --lr 1e-3","CRF_LSTM.py  --lr 1e-3 ","Pure_BERT.py --lr 1e-5 ","CRF_LSTM_BERT.py --lr 1e-5"]

test_aug_ratio = ["0.3","0.5","0.7", "1"]

model = test_model[0]
for aug_ratio in test_aug_ratio :
    print("model:" ,model, "aug_ratio",aug_ratio)
    command = f"/bin/python scripts/{model} --aug_ratio {aug_ratio}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"命令运行失败：{e}")
    