
import subprocess

test_model = ["slu_baseline","CRF_LSTM","Pure_BERT","CRF_LSTM_BERT"]

test_lr = ["1e-3","1e-4","1e-5"]

model = test_model[3]
for lr in test_lr :
    print("model:" ,model, "lr",lr)
    command = f"/bin/python scripts/{model}.py --lr {lr}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"命令运行失败：{e}")
    