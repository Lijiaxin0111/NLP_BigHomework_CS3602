
from utils.example import Example
import json 
import os
import random
import re

'''
 数据增强：对出现在slot中的值，在ontology寻找相同slot的近义词进行替换

 - poi\终点\起点 名称\目标\修饰 :   基于train_split中的数据进行随机替换
 (考虑在./lexicon/poi_name.txt随机选词进行替换,可能导致数据development的数据泄露)

 - 请求类型: 除了定位之外的词, 任意替换
 - 路线偏好： 任意替换
 - 序列号: 对数字随机替换
 - 页码： 相互替换
 

 - 操作： 暂不替换
 - 对象： 暂不替换
 
'''



AUGMENT_SLOT = ["请求类型","路线偏好","序列号","页码",
                "poi名称","poi修饰","poi目标","起点名称","起点修饰","起点目标","终点名称","终点修饰","终点目标","途经点名称"]
data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) , "data" )


with open(os.path.join(data_root, "ontology.json"), "rb",) as onto_file:
    ontology_slot = json.load(onto_file)["slots"]

with open(os.path.join(  os.path.join(data_root, "lexicon"),"poi_name.txt"), "rt",encoding='UTF-8') as name_file:
    names = name_file.readlines()

with open(os.path.join(  os.path.join(data_root, "lexicon"),"train_split.json"), "rb") as split_flie:
    splits = json.load(split_flie)

xiushi_keys = [key for key in splits.keys()]



def get_new_num(input_string):
    # 定义正则表达式匹配模式
    pattern = r'([一二三四五六七八九])'

    # 使用正则表达式找到匹配的数字
    matches = re.findall(pattern, input_string)

    # 将匹配的数字进行随机替换
    for match in matches:
        # 获取替换的数字
        replacements = ['一', '二', '三', '四', '五', '六', '七', '八', '九']
        replacements.remove(match)  # 移除原数字，避免重复替换

    replacement = random.choice(replacements)
    return replacement, match

     


def replace_masr(dict_utt , i, old, new ):
    # i 是待replace的语义的索引
    dict_utt["manual_transcript"] = dict_utt["manual_transcript"].replace(old, new)
    dict_utt["asr_1best"] = dict_utt["asr_1best"].replace(old, new)

    dict_utt["semantic"][i][2] = dict_utt["semantic"][i][2].replace(old, new)

def get_new_str(slot_idx, old_value):

    if slot_idx[0] == "页码" or slot_idx[0] == "路线偏好" or slot_idx[0] == "请求类型":
        res_values = ontology_slot[slot_idx[0]].copy()
     
        res_values.remove(old_value)
        if slot_idx[0] == "请求类型":
            
            res_values.remove("定位")
        return random.choice(res_values)



    if  slot_idx[0] == "序列号":
        return get_new_num(old_value)
    
    if slot_idx[0] in  [ "poi修饰","起点修饰","终点修饰"]:
        xiushi_key = random.choice(xiushi_keys)
        

        return random.choice(splits[xiushi_key])
    
    if  slot_idx[0]  in [ "poi名称","poi修饰","poi目标","起点名称","起点修饰","起点目标","终点名称","终点修饰","终点目标","途经点名称"]:
        
        return  random.choice( splits[slot_idx[0]])
    
    




def data_augment_example(example , aug_ratio = 0):

    slot_idxs =  [(example.ex["semantic"][i][1], i) for i in range(len(example.ex["semantic"])) if example.ex["semantic"][i][1] in AUGMENT_SLOT and  example.ex["semantic"][i][2] != "定位" ]
    # print([(example.ex["semantic"][i][2], i) for i in range(len(example.ex["semantic"])) ])

    augment_ratio = aug_ratio
    
    tmp = random.random()


    if len(slot_idxs) == 0 or tmp > augment_ratio:
        return example
    
    augment_slot = random.sample(slot_idxs, k = 1 )[0]


    new_value = get_new_str(augment_slot, example.ex["semantic"][augment_slot[1]][2])
   

    dict_utt = example.ex

    if augment_slot[0] == "序列号":

        replace_masr(dict_utt, augment_slot[1], new_value[1], new_value[0])

    else:
        replace_masr(dict_utt, augment_slot[1], example.ex["semantic"][augment_slot[1]][2] , new_value  )
    # print("before " ,example.utt)
    # print("after " , Example(  dict_utt, example.did).utt)

    return Example(  dict_utt, example.did)
    







