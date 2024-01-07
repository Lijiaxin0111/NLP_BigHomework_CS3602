# import jieba
# from gensim.models import Word2Vec
# from sklearn.metrics.pairwise import cosine_similarity

# # 示例数据：假设你有一个中文词库
# corpus = [
#     '人工智能',
#     '机器学习',
#     '深度学习',
#     '自然语言处理',
#     '数据科学',
#     # 其他词汇...
# ]

# # 分词
# tokenized_corpus = [list(jieba.cut(sentence)) for sentence in corpus]

# # 训练Word2Vec模型
# model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# # 计算相似度
# def get_most_similar(word, model):
#     try:
#         word_vector = model.wv[word]
#         similarities = [(other_word, cosine_similarity([word_vector], [model.wv[other_word]])[0][0]) for other_word in model.wv.index_to_key]
#         similarities.sort(key=lambda x: x[1], reverse=True)
#         return similarities
#     except KeyError:
#         return []

# # 示例：找到与'人工智能'相似度最高的单词
# target_word = '人工智能'
# similar_words = get_most_similar(target_word, model)

# # 输出相似度最高的前几个单词
# top_n = 5
# print(f"与'{target_word}'相似度最高的前{top_n}个单词:")
# for word, similarity in similar_words[:top_n]:
#     print(f"{word}: {similarity}")


# ----------------------------

# import jieba
# import difflib

# # 示例数据：假设你有一个中文词库
# corpus = [
#     ['人工智能'],
#     ['机器学习'],
#     ['深度学习'],
#     ['自然语言处理'],
#     ['数据科学'],
#     # 其他词汇...
# ]

# # 分词
# # tokenized_corpus = [list(jieba.cut(sentence)) for sentence in corpus]
# tokenized_corpus = corpus
# print(tokenized_corpus)

# # 输入的目标词
# target_word = '自然语言处理'

# # 找到与目标词最接近的词
# most_similar_word = difflib.get_close_matches(target_word, tokenized_corpus, n=1, cutoff=0.8)
# print(most_similar_word)
# if most_similar_word:
#     most_similar_word = most_similar_word[0]
#     print(f"与'{target_word}'最接近的词是'{most_similar_word}'")
# else:
#     print(f"无法找到与'{target_word}'接近的词")


# ------------------------

from pypinyin import pinyin, lazy_pinyin
from Levenshtein import distance
from multiprocessing import Pool

def get_pinyin(word):
    # 使用pinyin函数获取带声调的拼音列表
    pinyin_list = lazy_pinyin(word)
    # print(pinyin_list)
    # 将带声调的拼音列表转换为不带声调的拼音列表
    pinyin_without_tone = "".join([''.join(item) for item in pinyin_list])



    return pinyin_without_tone

def build_index(word_list):
    index = {}
    for word in word_list:
        pinyin_key = get_pinyin(word)
        
        if pinyin_key in index:
            index[pinyin_key].append(word)
        else:
            index[pinyin_key] = [word]
    return index

def find_most_similar_word_parallel(target_word, index):
    target_pinyin = get_pinyin(target_word)
    min_distance = float('inf')
    most_similar_word = None
    

    for word in index.get(target_pinyin, index):
        current_distance = distance(target_pinyin, get_pinyin(word))
        print(word , ":" , target_word, "-" , current_distance )
        if current_distance < min_distance:
            min_distance = current_distance
            most_similar_word = word

    return most_similar_word

if __name__ == "__main__":
    word_list = ["苹果", "香蕉", "橙子", "西瓜", "人工智能"]
    target_word = "人工"

    # 构建索引
    index = build_index(word_list)

    # 使用多进程进行并行处理
    with Pool() as pool:
        result = pool.apply(find_most_similar_word_parallel, args=(get_pinyin(target_word), index))

    print(f"与 '{target_word}' 读音最相似的词是 '{result}'")

