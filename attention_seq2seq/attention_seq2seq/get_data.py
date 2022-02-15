import jieba
import tensorflow as tf
from tqdm import tqdm

question_list = [];
answer_list = []
question_vocab = set();
answer_vocab = set()

flag = True
count = 0
max_count = 500
with open("data/xiaohuangji50w_nofenci.conv", errors="ignore", encoding="UTF-8") as f:
    context = f.readlines()
    for line in context:
        if count >= max_count:
            break
        if line.startswith('E'):
            pass
        if line.startswith('M'):
            line = jieba.lcut(line.replace('M', '').strip())
            if flag:
                question = ["GO"] + line + ["END"]
                for item in question:
                    question_vocab.add(item)
                question_list.append(line)
                flag = False
            else:
                answer = ["GO"] + line + ["END"]
                for item in answer:
                    answer_vocab.add(item)
                answer_list.append(line)
                count += 1
                flag = True

question_vocab = ["PAD"] + list(sorted(question_vocab))
answer_vocab = ["PAD"] + list(sorted(answer_vocab))


# 这里截取一部分数据


def get_dataset():
    question_tokens_ids = []
    answer_tokens_ids = []
    for question, answer in zip(tqdm(question_list), answer_list):
        question_tokens_ids.append([question_vocab.index(char) for char in question])
        answer_tokens_ids.append([answer_vocab.index(char) for char in answer])
    return question_vocab, answer_vocab, question_tokens_ids, answer_tokens_ids


print(question_list[1])
print(answer_list[1])
question_vocab, answer_vocab, question_tokens_ids, answer_tokens_ids = get_dataset()
print(question_tokens_ids[1])
print([question_vocab[index] for index in question_tokens_ids[1]])
print(answer_tokens_ids[1])
print([answer_vocab[index] for index in answer_tokens_ids[1]])

