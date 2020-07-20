import torch
from Models import get_model
from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import pdb
import dill as pickle
import datetime
import argparse
from Models import get_model
from Beam import beam_search
# from nltk.corpus import wordnet
from torch.autograd import Variable
import re
import random
from random import shuffle
import config

def multiple_replace(dict, text):  # 格式化标点符号
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


def translate_sentence(sentence, model, opt, SRC, TRG):
    model.eval()
    indexed = []
    sentence = SRC.preprocess(sentence)  # 预处理输入数据
    # print("sentence",sentence)
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 or opt.floyd == True:
            indexed.append(SRC.vocab.stoi[tok])
    sentence = Variable(torch.LongTensor([indexed]))  # 转tensor数据
    # print("sentence",sentence)
    if opt.device == 0:
        sentence = sentence.cuda()
    sentence = beam_search(sentence, model, SRC, TRG, opt)
    if opt.k == 1:
        return multiple_replace({' ?': '?', ' !': '!', ' .': '.', '\' ': '\'', ' ,': ','}, sentence[0])
    else:
        res = []
        for i in sentence:
            print("i", i)
            res += i.split(" ")
        # print("list(set(res))",list(set(res)))
        return multiple_replace({' ?': '?', ' !': '!', ' .': '.', '\' ': '\'', ' ,': ','}, " ".join(list(set(res))))


def translate_paragraph(opt, model, SRC, TRG):
    sentences = opt.text.lower().split('.')  #
    translated = []
    for sentence in sentences:
        translated.append(translate_sentence(sentence, model, opt, SRC, TRG).capitalize())
    return (' '.join(translated))


def translate_one_sentence(opt, model, SRC, TRG):
    sentences = opt.text.lower()
    translated = translate_sentence(sentences, model, opt, SRC, TRG).capitalize()
    return translated


def count_(word_list):
    count_dict = {}
    for item in word_list:
        count_dict[item] = count_dict[item] + 1 if item in count_dict else 1
    return sorted(count_dict.items(), key=lambda x: x[1], reverse=True)

def main_simple(model_Key):
    f = open('./data/data', "r", encoding='utf-8')
    fl = f.readlines()
    f.close()
    src_sentences = []
    src_sen = []
    record = {}
    for i, line in enumerate(fl):
        if line.startswith("第"):
            if config.tag:
                src_sen.append(" ".join(re.findall(r'.{2}', re.sub("\D", "", fl[i + 1]))) + " <tag>")  # 不加 <tag> 每句分界线
            else:
                src_sen.append(" ".join(re.findall(r'.{2}', re.sub("\D", "", fl[i + 1]))))  # 不加 <tag> 每句分界线
            if " ".join(sorted(re.findall(r'.{2}', re.sub("\D", "", fl[i + 1])))) not in record:
                record[" ".join(sorted(re.findall(r'.{2}', re.sub("\D", "", fl[i + 1]))))] = 1
            else:
                record[" ".join(sorted(re.findall(r'.{2}', re.sub("\D", "", fl[i + 1]))))] += 1
        if "#" in line or i == len(fl) - 1:
            src_sen.reverse()
            src_sentences.append(src_sen)
            src_sen = []
    datalis = []
    num = config.history
    for reslis in src_sentences:
        i = 0
        while i < len(reslis):
            if i >= num:
                datalis.append((" ".join(reslis[i - num:i]).rstrip(" <tag>"),
                                " ".join(reslis[i].replace(" <tag>", "").split(" "))))  # 预测前n个 输出值不按大小顺序
                i += 1  # 不平滑 +=1  平滑 +=2
            else:
                i += 1

    res_datalis=[]
    testlis = []
    trainlis = []
    for key in model_Key.keys():
        for model_path in model_Key[key]:
            parser = argparse.ArgumentParser()
            parser.add_argument('-premodels', default=True)  # 是否加载原来的权重 和 vecab
            parser.add_argument('-load_weights', default="weights_" + key)  # 如果加载预训练的权重，把路径到文件夹，以前的权重和泡菜保存
            parser.add_argument('-premodels_path', default="model_weights_" + model_path)  # 预训练的模型文件名
            parser.add_argument('-k', type=int, default=1)  # topK 感觉不需要，设置为1 就OK 或者将其删除
            parser.add_argument('-max_len', type=int, default=32)  # 最长长度 需加上起始位置
            parser.add_argument('-d_model', type=int, default=512)  # 嵌入向量和层的维数(默认为512)
            parser.add_argument('-n_layers', type=int, default=6)  # 在Transformer模型中有多少层(默认=6)
            parser.add_argument('-heads', type=int, default=8)  # 需要分割多少个头部以获得多个头部的注意(默认值为8)
            parser.add_argument('-dropout', type=int, default=0.1)  # 决定多大的dropout将(默认=0.1)
            parser.add_argument('-cuda', default=True, action='store_true')
            parser.add_argument('-floyd', action='store_true')
            opt = parser.parse_args()
            opt.device = 0 if opt.cuda is True else -1
            if opt.device == 0:
                assert torch.cuda.is_available()
            SRC, TRG = create_fields(opt)
            model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
            print("len(datalis)",len(datalis))
            for i,(opt.text,tag) in enumerate(datalis):
                result_phrase = translate_one_sentence(opt, model, SRC, TRG)
                similar_values = set(result_phrase.split(" ")).intersection(set(tag.split(" ")))
                if len(similar_values)>=3:
                    res_datalis.append((opt.text," ".join(similar_values)))
                if i%500==1:
                    print("已经运行至第 %d 条,合计获得 %d 条数据"%(i,len(res_datalis)))
    f = open("./data/in2.txt", "w")
    g = open("./data/out2.txt", "w")
    for i, line in enumerate(res_datalis):
        if i % 10 < 3:
            testlis.append(line)
        else:
            trainlis.append(line)
    shuffle(trainlis)
    for ins, outs in trainlis:
        f.write(ins + "\n")
        g.write(outs + "\n")
    print("合计 %d 条训练集" % len(trainlis))

    shuffle(testlis)
    for ins, outs in testlis:  # np.random.shuffle(trainlis)
        f.write(ins + "\n")
        g.write(outs + "\n")
    print("合计 %d 条测试集" % len(testlis))
    f.close()
    g.close()

if __name__ == '__main__':
    tic = datetime.datetime.now()
    model_Key={
        "5": "30 94 100 24".split(" "),  # three +5  四打三还行 考虑 周期 ？
        }
    main_simple(model_Key)
    end = datetime.datetime.now()
    print("Answer a question in %s seconds" % (end - tic))

