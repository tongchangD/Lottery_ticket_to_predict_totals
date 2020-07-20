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


def main_total(model_Key):
    history_sum={}
    # res_pre={}
    for key in model_Key.keys():
        for model_path in model_Key[key]:
            parser = argparse.ArgumentParser()
            parser.add_argument('-premodels', default=True)  # 是否加载原来的权重 和 vecab
            parser.add_argument('-load_weights', default="weights_"+key)  # 如果加载预训练的权重，把路径到文件夹，以前的权重和泡菜保存
            parser.add_argument('-premodels_path', default="model_weights_"+model_path)  # 预训练的模型文件名
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
            # assert opt.k > 0
            # assert opt.max_len > 10 #　不需要断言最长长度
            # print(opt.load_weights,opt.premodels_path)
            SRC, TRG = create_fields(opt)
            model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
            lis = [
                '01 10 07 11 06', '10 08 06 07 03', '07 04 02 11 03', '06 03 08 02 04', '02 04 03 10 08',
                '10 05 09 07 04', '01 07 10 04 03', '11 03 06 02 04', '03 02 11 06 04', '01 09 05 04 10',
                '08 05 03 01 10', '11 09 07 06 01', '02 11 05 06 04', '09 05 04 03 11', '07 05 11 09 03',
                '05 10 03 09 04', '06 03 05 07 02', '05 06 03 10 02', '05 07 09 06 04', '05 09 08 03 07',
                '07 10 06 08 09', '11 08 09 06 02', '08 06 07 10 04', '09 07 02 05 01', '03 07 08 01 11',
                '04 02 08 06 01', '07 10 11 04 02', '11 02 07 03 08', '07 06 11 05 09', '05 01 10 04 06',
                '03 06 02 08 04', '06 10 07 08 03', '03 05 06 08 10', '11 05 01 08 02', '01 11 10 06 08',
                '10 04 02 08 09', '10 11 02 08 04', '03 02 01 09 05', '05 09 08 04 06', '10 02 11 06 03',

                '08 03 04 01 06', '11 04 02 09 06', '06 04 11 10 08', '08 11 03 05 04', '11 01 04 02 05',
                '05 09 01 04 06', '10 07 01 09 06', '10 08 09 01 11', '03 04 06 02 01', '02 10 08 09 04',
                '11 09 03 02 06', '04 02 06 01 09', '02 11 10 06 09', '10 01 11 09 04', '11 05 06 08 02',
                '02 07 03 08 11', '11 06 03 09 01', '01 04 11 05 03', '01 06 04 11 02', '03 08 02 09 04',
                '05 07 03 01 10', '06 08 09 02 05', '09 02 08 06 11', '06 08 02 03 01', '03 10 02 09 11',
                '11 05 08 06 09', '06 03 07 01 10', '08 04 06 01 07', '05 08 07 03 11', '03 10 05 02 09',
                '02 03 05 09 04', '07 04 01 03 06', '07 10 09 01 02', '03 01 05 02 09', '07 01 02 11 09',
                '11 07 01 03 06', '07 11 08 01 09', '11 01 08 07 06', '08 11 04 05 07', '01 09 06 10 07',

                '02 05 06 01 11', '06 10 11 02 08', '09 06 01 08 05', '06 05 02 10 08', '03 01 06 09 10',
                '11 09 06 04 07', '07 08 06 11 03', '07 03 11 04 02', '06 05 10 09 02', '08 04 10 05 11',
                '06 04 08 03 05', '01 07 06 11 04', '04 06 10 02 07', '07 05 04 01 06', '04 09 07 06 05',
                '01 07 10 09 06', '01 11 05 07 09', '05 04 11 09 07', '11 05 01 10 03', '07 02 11 05 01',
                '04 05 09 07 08', '03 01 09 10 07', '06 05 08 02 04', '04 09 10 11 05', '08 06 05 09 04',
                '01 06 10 03 04', '11 09 10 01 08', '01 04 10 03 02', '04 09 10 11 05', '09 03 10 02 07',
                '08 02 06 11 07', '09 06 10 01 02', '06 11 02 08 03', '10 03 04 09 07', '05 01 02 11 09',
                '04 10 07 08 11', '08 05 01 04 06', '06 09 10 02 07', '10 07 06 03 01', '10 09 11 07 02',

                '09 11 04 06 08', '06 04 07 08 10', '07 06 08 10 11', '09 08 03 10 04', '07 02 11 10 06',
                '04 08 09 05 10', '04 02 08 03 07', '01 02 07 10 09', '01 07 09 03 08', '11 04 01 06 03',
                '02 08 03 01 11', '02 01 06 04 03', '05 03 09 01 07', '05 03 08 04 07', '01 03 10 09 02',
                '10 02 05 04 08', '05 02 06 11 07', '01 04 07 09 10', '03 10 05 06 07', '09 03 05 07 10',
                '01 05 08 04 03', '09 05 08 03 11', '05 08 02 04 01', '03 11 06 01 04', '10 04 06 11 09',
                '07 05 08 11 10', '08 04 03 07 01', '04 07 10 06 02', '03 07 02 11 09', '03 08 11 07 01',
                '02 05 10 09 01', '05 06 08 09 11', '07 05 10 09 06', '07 01 10 08 06', '01 11 05 04 10',
                '09 07 08 06 04', '01 11 08 03 05', '01 07 05 08 04', '07 02 08 04 05'

                # '07 03 06 10 09', '08 04 03 11 02', '04 07 03 11 09', '03 02 07 01 04', '04 03 09 01 02',
                # '01 02 05 08 10', '07 10 05 03 01', '08 05 03 10 04', '02 01 06 03 04', '02 04 10 05 11',
                # '04 06 03 05 10', '06 01 05 02 10', '09 02 06 03 01', '08 09 01 03 07', '03 02 10 05 07',
                # '10 08 09 06 07', '03 06 08 10 11', '10 08 06 02 01'
            ]
            num = config.history
            for i, line in enumerate(lis):
                if i >= num - 1:
                    if config.tag:
                        opt.text = " <tag> ".join(lis[i - num + 1:i + 1])
                    else:
                        opt.text = " ".join(lis[i - num + 1:i + 1])
                    result_phrase = translate_one_sentence(opt, model, SRC, TRG)  # 预测值
                    if i+1 not in history_sum.keys():  # 统计第i+1期预测的值
                        history_sum[i+1]=[result_phrase]
                    else:
                        history_sum[i+1]+=[result_phrase]
                # if i != len(lis) - 1:
                #     res_pre[i+1]=lis[i+1]
    print("history_sum",history_sum)
    total2=[]
    total3=[]
    for i in range(3,len(lis)):
        # history_sum[i] # 预测的第i期 和 lis[i] 实际值 比较
        sorting =count_([word for words in history_sum[i] for word in words.split(" ")])

        # similar2=len(list(set(lis[i].split(" ")).intersection(set([a for a,b in sorting[:3]]))))  # 获取排列前三的数据
        # similar2=len(list(set(lis[i].split(" ")).intersection(set([a for a,b in sorting[-2:]+sorting[:1] ]))))  # 获取排列前三的数据
        # similar3=len(list(set(lis[i].split(" ")).intersection(set([a for a,b in sorting[-2:]+sorting[:2] ]))))  # 获取排列前四的数据
        similar2=len(list(set(lis[i].split(" ")).intersection(set([a for a,b in sorting[:4] ]))))  # 获取排列前三的数据
        similar3=len(list(set(lis[i].split(" ")).intersection(set([a for a,b in sorting[:5] ]))))  # 获取排列前四的数据

        if similar2 < 2:
            total2.append(-6)
        elif similar2==2:
            total2.append(0)
        elif similar2==3:
            total2.append(12)

        if similar3<=2:
            total3.append(-8)
        elif similar3==3:
            total3.append(11)
        elif similar3==4:
            total3.append(68)
    print("2",total2)
    sum2=0
    for i in total2:
        sum2+=i
    print("任二 一共投注 %d 期 最终结果 输赢值 为 %d"%(len(lis)-1,sum2))
    print("3",total3)
    sum3=0
    for i in total3:
        sum3+=i
    print("任三 一共投注 %d 期 最终结果 输赢值 为 %d"%(len(lis)-1,sum3))

    print("预测最后一期",count_([word for words in history_sum[len(lis)] for word in words.split(" ")]))  # 预测的最后一期


def count_(word_list):
    count_dict = {}
    for item in word_list:
        count_dict[item] = count_dict[item] + 1 if item in count_dict else 1
    return sorted(count_dict.items(), key=lambda x: x[1], reverse=True)

def main_simple(model_Key):


    lis = [
        '01 10 07 11 06', '10 08 06 07 03', '07 04 02 11 03', '06 03 08 02 04', '02 04 03 10 08',
        '10 05 09 07 04', '01 07 10 04 03', '11 03 06 02 04', '03 02 11 06 04', '01 09 05 04 10',
        '08 05 03 01 10', '11 09 07 06 01', '02 11 05 06 04', '09 05 04 03 11', '07 05 11 09 03',
        '05 10 03 09 04', '06 03 05 07 02', '05 06 03 10 02', '05 07 09 06 04', '05 09 08 03 07',
        '07 10 06 08 09', '11 08 09 06 02', '08 06 07 10 04', '09 07 02 05 01', '03 07 08 01 11',
        '04 02 08 06 01', '07 10 11 04 02', '11 02 07 03 08', '07 06 11 05 09', '05 01 10 04 06',
        '03 06 02 08 04', '06 10 07 08 03', '03 05 06 08 10', '11 05 01 08 02', '01 11 10 06 08',
        '10 04 02 08 09', '10 11 02 08 04', '03 02 01 09 05', '05 09 08 04 06', '10 02 11 06 03',

        '08 03 04 01 06', '11 04 02 09 06', '06 04 11 10 08', '08 11 03 05 04', '11 01 04 02 05',
        '05 09 01 04 06', '10 07 01 09 06', '10 08 09 01 11', '03 04 06 02 01', '02 10 08 09 04',
        '11 09 03 02 06', '04 02 06 01 09', '02 11 10 06 09', '10 01 11 09 04', '11 05 06 08 02',
        '02 07 03 08 11', '11 06 03 09 01', '01 04 11 05 03', '01 06 04 11 02', '03 08 02 09 04',
        '05 07 03 01 10', '06 08 09 02 05', '09 02 08 06 11', '06 08 02 03 01', '03 10 02 09 11',
        '11 05 08 06 09', '06 03 07 01 10', '08 04 06 01 07', '05 08 07 03 11', '03 10 05 02 09',
        '02 03 05 09 04', '07 04 01 03 06', '07 10 09 01 02', '03 01 05 02 09', '07 01 02 11 09',
        '11 07 01 03 06', '07 11 08 01 09', '11 01 08 07 06', '08 11 04 05 07', '01 09 06 10 07',

        '02 05 06 01 11', '06 10 11 02 08', '09 06 01 08 05', '06 05 02 10 08', '03 01 06 09 10',
        '11 09 06 04 07', '07 08 06 11 03', '07 03 11 04 02', '06 05 10 09 02', '08 04 10 05 11',
        '06 04 08 03 05', '01 07 06 11 04', '04 06 10 02 07', '07 05 04 01 06', '04 09 07 06 05',
        '01 07 10 09 06', '01 11 05 07 09', '05 04 11 09 07', '11 05 01 10 03', '07 02 11 05 01',
        '04 05 09 07 08', '03 01 09 10 07', '06 05 08 02 04', '04 09 10 11 05', '08 06 05 09 04',
        '01 06 10 03 04', '11 09 10 01 08', '01 04 10 03 02', '04 09 10 11 05', '09 03 10 02 07',
        '08 02 06 11 07', '09 06 10 01 02', '06 11 02 08 03', '10 03 04 09 07', '05 01 02 11 09',
        '04 10 07 08 11', '08 05 01 04 06', '06 09 10 02 07', '10 07 06 03 01', '10 09 11 07 02',

        '09 11 04 06 08', '06 04 07 08 10', '07 06 08 10 11', '09 08 03 10 04', '07 02 11 10 06',
        '04 08 09 05 10', '04 02 08 03 07', '01 02 07 10 09', '01 07 09 03 08', '11 04 01 06 03',
        '02 08 03 01 11', '02 01 06 04 03', '05 03 09 01 07', '05 03 08 04 07', '01 03 10 09 02',
        '10 02 05 04 08', '05 02 06 11 07', '01 04 07 09 10', '03 10 05 06 07', '09 03 05 07 10',
        '01 05 08 04 03', '09 05 08 03 11', '05 08 02 04 01', '03 11 06 01 04', '10 04 06 11 09',
        '07 05 08 11 10', '08 04 03 07 01', '04 07 10 06 02', '03 07 02 11 09', '03 08 11 07 01',
        '02 05 10 09 01', '05 06 08 09 11', '07 05 10 09 06', '07 01 10 08 06', '01 11 05 04 10',
        '09 07 08 06 04', '01 11 08 03 05', '01 07 05 08 04', '07 02 08 04 05'
    ]
    if config.history == len(lis):
        pres = []
        press = []
        for key in model_Key.keys():
            for model_path in model_Key[key]:
                parser = argparse.ArgumentParser()
                parser.add_argument('-premodels', default=True)  # 是否加载原来的权重 和 vecab
                parser.add_argument('-load_weights', default="weights_"+key)  # 如果加载预训练的权重，把路径到文件夹，以前的权重和泡菜保存
                parser.add_argument('-premodels_path', default="model_weights_"+model_path)  # 预训练的模型文件名
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
                # assert opt.k > 0
                # assert opt.max_len > 10 #　不需要断言最长长度
                # print(opt.load_weights,opt.premodels_path)
                SRC, TRG = create_fields(opt)
                model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
                if config.tag:
                    opt.text = " <tag> ".join(lis)
                else:
                    opt.text = " ".join(lis)
                result_phrase = translate_one_sentence(opt, model, SRC, TRG)  # 预测值
                pres.append(result_phrase)
                press += result_phrase.split(" ")
        print(pres)
        # print("press",len(press),press)
        print(count_(press))

    else:
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
                # assert opt.k > 0
                # assert opt.max_len > 10 #　不需要断言最长长度
                # print(opt.load_weights,opt.premodels_path)
                SRC, TRG = create_fields(opt)
                model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
                num=config.history
                sum=[]
                pres = []
                for i, line in enumerate(lis):
                    if i >= num - 1:
                        if config.tag:
                            opt.text = " <tag> ".join(lis[i - num + 1:i + 1])
                        else:
                            opt.text = " ".join(lis[i - num + 1:i + 1])
                        result_phrase = translate_one_sentence(opt, model, SRC, TRG)
                        if i != len(lis) - 1:
                            similar_values = set(result_phrase.split(" ")).intersection(set(lis[i + 1].split(" ")))
                            sum.append(len(similar_values))
                        pres.append(result_phrase)
                        # press += result_phrase.split(" ")
                total = 0
                for i in sum:
                    total += i
                print(model_path,len(sum), total, total / len(sum), sum)
                print(model_path,pres)

if __name__ == '__main__':
    tic = datetime.datetime.now()
    model_Key={
        # "one":"35 50 78 88 56 58 66 74 70".split(" "),
        # "3":"20 59 80 90 100".split(" "),
        # "after_two_mini": "50".split(" "),
        # "three":"24 52 86 60".split(" "),
        # "after_two": "56 60 64 80 94".split(" "),
        # "5": "30 94 100 24".split(" "),  # three +5  四打三还行 考虑 周期 ？
        "20200719":"7 10 17 20 27 30 37 40 47 50 57 60 67 70 77 80 87 90 97 100".split(" "),
        }
    main_total(model_Key)  # 统计整体的
    # main_simple(model_Key)  # 直接单独直接预测 默认前三期预测
    end = datetime.datetime.now()
    print("Answer a question in %s seconds" % (end - tic))

