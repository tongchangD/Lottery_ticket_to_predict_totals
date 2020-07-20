from data_loader import Language
from inference import inference
import re

CHINESE1 = Language(vocab_file="./couplet2/vocabs")



def chat_couplet(in_str,MODEL_DIR,codersum):
    # if len(in_str) == 0 or len(in_str.split(" ")) > config.MAX_LENGTH:
    #     output = u'您的输入太长了'
    # else:
    #     output = inference(in_str,  language=CHINESE,MODEL_DIR=MODEL_DIR,codersum=codersum)
    output = inference(in_str, language=CHINESE1, MODEL_DIR=MODEL_DIR, codersum=codersum)
    output_lis=[]
    for i,word  in enumerate(output.replace("<tag>","")):
        if i%2==1:
            output_lis.append(str(output[i-1])+str(output[i]))

    return " ".join(output_lis)

# "",

predicts=[]
lis=[
"03 02 07 09 11",
"01 05 11 04 10",
"08 04 05 01 11",
"10 03 02 09 08",
"07 09 11 05 04",
"01 10 03 09 06",
"11 06 09 08 01",
"10 05 11 06 04",



]
"",
# "",
# codersumlis=["400","2400","12000"]
# for codersum in codersumlis:
#     predict=[]
#     for i, line in enumerate(lis):
#         pre=chat_couplet(" ".join(line.split(" ")[:3]),"./model_save113", codersum)
#         if i !=len(lis)-1:
#             print(codersum," "*35+str(len(list(set(lis[i+1].split(" ")).intersection(set(pre.split(" ")))))))
#         predict.append(pre)
#     predicts.append(predict)
# pre4=[]
# for i, line in enumerate(lis):
#     pre = chat_couplet(line, "./model_save114", "4800")
#     if i != len(lis) - 1:
#         print(" " * 40 + str(len(list(set(lis[i + 1].split(" ")).intersection(set(pre.split(" ")))))))
#     pre4.append(pre)
# print("预测",pre4)
# print("实际",lis[1:])
# 前一条预测下一条
# pre5=[]
# for i, line in enumerate(lis):
#     pre = chat_couplet(line, "./model_save115", "400")
#     if i != len(lis) - 1:
#         print(" " * 40 + str(len(list(set(lis[i + 1].split(" ")).intersection(set(pre.split(" ")))))))
#     pre5.append(pre)
# print("预测",pre5)
# print("实际",lis[1:])
# print()
# 前五条预测下一条

#
# num=2  # 前几个预测下一个
# pres=[]
# print("len(lis)",len(lis))
# for i, line in enumerate(lis):
#     if i>=num:
#         pre = chat_couplet(" <tag> ".join(lis[i-num:i+1]),"./model_save1130501", "1800")
#         if i != len(lis) - 1:
#             print(" " * 20 + str(len(list(set(lis[i+1].split(" ")).intersection(set(pre.split(" ")))))))
#         # print("pre",pre)
#         pres.append(pre)
# print("预测",pres[-3:])
# print("实际",lis[-2:])
# print()

# need
def accquiresource(line):
    '''
    和值 15-45 31
    跨度 04-10 7
    单双 0:5-5:0  6
    :param line:
    :return:
    '''
    Sums={15:"r",16:"A",17:"B",18:"C",19:"D",20:"E",21:"F",22:"G",23:"H",24:"I",25:"J",
          26:"K",27:"L",28:"M",29:"N",30:"O",31:"P",32:"Q",33:"R",34:"S",35:"T",
          36:"U",37:"V",38:"W",39:"X",40:"Y",41:"Z",42:"a",43:"b",44:"c",45:"d"}  # 和值
    Span={4:"e",5:"f",6:"g",7:"h",8:"i",9:"j",10:"k"}  # 跨度
    Parity={0:"l",1:"m",2:"n",3:"o",4:"p",5:"q"}  # 奇偶
    # 46:"",47:"",48:"",49:"",50:"",51:"",52:"",53:"",54:"",55:"",
    #           56:"",57:"",58:"",59:"",60:"",61:"",:"",:"",:"",:"",:"",:"",:"",:"",
    result=[]
    sumvalue=0
    sumspan=0
    Paritys=[]
    for value in line.split(" "):
        sumvalue+=int(value)
        if int(value)%2==0:
            sumspan+=1
        result.append(value)
        Paritys.append(int(value))

    result.append(Sums[sumvalue])
    result.append(Span[max(Paritys)-min(Paritys)])
    result.append(Parity[sumspan])

    return " ".join(result)


pres=[]
CHINESE2 = Language(vocab_file="./couplet/vocabs")
print("len(lis)",len(lis))
for i, line in enumerate(lis):
    output = inference(accquiresource(lis[i]), language=CHINESE2, MODEL_DIR="./model_save113", codersum="1500")
    output_lis = re.findall(r'.{2}', output.replace("<tag>", ""))
    pre=" ".join(output_lis)
    if i != len(lis) - 1:
        print(" "* 20+str(len(list(set(lis[i+1].split(" ")).intersection(set(pre.split(" ")))))))
    pres.append(pre)
print("预测",pres[-3:])
print("实际",lis[-2:])
print()
# pre6=[]
# for i, line in enumerate(lis):
#     pre = chat_couplet(line, "./model_save116", "800")
#     if i != len(lis) - 1:
#         print(" " * 40 + str(len(list(set(lis[i + 1].split(" ")).intersection(set(pre.split(" ")))))))
#     pre6.append(pre)
# print("预测",pre6)
# print("实际",lis[1:])


# for pre3 in predicts:
#     print("pre3",pre3)
# before={}
# nums=[]
# for num in pre4[-2].split(" "):
#     if num not in before.keys():
#         before[num]=1
#     else:before[num]+=1
#     nums.append(num)
# for num in pre5[-2].split(" "):
#     if num not in before.keys():
#         before[num] = 1
#     else:
#         before[num] += 1
#     nums.append(num)
# for num in pre6[-2].split(" "):
#     if num not in before.keys():
#         before[num] = 1
#     else:
#         before[num] += 1
#     nums.append(num)
# print("before",len(before))
# print("before 差值",set(lis[-1].split(" ")).difference(set(nums)))
# res={}

# for num in pre4[-1].split(" "):
#     if num not in res.keys():
#         res[num]=1
#     else:res[num]+=1
# for num in pre5[-1].split(" "):
#     if num not in res.keys():
#         res[num] = 1
#     else:
#         res[num] += 1
# for num in pre6[-1].split(" "):
#     if num not in res.keys():
#         res[num] = 1
#     else:
#         res[num] += 1
# print("res",len(res))
# for key,value in res.items():
#     print(key,value)
#
# config
# f=open("txt","r")
# fl=f.readlines()
# f.close()
# sum1=0
# lis=[]
# for i in fl:
#     if not i.startswith("#"):
#        lis.append(i.strip())
# for i,line in enumerate(lis):
#     if i !=len(lis)-1:
#         predict=chat_couplet(line)
#         print(" "*40+str(len(list(set(lis[i+1].split(" ")) & set(predict.split(" "))))))
#         if len(list(set(lis[i+1].split(" ")) & set(predict.split(" "))))>=4:
#             sum1+=1
#             print("第%d行"%i)
#     else:
#         predict=chat_couplet(line)
# print (len(lis),sum1)
