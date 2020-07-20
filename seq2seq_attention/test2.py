from data_loader import Language
from inference import inference
import re
import os
from random import shuffle


def chat_couplet(in_str,MODEL_DIR,codersum):
    # if len(in_str) == 0 or len(in_str.split(" ")) > config.MAX_LENGTH:
    #     output = u'您的输入太长了'
    # else:
    #     output = inference(in_str,  language=CHINESE,MODEL_DIR=MODEL_DIR,codersum=codersum)
    output = inference(in_str, language=CHINESE, MODEL_DIR=MODEL_DIR, codersum=codersum)
    output_lis1=[]
    # print(output)
    Sums = {15: "r", 16: "A", 17: "B", 18: "C", 19: "D", 20: "E", 21: "F", 22: "G", 23: "H", 24: "I", 25: "J",
            26: "K", 27: "L", 28: "M", 29: "N", 30: "O", 31: "P", 32: "Q", 33: "R", 34: "S", 35: "T",
            36: "U", 37: "V", 38: "W", 39: "X", 40: "Y", 41: "Z", 42: "a", 43: "b", 44: "c", 45: "d"}  # 和值
    Span = {4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k"}  # 跨度
    Parity = {0: "l", 1: "m", 2: "n", 3: "o", 4: "p", 5: "q"}  # 奇偶
    ContrarySums=dict(zip(Sums.values(), Sums.keys()))
    ContrarySpan=dict(zip(Span.values(), Span.keys()))
    ContraryParity=dict(zip(Parity.values(), Parity.keys()))
    output_lis = re.findall(r'.{2}',re.sub("\D", "", output.replace("<tag>","")))

    for i in re.findall(r'[A-Za-z]', output.replace("<tag>","")):
        if i in ContrarySums.keys():
            output_lis1.append("和值"+str(ContrarySums[i]))
        if i in ContrarySpan.keys():
            output_lis1.append("跨度" +str(ContrarySpan[i]))

        if i in ContraryParity.keys():
            output_lis1.append("奇数" +str(ContraryParity[i]))

        # output_lis.append(i)
    return " ".join(output_lis)," ".join(output_lis1)

# "",

predicts=[]

f = open("./couplet/data", "r")
fl = f.readlines()
f.close()
fl.reverse()
lis = []
for line in fl:
    if "#" not in line:
        temp=line.strip().split(" ")[1].split("+")
        # temp.sort() # 排序
        lis.append(" ".join(temp) + " <tag>")
"""
采用滑动窗口
"""
datalis=[]
i=0
while i<len(lis):
    if i>=4:
        # data=[]
        # for j in " ".join(lis[i - 4:i - 1]).split(" "):
        #     if j not in data:
        #         data.append(j)
        # datalis.append((" ".join(data),lis[i]))
        datalis.append((" ".join(lis[i - 4:i - 1]).rstrip(" <tag>"), lis[i].replace(" <tag>", "")))  # 预测五个
        i+=1  # 不平滑 +=1  平滑 +=2
    else:
        i+=1
# "",
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


print("len(lis)",len(lis))
PATH="./model_save1130511"
modelspath=[]
for i in os.listdir(PATH):
    if "encoder_" in i:
        modelspath.append(i.replace("encoder_","").replace(".pth",""))
modelspath.sort()
# modelspath=['800', '1800']
# modelspath=["300","600","900","1200","1400"]          # 0507
# modelspath=["1000","1400","1800","2000"]              # 05071
# modelspath=["400","800","1300","1400","1900"]           # 0508
modelspath=["1200"]          # 05081
CHINESE = Language(vocab_file="./couplet/vocabs")
print(modelspath)
num=3  # 前几个预测下一个
trainset=[]
totallis=[]
dadsadasdada=[] # 统计结果预测正确的数字包含哪些
for model in modelspath:
    prenum=0
    sum1 = 0
    prenums=[]
    pres = []
    for source,target in datalis:
        #print("source",source,"target",target)
        pre1,pre2 = chat_couplet(source,PATH, model)
        if i != len(lis) - 1:
            prenum+=len(list(set(target.split(" ")).intersection(set(pre1.split(" ")))))
            prenums.append(len(list(set(target.split(" ")).intersection(set(pre1.split(" "))))))
            dadsadasdada+=list(set(target.split(" ")).intersection(set(pre1.split(" "))))
            if len(list(set(target.split(" ")).intersection(set(pre1.split(" ")))))>=2:
                trainset.append((source," ".join(list(set(target.split(" ")).intersection(set(pre1.split(" ")))))))
                #print ("预测",pre1)
                print("source",source,"target2"," ".join(list(set(target.split(" ")).intersection(set(pre1.split(" "))))))
            else:
                print ("预测",pre1)
                print ("实际",target)
            sum1+=1
        pres.append(pre1+" "+pre2)
    totallis.append(pres[-1].strip())
    print("历史数",prenums)
    #print("预测", pres[-4:])
    """
    if len(lis) == 5:
        print("实际", lis[-1:])
    elif len(lis)==6:
        print("实际", lis[-2:])
    elif len(lis)>=7:
        print("实际", lis[-3:])
    """
    print("历史平值值",model, prenum, prenum / sum1,"预测",pres[-1:])
tempkey={}
for j in dadsadasdada:
    if j not in tempkey.keys():
        tempkey[j]=1
    else:
        tempkey[j]+=1
total=sorted(tempkey.items(), key=lambda x: x[1], reverse=True)
print("预测正确的结果数字",len(total),list(set(dadsadasdada)) )
print()

"""
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
for model in modelspath:
    prenum=0
    prenums=[]
    sum1=0
    for i, line in enumerate(lis):
        output = inference(accquiresource(lis[i]), language=CHINESE2, MODEL_DIR=PATH, codersum=model)
        output_lis = re.findall(r'.{2}', output.replace("<tag>", ""))
        pre=" ".join(output_lis)
        if i != len(lis) - 1:
            # print(" "* 20+str(len(list(set(lis[i+1].split(" ")).intersection(set(pre.split(" ")))))))
            prenum += len(list(set(lis[i + 1].split(" ")).intersection(set(pre.split(" ")))))
            prenums.append(len(list(set(lis[i + 1].split(" ")).intersection(set(pre.split(" "))))))

            sum1 += 1
        pres.append(pre)
    print("历史数",prenums)
    print("历史平值值",model, prenum, prenum / sum1,"预测",pres[-1:])
    # print("实际",lis[-2:])
"""
testlis=[]
trainlis=[]
f = open("./couplet1/"+modelspath[0]+"train_in.txt", "w")
g = open("./couplet1/"+modelspath[0]+"train_out.txt", "w")
f1 = open("./couplet1/"+modelspath[0]+"test_in.txt", "w")
g1 = open("./couplet1/"+modelspath[0]+"test_out.txt", "w")
for i,line in enumerate(trainset):
    if i%10==1:
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
    f1.write(ins + "\n")
    g1.write(outs + "\n")
print("合计 %d 条测试集" % len(testlis))
