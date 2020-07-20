import re
"""
import random
import datetime
import time
f=open("txt","r")
fl=f.readlines()
f.close()
#fl.reverse()
lis3=[]
lis5=[]
for i,line in enumerate(fl) :
    listemp3=[int(num) for num in  line.strip().split(" ")[1].split("+")[:3]]
    listemp3.sort()
    listemp=[str(num) for num in listemp3]
    lis3.append(" ".join(listemp))

    listemp5=[int(num) for num in  line.strip().split(" ")[1].split("+")]
    listemp5.sort()
    listemp=[str(num) for num in listemp5]
    lis5.append(" ".join(listemp))

#print (lis)
res3={}
numres3={}

for i in lis3:
    if i not in res3.keys():
        res3[i]=1
    else:res3[i]+=1
    for j in i.split(" "):
        if j not in numres3.keys():
            numres3[j]=1
        else:numres3[j]+=1

numres5={}
for i in lis5:
    for j in i.split(" "):
        if j not in numres5.keys():
            numres5[j]=1
        else:numres5[j]+=1


for key,value in res3.items():
    print (key,"---",value)
print (len(lis3))

print (len(res3))
import operator
for key,value in sorted(numres3.items(), key=operator.itemgetter(1)):
    print (key,"---",value)
print("3 over")
for key,value in sorted(numres5.items(), key=operator.itemgetter(1)):
    print (key,"---",value)
print("5 over")
#lis=[]
#for i,line in enumerate(fl) :
#    lis.append("3%2d"%(i+1)+" "+line.strip().split(" ")[1])
#print (lis)
#lis.reverse()
#for i in lis:
#    print (i)
# print(lis)

"""
# """
f=open("txt","r",encoding="utf-8")
fl=f.readlines()
f.close()
fl.reverse()
#for i in range(len(fl)):
#    print (str(840-i)+" "+fl[i].strip())
lis=[]
for i,line in enumerate(fl):
    if i%2==0:
        lis.append(" ".join(re.findall(r'.{2}',re.sub("\D", "", line.replace("<tag>","")))))
print("'"+"','".join(lis)+"'")
print(len(lis))
# """
"""
key={}
for i,lis in enumerate(list1):
    tempkey = {}
    for j in lis :
        for tem in j.strip().split(" "):
            if tem not in tempkey.keys():
                tempkey[tem]=1
            else:tempkey[tem]+=1
    key[str(i+1)]=sorted(tempkey.items(), key=lambda x: x[1], reverse=True)

for k,v in key.items():
    print(k,len(v),v)
"""
# f=open("data.csv","r")
# fl=f.readlines()
# f.close()
# fl.reverse()
#for i in range(len(fl)):
#    print (str(840-i)+" "+fl[i].strip())
# lis=[]
# for i in fl:
#     if "#" not in i:
#         print (i.strip().split(" ")[1].replace("+"," "))
#         lis.append(i.strip().split(" ")[1].replace("+","\t"))
#
# f=open("data.csv","w")
# for line in lis:
#     f.write(line+"\n")
# f.close()

"""

f=open("history.txt","r")
fl=f.readlines()
f.close()
#fl.reverse()
#for i in range(len(fl)):
#    print (str(840-i)+" "+fl[i].strip())
temp=200605
for i,line in enumerate(fl):
    if line.startswith("第"):
        if int(line.split(" ")[0][1:7])==temp or int(line.split(" ")[0][1:7])==temp-1:
            temp=int(line.split(" ")[0][1:7])
        else:
            print (i,line.strip())
            temp=int(line.split(" ")[0][1:7])
        #print (" ".join(re.findall(r'.{2}',re.sub("\D", "", lis.replace("<tag>","")))))
"""


# f=open("data","r")
# fl=f.readlines()
# f.close()
# f=open("data1","r")
# gl=f.readlines()
# f.close()
#
# lis1=[]
# lis2=[]
# for i in fl:
#     lis1.append(i.split(",")[0])
# for i in gl:
#     lis2.append(i.split(",")[0])
#
# print(set(lis1).difference(set(lis2)))
# print(set(lis2).difference(set(lis1)))