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
f=open("txt","r")
fl=f.readlines()
f.close()
fl.reverse()
for i in range(len(fl)):
    print (str(840-i)+" "+fl[i].strip())

