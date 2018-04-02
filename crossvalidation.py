import csv
import random
import matplotlib.pyplot as plt
def plot(x,y):
    plt.figure(1)
    plt.xlabel("k")
    plt.ylabel("ErrorNum")
    plt.title("ErrorNum")
    plt.xlim(min(x)-1,max(x)+1)
    plt.ylim(0,max(y)*1.1)
    plt.xticks(range(int(min(x)-1),int(max(x)+2)))
    plt.plot(x, y)
    plt.show()
def distance(a,b):
    dst=0
    for i in range(0,256):
        if a[i]!=b[i]:
            dst=dst+1
    return dst
def getclass(k,a):
    item1=[]
    maxres=0
    maxcls=0
    tempres=0
    for i in range(0,k):
        item1.append(a[i].cls)
    item2=set(item1)
    for itm in item2:
        tempres=item1.count(itm)
        if(tempres>maxres):
            maxcls=itm
            maxres=tempres
    return maxcls
def getid(j,i):
    if(j==0):
        return 223+i
    else:
        if(i<j*223):
            return i
        else:
            return 223+i
def comclass(c):
    try:
        cls=c.index('1')
    except:
        cls=0;
    return cls+1
def sort_data(x):
    return x.dst
class knndata:
    id=0
    dst=0
    cls=0
    def __init__(self, id, dst,cls):
        self.id = id
        self.dst = dst
        self.cls = cls
traindata=[]
classdata=[]
wrongk=[0,0,0,0]
kval=[1,3,5,10]
csv_file1=csv.reader(open('semeion_train.csv','r'))
for graph in csv_file1:
    temp=graph[0].split(' ')
    traindata.append(temp)
random.shuffle(traindata)# 随机打乱list
for adata in traindata:
    classdata.append(comclass(adata[256:266]))#计算随即打乱的list的类
# 1115 223
traindataset=[]
testdataset=[]
#wrongk=[0]*20
#wrongkset=[]
#for i in range(0,5):
#    wrongkset.append(wrongk)
wrongkset=[[0 for x in range(20)] for x in range(5)]
slic=[0,223,446,669,892,1115]
for j in range(0,5):
    testdataset.append(traindata[slic[j]:slic[j+1]])
    print(len(testdataset[j]))
    traindataset.append(traindata[0:slic[j]]+traindata[slic[j+1]:1115])
    print(len(traindataset[j]))
for j in range(0,5):#总共五组数据
    for testitem in testdataset[j]:
        resdst = []
        knnres = []
        clas = comclass(testitem[256:266])#计算给测试集的类
        for i in range(0,len(traindataset[j])):
            res=distance(testitem,traindataset[j][i])
            resdst.append(res)
            tempknn = knndata(i, res, classdata[getid(j,i)])
            knnres.append(tempknn)
        result = sorted(knnres, key=sort_data, reverse=False)  # 排序，升序
        for kk in range(1, 21):#其实是1到20但是因为这个
            if (kk == 1):
                if (result[0].cls != clas):
                    wrongkset[j][0] = wrongkset[j][0] + 1
            else:
                krescls = getclass(kk, result)
                if (krescls != clas):
                    wrongkset[j][kk-1] = wrongkset[j][kk-1] + 1
resultset=[]
for j in range(0,5):
    print(wrongkset[j])
for i in range(0,20):
    sum=0
    for j in range(0,5):
        sum=sum+wrongkset[j][i]
    resultset.append(sum/5)
print(resultset)
plot(range(len(resultset)),resultset)