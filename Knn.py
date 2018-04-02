import csv
def distance(a,b):# 计算训练集和测试集的距离
    dst=0
    for i in range(0,256):
        if a[i]!=b[i]:
            dst=dst+1
    #print(dst)
    return dst
def getclass(k,a):#根据knn分类器的思想取排好序的数据前k个确定测试集数据应该属于哪个类
    item1=[]
    maxres=0
    maxcls=0
    tempres=0
    for i in range(0,k):#把类取出来
        item1.append(a[i].cls)
    #print(item1)
    item2=set(item1)#去重复
    #print(item2)
    for itm in item2:
        tempres=item1.count(itm)
        if(tempres>maxres):
            maxcls=itm
            maxres=tempres
    #print(maxcls)
    return maxcls
def comclass(c):# 计算csv文件中给的数据属于哪一类
    try:
        cls=c.index('1')
    except:
        cls=0;
    #print (cls+1)
    return cls+1
def sort_data(x):#对我定义的类进行排序
    return x.dst
class knndata:#用来存储测试集数据与训练集数据的距离及训练集数据的类
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
csv_file1=csv.reader(open('e:/semeion_train.csv','r'))#读取训练集
for graph in csv_file1:
    temp=graph[0].split(' ')
    traindata.append(temp)#将训练集数据从str变成列表并存入traindata中方便以后的计算
    classdata.append(comclass(temp[256:266]))#计算每行训练集的类并加入列表中
csv_file2=csv.reader(open('e:/semeion_test.csv','r'))# 读取测试集
for graph in csv_file2:
    temp=graph[0].split(' ')
    resdst=[]#存储这一行测试集数据与所有训练集数据之间的距离，每一次循环前清空
    knnres=[]#存储所有训练集数据与该行测试技术局之间的距离，所有训练集数据的类
    clas=comclass(temp[256:266])#计算测试集数据正确该属于哪一类（标准答案）
    for i in range(0,len(traindata)):#逐个计算该测试集数据与所有训练集数据之间的距离
        res=distance(temp,traindata[i])
        resdst.append(res)
        tempknn=knndata(i,res,classdata[i])#自定义类，存距离及类别
        knnres.append(tempknn)
    #print(res)
    result=sorted(knnres,key=sort_data,reverse=False)#排序，升序
    for kk in range(0,4):#不同k值下计算结果，并比较来确定是否正确
        if(kval[kk]==1):
            if(result[0].cls!=clas):
                wrongk[0]=wrongk[0]+1
        else:
            krescls=getclass(kval[kk],result)
            if(krescls!=clas):
                wrongk[kk] = wrongk[kk] + 1
print(wrongk)
for i in wrongk:
    print(str(i/478)+"  ")



