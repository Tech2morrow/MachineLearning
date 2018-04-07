# 朴素贝叶斯分类器

## 问题描述

 编写一个朴素贝叶斯分类器
① 训练集为 train.csv；
② 测试集为 test.csv；
③ 对测试集进行预测，给出每个样例的预测结果，并计算分类精度
二、 选做
① 采用拉普拉斯平滑的手段对实验进行改进。
② 解释为什么采用拉普拉斯平滑更合理。

## 解决方法

1. 分析问题
   1. 特征输出有两个类别：0和1
   2. 有17组训练集数据，共8个特征
   3. 其中前6个特征为离散数据，后两个为连续数据。处理方法不同
2. 读入训练集，并用多个字典存储每个特征0和1情况下对应的值。
3. 用一个列表存储训练集的结果，方便计算先验概率
4. 函数comprob和comprobSpecial分别用来计算离散数据的概率和连续数据的概率
5. 逐个读取测试集
   1. 对所有特征分别计算结果为0和结果为1的概率并存储于两个list中
   2. 每个list分别把内部所有元素相乘再乘以先验概率
   3. 比较大小取更大的值，并由值确定估算结果。
   4. 与结果比较，若错误则错误个数+1，同时记录估算结果

6.输出结果

**选作部分**

在计算离散数据的概率时加入拉普拉斯平滑的公式即可。

## 实验分析

#### 必做部分

计算特征取值为离散值的概率
$$
P(X_j=X_j^{(test)}|Y=C_k) = \frac{m_{kj^{test}}}{m_k}
$$

```python
def comprob(str,val,dict):
    #计算离散数据的概率
    k=dict[str].count(val)
    return k/len(dict[str])
```

计算特征取值为连续值的概率
$$
P(X_j=X_j^{(test)}|Y=C_k) = \frac{1}{\sqrt{2\pi\sigma_k^2}}exp\Bigg{(}-\frac{(X_j^{(test)} - \mu_k)^2}{2\sigma_k^2}\Bigg{)}
$$

```python
def comprobSpecial(str,val,dict):
    #计算连续数据的概率
    ave=getaverage(dict,str)
    var=getVar(dict,str)
    arg1=(-1)*math.pow((float(val)-ave),2)/(2*var)
    arg2=1/(pow((2*math.pi*var),0.5))
    res=arg2*math.exp(arg1)
    return res

```

读取训练集数据，存入字典中，结果-特征值

```python
for i in range(1,len(contentset)):
    labelist.append(contentset[i][9])#存结果0还是1，方便先验概率的计算
    codict[contentset[i][9]].append(contentset[i][1])
    rodict[contentset[i][9]].append(contentset[i][2])
    kndict[contentset[i][9]].append(contentset[i][3])
    tedict[contentset[i][9]].append(contentset[i][4])
    umdict[contentset[i][9]].append(contentset[i][5])
    todict[contentset[i][9]].append(contentset[i][6])
    dendict[contentset[i][9]].append(contentset[i][7])
    sudict[contentset[i][9]].append(contentset[i][8])
```

计算概率存入列表最后相乘得结果

```python
 result0=[]
    res=comprob('0',csvline[1],codict)
    result0.append(res)
    result0.append(comprob('0',csvline[2],rodict))
    result0.append(comprob('0', csvline[3], kndict))
    result0.append(comprob('0', csvline[4], tedict))
    result0.append(comprob('0', csvline[5], umdict))
    result0.append(comprob('0', csvline[6], todict))
    result0.append(comprobSpecial('0', csvline[7], dendict))
    result0.append(comprobSpecial('0', csvline[8], sudict))
    res0=getResult(result0)*labelval[0]
    result1=[]
    result1.append(comprob('1', csvline[1], codict))
    result1.append(comprob('1', csvline[2], rodict))
    result1.append(comprob('1', csvline[3], kndict))
    result1.append(comprob('1', csvline[4], tedict))
    result1.append(comprob('1', csvline[5], umdict))
    result1.append(comprob('1', csvline[6], todict))
    result1.append(comprobSpecial('1', csvline[7], dendict))
    result1.append(comprobSpecial('1', csvline[8], sudict))
    res1=getResult(result1)*labelval[1]
```

取结果更大的获得预测结果

与实际结果比较来判断是否预测正确

```python
    if res0==max(res0,res1):
        cc=0
    else:
        cc=1
    if(str(cc)!=csvline[9]):#如果预测不符
        wrongres=wrongres+1
```

#### 选作部分

拉普拉斯平滑（是针对离散数据的）
$$
P(X_j=X_j^{(test)}|Y=C_k) = \frac{m_{kj^{test}} + \lambda}{m_k + O_j\lambda}
$$

```python
def comprob(str,val,dict,larg):
    k=dict[str].count(val)
    #拉普拉斯平滑公式，因为取值个数前六个特征有的是2有的是3，所以加一个列表存起来
    return (k+1)/(len(dict[str])+laplacearg[larg-1])
```

拉普拉斯平滑公式，其中Oj表示j个特征值的取值个数，前六个特征取值个数并不相同，所以用列表存了下来

### 选作问题

<u>为什么采用拉普拉斯平滑更合理？</u>

答：某些时候，可能某些类别在样本中没有出现，这样可能导致
$$
P(X_j=X_j^{(test)}|Y=C_k)
$$
为0，这样会影响后验的估计，为了解决这种情况，引入拉普拉斯平滑。
