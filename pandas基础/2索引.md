## 问题与练习
### 1. 问题
#### 【问题一】 如何更改列或行的顺序？如何交换奇偶行（列）的顺序？

```python
import numpy as np
import pandas as pd
df = pd.read_csv('data/table.csv',index_col='ID')
df.head()
```

```python
#奇偶换行
n=df.shape[0]
if n%2==0:
    ls=[i+(-1)**i for i in range(df.shape[0])]
else:
    ls=[i+(-1)**i for i in range(df.shape[0]-1)]+[df.shape[0]-1]
    
df.loc[[df.index[i] for i in ls ]].head()
```

#### 【问题二】 如果要选出DataFrame的某个子集，请给出尽可能多的方法实现。

```python
df.loc[[1101,1102]] ##取多行
df[['School','Class']] ##取多列
df.iloc[1:3,2:4]
df.loc[[1101,1102],['School','Class']]
df.ix[[1101,1102],2:3]
df.ix[[1101,1102],['School','Class']]
```

#### 【问题三】 query函数比其他索引方法的速度更慢吗？在什么场合使用什么索引最高效？


- 使用query，速度最慢
- 把要查询的键用set_index设为index后再用loc切片查询会加快速度。



#### 【问题四】 单级索引能使用Slice对象吗？能的话怎么使用，请给出一个例子。

```python
idx=pd.IndexSlice
m,n=1101,1102
df.loc[idx[m,n],:]
```

#### 【问题五】 如何快速找出某一列的缺失值所在索引？

```python
df.loc[idx[m,n],'School']=np.NaN
df.loc[idx[m,n],:]
```

```python
df.loc[df.School.isnull(),:].index
```

#### 【问题六】 索引设定中的所有方法分别适用于哪些场合？怎么直接把某个DataFrame的索引换成任意给定同长度的索引？


- index_col 读取的时候指定索引列
- reindex是指重新索引，它的重要特性在于索引对齐，很多时候用于重新排序
-  set_index：将某些列作为索引
- reset_index方法，它的主要功能是将索引重置
- rename_axis是针对多级索引的方法，作用是修改某一层的索引名
-  rename方法用于修改列或者行索引标签，而不是索引名


#### 【问题七】 多级索引有什么适用场合？



多级索引是将多个不同或相同的索引方法组合使用，对单级索引空间或者空间范围进行多级划分，解决超大型数据量的GIS系统检索、分析、显示的效率问题。



#### 【问题八】 什么时候需要重复元素处理？


- 数据行重复或者列重复时


### 2. 练习
#### 【练习一】 现有一份关于UFO的数据集，请解决下列问题：

```python
df=pd.read_csv('data/UFO.csv')
df.head(3)
```

#### （a）在所有被观测时间超过60s的时间中，哪个形状最多？


```python
df.loc[df['duration (seconds)']>60,'shape'].value_counts()[:3]

###light最多  10713
```

#### （b）对经纬度进行划分：-180°至180°以30°为一个划分，-90°至90°以18°为一个划分，请问哪个区域中报告的UFO事件数量最多？

```python
df['longitude_2']=pd.cut(df.longitude,bins=[30*i for i in range(-6,7)])
df['latitude_2']=pd.cut(df.latitude,bins=[18*i for i in range(-5,6)])
df.groupby(['longitude_2','latitude_2']).datetime.count().sort_values(ascending=False).head(3)
## 维度区间(-90, -60] 经度区间 (36, 54]   报告最多    27891
```

#### 【练习二】 现有一份关于口袋妖怪的数据集，请解决下列问题：

```python
df=pd.read_csv('data/Pokemon.csv',index_col=0)
df.head(5)
```

#### （a）双属性的Pokemon占总体比例的多少？

```python
from math import isnan
((1-df['Type 1'].isnull())&(1-df['Type 2'].isnull())).sum()/df.shape[0]
##0.5175
```

#### （b）在所有种族值（Total）不小于580的Pokemon中，非神兽（Legendary=False）的比例为多少？

```python
((df.Total >=580) & (df.Legendary==False)).sum()/(df.Total >=580).sum()
##0.42
```

#### （c）在第一属性为格斗系（Fighting）的Pokemon中，物攻排名前三高的是哪些？

```python
df[df['Type 1']=='Fighting'].sort_values(by='Attack',ascending=False)[:3]
```

#### （d）请问六项种族指标（HP、物攻、特攻、物防、特防、速度）极差的均值最大的是哪个属性（只考虑第一属性，且均值是对属性而言）？

```python
ls=['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']
tmp=df.groupby('Type 1')[ls]
(tmp.max()-tmp.min()).mean(axis=1).sort_values(ascending=False)[:3]

##Psychic
```

#### （e）哪个属性（只考虑第一属性）的神兽比例最高？该属性神兽的种族值也是最高的吗？

```python
tp= df[df['Legendary']==True].groupby('Type 1').Name.count()/df.groupby('Type 1').Name.count()
tp.sort_values(ascending=False)[:3]
##Flying 占比最高 0.5 
```
