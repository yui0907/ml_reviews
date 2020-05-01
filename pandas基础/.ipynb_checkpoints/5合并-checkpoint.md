## 问题与练习

### 1. 问题
#### 【问题一】 请思考什么是append/assign/combine/update/concat/merge/join各自最适合使用的场景，并举出相应的例子。



- append：主要是添加**行**
   1. 利用Series序列添加行（必须指定行的name）
    - s = pd.Series({'Gender':'F','Height':188},name='new_row')
    - df_append.append(s)
   2. 用DataFrame添加表（算是多行）
    - df_temp = pd.DataFrame({'Gender':['F','M'],'Height':[188,176]},index=['new_1','new_2'])
    - df_append.append(df_temp)
- assign方法：该方法主要用于添加列，列名直接由参数指定
   1. 添加单列：
    - s = pd.Series(list('abcd'),index=range(4))
    - df_append.assign(Letter=s) ##Letter即列名
   2.添加多列：
    - df_append.assign(col1=lambda x:x['Gender']*2,col2=s)


- combine：表的填充函数，可以根据某种规则填充,df1.combine(df2,func)
    
    - 首先脑子里先生成一张新表df3，它的行列为df1和df2的并集，然后将原来的df1和df2的大小扩充到df3（根据索引对齐扩充而并不是拼接），缺失值为NaN，然后对df1和df2同时逐列进行func参数的规则操作(来自助教大大GYH的解释）

```
 #combine
import numpy as np
import pandas as pd
df = pd.read_csv('data/table.csv')
#
df1 = df.loc[:1,['Gender','Height']].copy()
df2 = df.loc[10:11,['Gender','Height']].copy()
df1.combine(df2,lambda x,y:print(x,y))  ##x,y分别代表
# 这里可能是仅按列打印扩充后的df1,df2，不做任何基于规则的填充，导致结果是一个全NAN的df？

#combine2 ：根据列均值的大小填充
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [8, 7], 'B': [6, 5]})
df1.combine(df2,lambda x,y:x if x.mean()>y.mean() else y)

# （默认状态下，后面的表没有的行列都会设置为NaN）
df2 = pd.DataFrame({'B': [8, 7], 'C': [6, 5]},index=[1,2])
df1.combine(df2,lambda x,y:x if x.mean()>y.mean() else y)

##overwrite=False,使得df1原来符合条件的值不会被覆盖
df1.combine(df2,lambda x,y:x if x.mean()>y.mean() else y,overwrite=False) 
```


- update:
    - 返回的框索引只会与被调用框的一致（默认使用左连接)
    - 第二个框中的nan元素不会起作用
    - 没有返回值，直接在df上操作
 ```
 df1 = pd.DataFrame({'A': [1, 2, 3],
                    'B': [400, 500, 600]})
df2 = pd.DataFrame({'B': [4, 5, 6],
                    'C': [7, 8, 9]})
df1.update(df2)
##返回格式与df1一致
```
    - 部分更新
```
df1 = pd.DataFrame({'A': ['a', 'b', 'c'],
                    'B': ['x', 'y', 'z']})
df2 = pd.DataFrame({'B': ['d', 'e']}, index=[1,2])
df1.update(df2)
```
    - 缺失值不会更新
```
df1 = pd.DataFrame({'A': [1, 2, 3],
                    'B': [400, 500, 600]})
df2 = pd.DataFrame({'B': [4, np.nan, 6]})
df1.update(df2)
```


- concat： 可以在**两个维度**上**拼接**，默认纵向拼接（axis=0）。
    - 拼接方式默认外连接
```
df1 = pd.DataFrame({'A': ['A0', 'A1'],
                    'B': ['B0', 'B1']},
                    index = [0,1])
df2 = pd.DataFrame({'A': ['A2', 'A3'],
                    'B': ['B2', 'B3']},
                    index = [1,2])
df3 = pd.DataFrame({'A': ['A1', 'A3'],
                    'D': ['D1', 'D3'],
                    'E': ['E1', 'E3']},
                    index = [1,3])
pd.concat([df1,df2]) ##行索引为0，1，1，2
```
    -  axis=1时沿列方向拼接
```
pd.concat([df1,df2],axis=1)
```
    -  join设置为内连接（由于默认axis=0，因此列取交集）
```
pd.concat([df3,df1],join='inner') ##列的交集为A。 行索引：1，3，0，1 列：A 
```
    - 其他设置
```
pd.concat([df3,df1],join='outer',sort=True) #sort设置列排序，默认为False
pd.concat([df3,df1],verify_integrity=True,sort=True)  ##报错，verify_integrity检查axis上是否唯一（）
## 同样，可以添加Series：
s = pd.Series(['X0', 'X1'], name='X')
pd.concat([df1,s],axis=1)
##key参数用于对不同的数据框增加一个标号（索引），便于索引：
pd.concat([df1,df2], keys=['x', 'y'])
```


- merge函数的作用是将两个pandas对象**横向**合并，遇到重复的索引项时会使用笛卡尔积，默认inner连接，可选left、outer、right连接

```
pd.merge(left, right, on='key1')

pd.merge(left, right, how='outer', on=['key1','key2'])
```


- join函数: 函数作用是将多个pandas对象横向拼接，遇到重复的索引项时会使用笛卡尔积，默认左连接，可选inner、outer、right连接
  
```
left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3'],
                     'key': ['K0', 'K1', 'K0', 'K1']})
right = pd.DataFrame({'C': ['C0', 'C1'],
                      'D': ['D0', 'D1']},
                     index=['K0', 'K1'])
left.join(right, on='key')
```
     还可以与索引进行匹配
```
left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3'],
                     'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1']})
index = pd.MultiIndex.from_tuples([('K0', 'K0'), ('K1', 'K0'),
                                   ('K2', 'K0'), ('K2', 'K1')],names=['key1','key2'])
right = pd.DataFrame({'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']},
                     index=index)
left.join(right, on=['key1','key2'])
```


#### 【问题二】 merge_ordered和merge_asof的作用是什么？和merge是什么关系？
- 可能功能更细更全吧。


#### 【问题三】 请构造一个多级索引与多级索引合并的例子，尝试使用不同的合并函数。

```python
left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3'],
                     'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1']})

left=left.set_index(['key1','key2'])
index = pd.MultiIndex.from_tuples([('K0', 'K0'), ('K1', 'K0'),
                                   ('K2', 'K0'), ('K2', 'K1')],names=['key1','key2'])
right = pd.DataFrame({'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']},
                     index=index)
left.join(right, on=['key1','key2'],how='inner')
```

#### 【问题四】 上文提到了连接的笛卡尔积，那么当连接方式变化时（inner/outer/left/right），这种笛卡尔积规则会相应变化吗？请构造相应例子。


- 无论是那种连接，都是先把表以笛卡尔积的方式连接，然后通过on来筛选数据。连接不会影响笛卡尔积的规则变化


### 2. 练习
#### 【练习一】有2张公司的员工信息表，每个公司共有16名员工，共有五个公司，请解决如下问题：

```python
df=pd.read_csv('data/Employee1.csv')
df.head(2)
```

```python
df2=pd.read_csv('data/Employee2.csv')
df2.head(2)
```

#### (a) 每个公司有多少员工满足如下条件：既出现第一张表，又出现在第二张表。

```python
pd.merge(df, df2, on='Name').shape[0]

##16个
```

#### (b) 将所有不符合(a)中条件的行筛选出来，合并为一张新表，列名与原表一致。

```python
res = pd.concat([df,df2]).drop_duplicates(subset=['Name'],keep=False)
```

#### (c) 现在需要编制所有80位员工的信息表，对于(b)中的员工要求不变，

- 对于满足(a)条件员工，它们在某个指标的数值，取偏离它所属公司中满足(b)员工的均值数较小的哪一个，

- 例如：P公司在两张表的交集为{p1}，并集扣除交集为{p2,p3,p4}，那么如果后者集合的工资均值为1万元，且p1在表1的工资为13000元，在表2的工资为9000元，那么应该最后取9000元作为p1的工资，最后对于没有信息的员工，利用缺失值填充。


```python
lst=list(pd.merge(df, df2, on='Name').Name)
rr=res.groupby('Company').mean()
tmp1=df[df.Name.isin(lst)].sort_values(by='Name').reset_index(drop=True)
tmp2=df2[df2.Name.isin(lst)].sort_values(by='Name').reset_index(drop=True)
```

```python
def clg(l,m):
    a=tmp1.loc[tmp1.Name==l,m].values
    b=tmp2.loc[tmp2.Name==l,m].values
    c=rr.loc[tmp1.loc[tmp1.Name==l,'Company']][m].values
    t=[abs(a/c-1),abs(b/c-1)]
    return [a,b][t.index(min(t))]
```

```python
#直接在tmp1上修改
for ll in lst:
    tmp1.loc[tmp1.Name==ll,'Age']= clg(ll,'Age')[0]
    tmp1.loc[tmp1.Name==ll,'Height']= clg(ll,'Height')[0]
    tmp1.loc[tmp1.Name==ll,'Weight']= clg(ll,'Weight')[0]
    tmp1.loc[tmp1.Name==ll,'Salary']= clg(ll,'Salary')[0]
             
##结果,在b的结果上拼接tmp1
res = pd.concat([res,tmp1])
```

#### 【练习二】有2张课程的分数表（分数随机生成），但专业课（学科基础课、专业必修课、专业选修课）与其他课程混在一起，请解决如下问题：

```python
df1=pd.read_csv('data/Course1.csv')
df1.head(2)
```

```python
df2=pd.read_csv('data/Course2.csv')
df2.head(2)
```

#### (a) 将两张表分别拆分为专业课与非专业课（结果为四张表）。

```python
tp=['学科基础课','专业必修课','专业选修课']
df1_pro=df1.loc[df1['课程类别'].isin(tp)]
df1_unpro=df1.loc[~df1['课程类别'].isin(tp)]
df2_pro=df2.loc[df2['课程类别'].isin(tp)]
df2_unpro=df2.loc[~df2['课程类别'].isin(tp)]
```

#### (b) 将两张专业课的分数表和两张非专业课的分数表分别合并。

```python
df_pro=pd.concat([df1_pro,df2_pro])
df_unpro=pd.concat([df1_unpro,df2_unpro])
```

#### (c) 不使用(a)中的步骤，请直接读取两张表合并后拆分。

```python
df=pd.concat([df1,df2])
df_pro=df.loc[df['课程类别'].isin(tp)]
df_unpro=df.loc[~df['课程类别'].isin(tp)]
```

#### (d) 专业课程中有缺失值吗，如果有的话请在完成(3)的同时，用组内（3种类型的专业课）均值填充缺失值后拆分。

```python
df_pro.isnull().sum()  
#分数有3个缺失值
```

```python
df_pro['分数'] =df_pro.groupby('课程类别')['分数'] .transform(lambda x: x.fillna(x.mean()))
```
