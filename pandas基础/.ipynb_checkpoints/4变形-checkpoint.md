### 问题与练习
### 1. 问题
#### 【问题一】 上面提到了许多变形函数，如melt/crosstab/pivot/pivot_table/stack/unstack函数，请总结它们各自的使用特点。



- pivot:作用将压缩状态展开。原表中两个类别放在一列，pivot可将某列的取值作为新的列。局限：1.功能上较少、2.不允许values中出现重复的行列索引对
- pivot_table:在实现pivot基础上，克服了其局限.1.支持聚合参数aggfunc：对组内进行聚合统计，可传入各类函数，默认为'mean'；也支持边际汇总 margins；2.重复索引对则取values的平均
- crosstab：特殊的透视表，用于统计使用，比如计数(默认）；如果有传values，也可对values指定聚合函数进行组内计算。并且支持margins和normalize（可指定all或者index或者columns）
- melt：可认为是pivot的逆操作，将展开状态的数据进行压缩。指定压缩变量value_vars和压缩后列名。
- stack：功能与melt类似，可以看做将横向的索引放到纵向，参数level可指定变化的列索引是哪一层（或哪几层，需要列表）
- unstack：stack的逆操作。

```python
import numpy as np
import pandas as pd
df = pd.read_csv('data/table.csv')
df.head(3)
```

#### 【问题二】 变形函数和多级索引是什么关系？哪些变形函数会使得索引维数变化？具体如何变化？


- pivot_table等压缩数据的变形函数会导致多级索引；比如当传入单级行列值但是传入聚合参数后，相当于将columns先pivot后进行分组。在列索引上有了2级索引； 当传入多个行列值，也会导致多级索引
- stack、melt等展开函数可以将多级索引变为1级索引

```python
res=pd.pivot_table(df,index='School',columns='Gender',values='Height',aggfunc=['mean','sum'])
res
```

```python
##2级索引变单级索引
res.columns=['_'.join([i[1],i[0]]) for i in res.columns]
res
```

#### 【问题三】 请举出一个除了上文提过的关于哑变量方法的例子。

```python
df_d = df[['Class','Gender','Weight']]
classes = set(df_d['Class'])
for k in classes:
    df_d[k]=0
    df_d.loc[df.Class==k,k]=1

df_d.head()  
```

#### 【问题四】 使用完stack后立即使用unstack一定能保证变化结果与原始表完全一致吗？


- 如果都是默认参数则能完全一致
- 如果stack的level和unstack的level所对应起来，也能，但是unstack后的索引层级顺序可能会有变化。


<!-- #region -->


#### 【问题五】 透视表中涉及了三个函数，请分别使用它们完成相同的目标（任务自定）并比较哪个速度最快。

<!-- #endregion -->

```python
%timeit df.pivot(index='ID',columns='Gender',values='Height')
%timeit pd.pivot_table(df,index='ID',columns='Gender',values='Height')
%timeit pd.crosstab(index=df['Address'],columns=df['Gender'],values=1,aggfunc='count')
# 1.6 ms ± 27.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# 7.1 ms ± 261 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# 6.3 ms ± 134 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

#### 【问题六】 既然melt起到了stack的功能，为什么再设计stack函数？


- stack 可针对多级列索引的不同层级进行压缩
- melt 可以指定列进行压缩
- 两者功能类似，但对象可以有所不同


### 2. 练习
#### 【练习一】 继续使用上一章的药物数据集：
#### (a) 现在请你将数据表转化成如下形态，每行需要显示每种药物在每个地区的10年至17年的变化情况，且前三列需要排序：
![avatar](data/drug_pic.png)


```python
df=pd.read_csv('data/Drugs.csv')
##df.pivot(index='index',columns='YYYY',values='DrugReports').fillna('-')
```

```python
tmp=pd.pivot_table(df,index=['State','COUNTY','SubstanceName'],columns=['YYYY'],values=['DrugReports'])\
                .fillna('-').reset_index()
##二级索引变1级
tmp.columns=[ i[0] if i[1]=='' else  i[1]  for i in tmp.columns   ]
tmp.head()
```

```python
df.head(3)
```

#### (b) 现在请将(a)中的结果恢复到原数据表，并通过equal函数检验初始表与新的结果是否一致（返回True）

```python
tmp2=tmp.melt(id_vars=['State','COUNTY','SubstanceName'],value_vars=[i for i in tmp.columns[3:]],value_name='DrugReports').rename(columns={'variable':'YYYY'})
tmp2.DrugReports=tmp2.DrugReports.apply(lambda x :np.nan if x=='-' else x )
result=tmp2.dropna().reset_index(drop=True).iloc[:,[3,0,1,2,4]]

###类型转换排序后才相等
result['DrugReports']=result['DrugReports'].astype('int64') 
result['YYYY']=result['YYYY'].astype('int64') 
result.sort_values(by=['YYYY','State','COUNTY','SubstanceName']).reset_index(drop=True).equals(df.sort_values(by=['YYYY','State','COUNTY','SubstanceName']).reset_index(drop=True))  
```

#### 【练习二】 现有一份关于某地区地震情况的数据集，请解决如下问题：

```python
df=pd.read_csv('data/Earthquake.csv')
df.head(3)
```

#### (a) 现在请你将数据表转化成如下形态，将方向列展开，并将距离、深度和烈度三个属性压缩：
![avatar](data/earthquake_pic.png)


```python
result=pd.pivot_table(df,index=['日期','时间','维度','经度'],columns=['方向'],values=['深度','烈度','距离']).stack(0).fillna('-')
result
```

#### (b) 现在请将(a)中的结果恢复到原数据表，并通过equal函数检验初始表与新的结果是否一致（返回True）

```python
tmp=result.unstack(4).stack(0)
result=tmp.apply(lambda x:x.apply(lambda x:np.nan if x=='-' else x )).dropna().reset_index().iloc[:,[0,1,2,3,4,7,5,6]]
##同样是需要序号有变，需要将数据进行排序后才相等
result.sort_values(by=[c for c in result.columns]).reset_index(drop=True).equals(df.sort_values(by=[c for c in result.columns]).reset_index(drop=True))
```

```python

```

```python

```
