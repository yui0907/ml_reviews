## 问题与练习

### 1. 问题
#### 【问题一】 请思考什么是append/assign/combine/update/concat/merge/join各自最适合使用的场景，并举出相应的例子。



#### 【问题二】 merge_ordered和merge_asof的作用是什么？和merge是什么关系？
#### 【问题三】 请构造一个多级索引与多级索引合并的例子，尝试使用不同的合并函数。
#### 【问题四】 上文提到了连接的笛卡尔积，那么当连接方式变化时（inner/outer/left/right），这种笛卡尔积规则会相应变化吗？请构造相应例子。

### 2. 练习
#### 【练习一】有2张公司的员工信息表，每个公司共有16名员工，共有五个公司，请解决如下问题：

```python
import pandas as pd 
df=pd.read_csv('data/Employee1.csv')
df.head(2)
```

```python
df2=pd.read_csv('data/Employee2.csv').head()
df2.head(2)
```

#### (a) 每个公司有多少员工满足如下条件：既出现第一张表，又出现在第二张表。

```python
pd.merge(df, df2, on='Name').shape[0]

##3个，但是除了姓名一样，其他的属性不一样。。
```

#### (b) 将所有不符合(a)中条件的行筛选出来，合并为一张新表，列名与原表一致。

```python
tmp=list(pd.merge(df, df2, on='Name').Name)
print(tmp)
#['a1', 'a3', 'a6']
```

```python
res = pd.concat([df,df2])
res=res.query('Name not in ["a1", "a3", "a6"]').reset_index(drop=True)
```

#### (c) 现在需要编制所有80位员工的信息表，对于(b)中的员工要求不变，对于满足(a)条件员工，它们在某个指标的数值，取偏离它所属公司中满足(b)员工的均值数较小的哪一个，例如：P公司在两张表的交集为{p1}，并集扣除交集为{p2,p3,p4}，那么如果后者集合的工资均值为1万元，且p1在表1的工资为13000元，在表2的工资为9000元，那么应该最后取9000元作为p1的工资，最后对于没有信息的员工，利用缺失值填充。


```python
tmp=pd.merge(df, df2, on='Name')
```

```python







#### 【练习二】有2张课程的分数表（分数随机生成），但专业课（学科基础课、专业必修课、专业选修课）与其他课程混在一起，请解决如下问题：

pd.read_csv('data/Course1.csv').head()

pd.read_csv('data/Course2.csv').head()

#### (a) 将两张表分别拆分为专业课与非专业课（结果为四张表）。
#### (b) 将两张专业课的分数表和两张非专业课的分数表分别合并。
#### (c) 不使用(a)中的步骤，请直接读取两张表合并后拆分。
#### (d) 专业课程中有缺失值吗，如果有的话请在完成(3)的同时，用组内（3种类型的专业课）均值填充缺失值后拆分。
```
