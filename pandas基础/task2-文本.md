## 问题与练习
### 1. 问题

#### 【问题一】 str对象方法和df/Series对象方法有什么区别？



```python
#### 【问题二】 给出一列string类型，如何判断单元格是否是数值型数据？
#### 【问题三】 rsplit方法的作用是什么？它在什么场合下适用？
#### 【问题四】 在本章的第二到第四节分别介绍了字符串类型的5类操作，请思考它们各自应用于什么场景？
```

### 2. 练习
#### 【练习一】 现有一份关于字符串的数据集，请解决以下问题：
#### （a）现对字符串编码存储人员信息（在编号后添加ID列），使用如下格式：“×××（名字）：×国人，性别×，生于×年×月×日”



```python
import pandas as pd
df = pd.read_csv('data/String_data_one.csv',index_col='人员编号').astype('str')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>国籍</th>
      <th>性别</th>
      <th>出生年</th>
      <th>出生月</th>
      <th>出生日</th>
    </tr>
    <tr>
      <th>人员编号</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>aesfd</td>
      <td>2</td>
      <td>男</td>
      <td>1942</td>
      <td>8</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fasefa</td>
      <td>5</td>
      <td>女</td>
      <td>1985</td>
      <td>10</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>aeagd</td>
      <td>4</td>
      <td>女</td>
      <td>1946</td>
      <td>10</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>aef</td>
      <td>4</td>
      <td>男</td>
      <td>1999</td>
      <td>5</td>
      <td>13</td>
    </tr>
    <tr>
      <th>5</th>
      <td>eaf</td>
      <td>1</td>
      <td>女</td>
      <td>2010</td>
      <td>6</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['ID']=df['姓名']+':'+df['国籍']+'国人，性别'+df['性别']+'，生于'+df['出生年']+'年'+df['出生月']+'月'+df['出生日']+'日'
```


```python
cols = [df.columns.values[-1]]+list(df.columns.values[:-1])
df = df[cols]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>姓名</th>
      <th>国籍</th>
      <th>性别</th>
      <th>出生年</th>
      <th>出生月</th>
      <th>出生日</th>
    </tr>
    <tr>
      <th>人员编号</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>aesfd:2国人，性别男，生于1942年8月10日</td>
      <td>aesfd</td>
      <td>2</td>
      <td>男</td>
      <td>1942</td>
      <td>8</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fasefa:5国人，性别女，生于1985年10月4日</td>
      <td>fasefa</td>
      <td>5</td>
      <td>女</td>
      <td>1985</td>
      <td>10</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>aeagd:4国人，性别女，生于1946年10月15日</td>
      <td>aeagd</td>
      <td>4</td>
      <td>女</td>
      <td>1946</td>
      <td>10</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>aef:4国人，性别男，生于1999年5月13日</td>
      <td>aef</td>
      <td>4</td>
      <td>男</td>
      <td>1999</td>
      <td>5</td>
      <td>13</td>
    </tr>
    <tr>
      <th>5</th>
      <td>eaf:1国人，性别女，生于2010年6月24日</td>
      <td>eaf</td>
      <td>1</td>
      <td>女</td>
      <td>2010</td>
      <td>6</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



#### （b）将（a）中的人员生日信息部分修改为用中文表示（如一九七四年十月二十三日），其余返回格式不变。


```python
dict_map ={'0':'零','1':'一','2':'二','3':'三','4':'四','5':'五','6':'六','7':'七','8':'八','9':'九','10':'十'}

def f(x,n=2):
    x=int(x)
    if n==0:
        tp1 = [dict_map[i] for i in str(x) ]
        tp2 = ''.join(tp1)
        return tp2+'年'
    elif n==1:
        if x <= 10:return dict_map[str(x)]+'月'
        else :return '十'+ dict_map[str(x)[1]]+'月'
    else :
        if x <= 10:return dict_map[str(x)]+'日'
        elif x< 20:return '十'+ dict_map[str(x)[1]]+'日'
        elif x==20 or x==30 :return dict_map[str(x)[0]] + '十日'
        else :return dict_map[str(x)[0]] + '十'+ dict_map[str(x)[1]]+'日'

df['出生信息'] = df['出生年'].apply(lambda x :f(x,0))+df['出生月'].apply(lambda x:f(x,1))+df['出生日'].apply(lambda x:f(x,2))
```


```python
df=df.iloc[:,[0,1,2,3,7]]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>姓名</th>
      <th>国籍</th>
      <th>性别</th>
      <th>出生信息</th>
    </tr>
    <tr>
      <th>人员编号</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>aesfd:2国人，性别男，生于1942年8月10日</td>
      <td>aesfd</td>
      <td>2</td>
      <td>男</td>
      <td>一九四二年八月十日</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fasefa:5国人，性别女，生于1985年10月4日</td>
      <td>fasefa</td>
      <td>5</td>
      <td>女</td>
      <td>一九八五年十月四日</td>
    </tr>
    <tr>
      <th>3</th>
      <td>aeagd:4国人，性别女，生于1946年10月15日</td>
      <td>aeagd</td>
      <td>4</td>
      <td>女</td>
      <td>一九四六年十月十五日</td>
    </tr>
    <tr>
      <th>4</th>
      <td>aef:4国人，性别男，生于1999年5月13日</td>
      <td>aef</td>
      <td>4</td>
      <td>男</td>
      <td>一九九九年五月十三日</td>
    </tr>
    <tr>
      <th>5</th>
      <td>eaf:1国人，性别女，生于2010年6月24日</td>
      <td>eaf</td>
      <td>1</td>
      <td>女</td>
      <td>二零一零年六月二十四日</td>
    </tr>
  </tbody>
</table>
</div>



#### （c）将（b）中的ID列结果拆分为原列表相应的5列，并使用equals检验是否一致。


```python

```
