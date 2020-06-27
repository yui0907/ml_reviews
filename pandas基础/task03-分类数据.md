# 第8章 分类数据


```python
import pandas as pd
import numpy as np
df = pd.read_csv('data/table.csv')
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
      <th>School</th>
      <th>Class</th>
      <th>ID</th>
      <th>Gender</th>
      <th>Address</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Math</th>
      <th>Physics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>1101</td>
      <td>M</td>
      <td>street_1</td>
      <td>173</td>
      <td>63</td>
      <td>34.0</td>
      <td>A+</td>
    </tr>
    <tr>
      <th>1</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>1102</td>
      <td>F</td>
      <td>street_2</td>
      <td>192</td>
      <td>73</td>
      <td>32.5</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>1103</td>
      <td>M</td>
      <td>street_2</td>
      <td>186</td>
      <td>82</td>
      <td>87.2</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>3</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>1104</td>
      <td>F</td>
      <td>street_2</td>
      <td>167</td>
      <td>81</td>
      <td>80.4</td>
      <td>B-</td>
    </tr>
    <tr>
      <th>4</th>
      <td>S_1</td>
      <td>C_1</td>
      <td>1105</td>
      <td>F</td>
      <td>street_4</td>
      <td>159</td>
      <td>64</td>
      <td>84.8</td>
      <td>B+</td>
    </tr>
  </tbody>
</table>
</div>



## 一、category的创建及其性质
### 1. 分类变量的创建
#### （a）用Series创建


```python
pd.Series(["a", "b", "c", "a"], dtype="category")
```




    0    a
    1    b
    2    c
    3    a
    dtype: category
    Categories (3, object): [a, b, c]



#### （b）对DataFrame指定类型创建


```python
temp_df = pd.DataFrame({'A':pd.Series(["a", "b", "c", "a"], dtype="category"),'B':list('abcd')})
temp_df.dtypes
```




    A    category
    B      object
    dtype: object



#### （c）利用内置Categorical类型创建


```python
cat = pd.Categorical(["a", "b", "c", "a"], categories=['a','b','c'])
pd.Series(cat)
```




    0    a
    1    b
    2    c
    3    a
    dtype: category
    Categories (3, object): [a, b, c]



#### （d）利用cut函数创建

#### 默认使用区间类型为标签


```python
pd.cut(np.random.randint(0,60,5), [0,10,30,60])
```




    [(10, 30], (0, 10], (10, 30], (30, 60], (30, 60]]
    Categories (3, interval[int64]): [(0, 10] < (10, 30] < (30, 60]]



#### 可指定字符为标签


```python
pd.cut(np.random.randint(0,60,5), [0,10,30,60], right=False, labels=['0-10','10-30','30-60'])
```




    [10-30, 30-60, 30-60, 10-30, 30-60]
    Categories (3, object): [0-10 < 10-30 < 30-60]



### 2. 分类变量的结构
#### 一个分类变量包括三个部分，元素值（values）、分类类别（categories）、是否有序（order）
#### 从上面可以看出，使用cut函数创建的分类变量默认为有序分类变量
#### 下面介绍如何获取或修改这些属性
#### （a）describe方法
#### 该方法描述了一个分类序列的情况，包括非缺失值个数、元素值类别数（不是分类类别数）、最多次出现的元素及其频数


```python
s = pd.Series(pd.Categorical(["a", "b", "c", "a",np.nan], categories=['a','b','c','d']))
s.describe()
```




    count     4
    unique    3
    top       a
    freq      2
    dtype: object



#### （b）categories和ordered属性
#### 查看分类类别和是否排序


```python
s.cat.categories
```




    Index(['a', 'b', 'c', 'd'], dtype='object')




```python
s.cat.ordered
```




    False



### 3. 类别的修改

#### （a）利用set_categories修改
#### 修改分类，但本身值不会变化


```python
s = pd.Series(pd.Categorical(["a", "b", "c", "a",np.nan], categories=['a','b','c','d']))
s.cat.set_categories(['new_a','c'])
```




    0    NaN
    1    NaN
    2      c
    3    NaN
    4    NaN
    dtype: category
    Categories (2, object): [new_a, c]



#### （b）利用rename_categories修改
#### 需要注意的是该方法会把值和分类同时修改


```python
s = pd.Series(pd.Categorical(["a", "b", "c", "a",np.nan], categories=['a','b','c','d']))
s.cat.rename_categories(['new_%s'%i for i in s.cat.categories])
```




    0    new_a
    1    new_b
    2    new_c
    3    new_a
    4      NaN
    dtype: category
    Categories (4, object): [new_a, new_b, new_c, new_d]



#### 利用字典修改值


```python
s.cat.rename_categories({'a':'new_a','b':'new_b'})
```




    0    new_a
    1    new_b
    2        c
    3    new_a
    4      NaN
    dtype: category
    Categories (4, object): [new_a, new_b, c, d]



#### （c）利用add_categories添加


```python
s = pd.Series(pd.Categorical(["a", "b", "c", "a",np.nan], categories=['a','b','c','d']))
s.cat.add_categories(['e'])
```




    0      a
    1      b
    2      c
    3      a
    4    NaN
    dtype: category
    Categories (5, object): [a, b, c, d, e]



#### （d）利用remove_categories移除


```python
s = pd.Series(pd.Categorical(["a", "b", "c", "a",np.nan], categories=['a','b','c','d']))
s.cat.remove_categories(['d'])
```




    0      a
    1      b
    2      c
    3      a
    4    NaN
    dtype: category
    Categories (3, object): [a, b, c]



#### （e）删除元素值未出现的分类类型


```python
s = pd.Series(pd.Categorical(["a", "b", "c", "a",np.nan], categories=['a','b','c','d']))
s.cat.remove_unused_categories()
```




    0      a
    1      b
    2      c
    3      a
    4    NaN
    dtype: category
    Categories (3, object): [a, b, c]



## 二、分类变量的排序
#### 前面提到，分类数据类型被分为有序和无序，这非常好理解，例如分数区间的高低是有序变量，考试科目的类别一般看做无序变量

### 1. 序的建立

#### （a）一般来说会将一个序列转为有序变量，可以利用as_ordered方法


```python
s = pd.Series(["a", "d", "c", "a"]).astype('category').cat.as_ordered()
s
```




    0    a
    1    d
    2    c
    3    a
    dtype: category
    Categories (3, object): [a < c < d]



#### 退化为无序变量，只需要使用as_unordered


```python
s.cat.as_unordered()
```




    0    a
    1    d
    2    c
    3    a
    dtype: category
    Categories (3, object): [a, c, d]



#### （b）利用set_categories方法中的order参数


```python
pd.Series(["a", "d", "c", "a"]).astype('category').cat.set_categories(['a','c','d'],ordered=True)
```




    0    a
    1    d
    2    c
    3    a
    dtype: category
    Categories (3, object): [a < c < d]



#### （c）利用reorder_categories方法
#### 这个方法的特点在于，新设置的分类必须与原分类为同一集合


```python
s = pd.Series(["a", "d", "c", "a"]).astype('category')
s.cat.reorder_categories(['a','c','d'],ordered=True)
```




    0    a
    1    d
    2    c
    3    a
    dtype: category
    Categories (3, object): [a < c < d]




```python
#s.cat.reorder_categories(['a','c'],ordered=True) #报错
#s.cat.reorder_categories(['a','c','d','e'],ordered=True) #报错
```

### 2. 排序

#### 先前在第1章介绍的值排序和索引排序都是适用的


```python
s = pd.Series(np.random.choice(['perfect','good','fair','bad','awful'],50)).astype('category')
s.cat.set_categories(['perfect','good','fair','bad','awful'][::-1],ordered=True).head()
```




    0       good
    1       fair
    2        bad
    3    perfect
    4    perfect
    dtype: category
    Categories (5, object): [awful < bad < fair < good < perfect]




```python
s.sort_values(ascending=False).head()
```




    29    perfect
    17    perfect
    31    perfect
    3     perfect
    4     perfect
    dtype: category
    Categories (5, object): [awful, bad, fair, good, perfect]




```python
df_sort = pd.DataFrame({'cat':s.values,'value':np.random.randn(50)}).set_index('cat')
df_sort.head()
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
      <th>value</th>
    </tr>
    <tr>
      <th>cat</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>good</th>
      <td>-1.746975</td>
    </tr>
    <tr>
      <th>fair</th>
      <td>0.836732</td>
    </tr>
    <tr>
      <th>bad</th>
      <td>0.094912</td>
    </tr>
    <tr>
      <th>perfect</th>
      <td>-0.724338</td>
    </tr>
    <tr>
      <th>perfect</th>
      <td>-1.456362</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sort.sort_index().head()
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
      <th>value</th>
    </tr>
    <tr>
      <th>cat</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>awful</th>
      <td>0.245782</td>
    </tr>
    <tr>
      <th>awful</th>
      <td>0.063991</td>
    </tr>
    <tr>
      <th>awful</th>
      <td>1.541862</td>
    </tr>
    <tr>
      <th>awful</th>
      <td>-0.062976</td>
    </tr>
    <tr>
      <th>awful</th>
      <td>0.472542</td>
    </tr>
  </tbody>
</table>
</div>



## 三、分类变量的比较操作

### 1. 与标量或等长序列的比较

#### （a）标量比较


```python
s = pd.Series(["a", "d", "c", "a"]).astype('category')
s == 'a'
```




    0     True
    1    False
    2    False
    3     True
    dtype: bool



#### （b）等长序列比较


```python
s == list('abcd')
```




    0     True
    1    False
    2     True
    3    False
    dtype: bool



### 2. 与另一分类变量的比较

#### （a）等式判别（包含等号和不等号）
#### 两个分类变量的等式判别需要满足分类完全相同


```python
s = pd.Series(["a", "d", "c", "a"]).astype('category')
s == s
```




    0    True
    1    True
    2    True
    3    True
    dtype: bool




```python
s != s
```




    0    False
    1    False
    2    False
    3    False
    dtype: bool




```python
s_new = s.cat.set_categories(['a','d','e'])
#s == s_new #报错
```

#### （b）不等式判别（包含>=,<=,<,>）
#### 两个分类变量的不等式判别需要满足两个条件：① 分类完全相同 ② 排序完全相同


```python
s = pd.Series(["a", "d", "c", "a"]).astype('category')
#s >= s #报错
```


```python
s = pd.Series(["a", "d", "c", "a"]).astype('category').cat.reorder_categories(['a','c','d'],ordered=True)
s >= s
```




    0    True
    1    True
    2    True
    3    True
    dtype: bool



## 四、问题与练习

#### 【问题一】 如何使用union_categoricals方法？它的作用是什么？
#### 【问题二】 利用concat方法将两个序列纵向拼接，它的结果一定是分类变量吗？什么情况下不是？
#### 【问题三】 当使用groupby方法或者value_counts方法时，分类变量的统计结果和普通变量有什么区别？
#### 【问题四】 下面的代码说明了Series创建分类变量的什么“缺陷”？如何避免？（提示：使用Series中的copy参数）


```python
cat = pd.Categorical([1, 2, 3, 10], categories=[1, 2, 3, 4, 10])
s = pd.Series(cat, name="cat")
cat
```




    [1, 2, 3, 10]
    Categories (5, int64): [1, 2, 3, 4, 10]




```python
s.iloc[0:2] = 10
cat
```




    [10, 10, 3, 10]
    Categories (5, int64): [1, 2, 3, 4, 10]



#### 【练习一】 现继续使用第四章中的地震数据集，请解决以下问题：
#### （a）现在将深度分为七个等级：[0,5,10,15,20,30,50,np.inf]，请以深度等级Ⅰ,Ⅱ,Ⅲ,Ⅳ,Ⅴ,Ⅵ,Ⅶ为索引并按照由浅到深的顺序进行排序。


```python
import pandas as pd
df = pd.read_csv('data/Earthquake.csv')
df.head(3)
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
      <th>日期</th>
      <th>时间</th>
      <th>维度</th>
      <th>经度</th>
      <th>方向</th>
      <th>距离</th>
      <th>深度</th>
      <th>烈度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003.05.20</td>
      <td>12:17:44 AM</td>
      <td>39.04</td>
      <td>40.38</td>
      <td>west</td>
      <td>0.1</td>
      <td>10.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007.08.01</td>
      <td>12:03:08 AM</td>
      <td>40.79</td>
      <td>30.09</td>
      <td>west</td>
      <td>0.1</td>
      <td>5.2</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1978.05.07</td>
      <td>12:41:37 AM</td>
      <td>38.58</td>
      <td>27.61</td>
      <td>south_west</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



#### （b）在（a）的基础上，将烈度分为4个等级：[0,3,4,5,np.inf]，依次对南部地区的深度和烈度等级建立多级索引排序。


```python

```


```python

```

#### 【练习二】 对于分类变量而言，调用第4章中的变形函数会出现一个BUG（目前的版本下还未修复）：例如对于crosstab函数，按照[官方文档的说法](https://pandas.pydata.org/pandas-docs/version/1.0.0/user_guide/reshaping.html#cross-tabulations)，即使没有出现的变量也会在变形后的汇总结果中出现，但事实上并不是这样，比如下面的例子就缺少了原本应该出现的行'c'和列'f'。基于这一问题，请尝试设计my_crosstab函数，在功能上能够返回正确的结果。


```python
foo = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])
bar = pd.Categorical(['d', 'e'], categories=['d', 'e', 'f'])
pd.crosstab(foo, bar)
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
      <th>col_0</th>
      <th>d</th>
      <th>e</th>
    </tr>
    <tr>
      <th>row_0</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


