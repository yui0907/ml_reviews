###  问题
#### 【问题一】 如何删除缺失值占比超过25%的列？



```python
import pandas as pd 
import numpy as np 
df_d = pd.DataFrame({'A':[np.nan,np.nan,np.nan,np.nan,1],'B':[np.nan,4,3,2,1],'C':[4,3,2,1,0]})
df_d
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>3.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
thresh = 0.25*df_d.shape[0]
df_d.dropna(axis=1,thresh=thresh)
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
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### 【问题二】 什么是Nullable类型？请谈谈为什么要引入这个设计？

    - 统一缺失值处理方法
    - 老版本的缺失值，在功能上比较混乱
        修改series的值时，如果含有np.nan，会改变数据类型，并且会导致某些数据类型转化困难；而None虽然不会改变数据结构（bool),但是传值时，也会变成np.nan；NaT是针对时间序列的缺失值，是Pandas的内置类型，是时序版本的np.nan。
        几种类型在功能上混乱和交叉，因此有了Nullable这个类型，使用该类型时，三种缺失值都会被替换为统一的NA符号，且不改变数据类型。

#### 【问题三】 对于一份有缺失值的数据，可以采取哪些策略或方法深化对它的了解？

    - 先整体统计每行每列缺失值个数
    - 行缺失较多的样本，可以理解为无效样本；不算多的情况，可以结合列缺失来分析；
    - 列缺失情况，可以从列名称、收集源信息、分组可视化  等等来交叉印证 属于随机缺失还是 非随机缺失，从而选择合适的缺失值处理技术

###  练习

#### 【练习一】现有一份虚拟数据集，列类型分别为string/浮点/整型，请解决如下问题：
#### （a）请以列类型读入数据，并选出C为缺失值的行。


```python
df= pd.read_csv('data/Missing_data_one.csv')
df[df.C.isna()]
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>not_NaN</td>
      <td>0.700</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>not_NaN</td>
      <td>0.972</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>not_NaN</td>
      <td>0.736</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>not_NaN</td>
      <td>0.684</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>not_NaN</td>
      <td>0.913</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



#### （b）现需要将A中的部分单元转为缺失值，单元格中的最小转换概率为25%，且概率大小与所在行B列单元的值成正比。


```python
import random
##将概率范围框定在[0.25-1]
k =(1.0-0.25)/(df.B.max()-df.B.min())
df['A_pro']=df.B.apply(lambda x:0.25+k*(x-df.B.min()))
df['A']=df.A_pro.apply(lambda x: np.nan if np.random.rand()< x else  'not_NaN' )
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>A_pro</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>0.922</td>
      <td>4.0</td>
      <td>0.914376</td>
    </tr>
    <tr>
      <th>1</th>
      <td>not_NaN</td>
      <td>0.700</td>
      <td>NaN</td>
      <td>0.562368</td>
    </tr>
    <tr>
      <th>2</th>
      <td>not_NaN</td>
      <td>0.503</td>
      <td>8.0</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>0.938</td>
      <td>4.0</td>
      <td>0.939746</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>0.952</td>
      <td>10.0</td>
      <td>0.961945</td>
    </tr>
  </tbody>
</table>
</div>



#### 【练习二】 现有一份缺失的数据集，记录了36个人来自的地区、身高、体重、年龄和工资，请解决如下问题：
#### （a）统计各列缺失的比例并选出在后三列中至少有两个非缺失值的行。


```python
df2 = pd.read_csv('data/Missing_data_two.csv')
##统计各列缺失值比例
df2.isna().sum()/df2.shape[0]
```




    编号    0.000000
    地区    0.000000
    身高    0.000000
    体重    0.222222
    年龄    0.250000
    工资    0.222222
    dtype: float64




```python
df2[df2.iloc[:,-3:].notna().sum(axis=1)>=2]
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
      <th>编号</th>
      <th>地区</th>
      <th>身高</th>
      <th>体重</th>
      <th>年龄</th>
      <th>工资</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>A</td>
      <td>157.50</td>
      <td>NaN</td>
      <td>47.0</td>
      <td>15905.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>B</td>
      <td>202.00</td>
      <td>91.80</td>
      <td>25.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>A</td>
      <td>166.61</td>
      <td>59.95</td>
      <td>77.0</td>
      <td>5434.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>B</td>
      <td>185.19</td>
      <td>NaN</td>
      <td>62.0</td>
      <td>4242.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>A</td>
      <td>187.13</td>
      <td>78.42</td>
      <td>55.0</td>
      <td>13959.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>C</td>
      <td>163.81</td>
      <td>57.43</td>
      <td>43.0</td>
      <td>6533.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>A</td>
      <td>183.80</td>
      <td>75.42</td>
      <td>48.0</td>
      <td>19779.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>B</td>
      <td>179.67</td>
      <td>71.70</td>
      <td>65.0</td>
      <td>8608.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>C</td>
      <td>186.08</td>
      <td>77.47</td>
      <td>65.0</td>
      <td>12433.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>B</td>
      <td>163.41</td>
      <td>57.07</td>
      <td>NaN</td>
      <td>6495.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>B</td>
      <td>175.99</td>
      <td>68.39</td>
      <td>NaN</td>
      <td>13130.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>A</td>
      <td>165.68</td>
      <td>NaN</td>
      <td>46.0</td>
      <td>13683.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>B</td>
      <td>166.48</td>
      <td>59.83</td>
      <td>31.0</td>
      <td>17673.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>C</td>
      <td>191.62</td>
      <td>82.46</td>
      <td>NaN</td>
      <td>12447.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>A</td>
      <td>172.83</td>
      <td>65.55</td>
      <td>23.0</td>
      <td>13768.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>B</td>
      <td>156.99</td>
      <td>51.29</td>
      <td>62.0</td>
      <td>3054.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>C</td>
      <td>200.22</td>
      <td>90.20</td>
      <td>41.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>A</td>
      <td>154.63</td>
      <td>49.17</td>
      <td>35.0</td>
      <td>14559.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>B</td>
      <td>157.87</td>
      <td>52.08</td>
      <td>67.0</td>
      <td>7398.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>A</td>
      <td>165.55</td>
      <td>NaN</td>
      <td>66.0</td>
      <td>19890.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>C</td>
      <td>181.78</td>
      <td>73.60</td>
      <td>63.0</td>
      <td>11383.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>A</td>
      <td>164.43</td>
      <td>57.99</td>
      <td>34.0</td>
      <td>19899.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>C</td>
      <td>172.39</td>
      <td>65.15</td>
      <td>43.0</td>
      <td>10362.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>B</td>
      <td>162.12</td>
      <td>55.91</td>
      <td>NaN</td>
      <td>13362.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>A</td>
      <td>183.73</td>
      <td>75.36</td>
      <td>58.0</td>
      <td>8270.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>31</td>
      <td>C</td>
      <td>181.19</td>
      <td>NaN</td>
      <td>41.0</td>
      <td>12616.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>32</td>
      <td>B</td>
      <td>167.28</td>
      <td>60.55</td>
      <td>64.0</td>
      <td>18317.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>35</td>
      <td>B</td>
      <td>170.12</td>
      <td>63.11</td>
      <td>77.0</td>
      <td>7398.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>36</td>
      <td>C</td>
      <td>180.47</td>
      <td>72.42</td>
      <td>78.0</td>
      <td>9554.0</td>
    </tr>
  </tbody>
</table>
</div>



#### （b）请结合身高列和地区列中的数据，对体重进行合理插值。

    - 根据地区分组，在组内按照身高进行线性插值


```python
df_res=pd.DataFrame(columns=df2.columns)
for _,sub_group in df2.groupby('地区'):
    sub_group=pd.DataFrame(sub_group)
    sub_group['体重']=sub_group[['身高','体重']].sort_values(by='身高').interpolate()['体重']
    df_res = pd.concat([df_res,sub_group],axis=0)
```


```python
df_res.sort_index().head()
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
      <th>编号</th>
      <th>地区</th>
      <th>身高</th>
      <th>体重</th>
      <th>年龄</th>
      <th>工资</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>A</td>
      <td>157.50</td>
      <td>53.58</td>
      <td>47.0</td>
      <td>15905.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>B</td>
      <td>202.00</td>
      <td>91.80</td>
      <td>25.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>C</td>
      <td>169.09</td>
      <td>62.18</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>A</td>
      <td>166.61</td>
      <td>59.95</td>
      <td>77.0</td>
      <td>5434.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>B</td>
      <td>185.19</td>
      <td>81.75</td>
      <td>62.0</td>
      <td>4242.0</td>
    </tr>
  </tbody>
</table>
</div>


