## word2vec
### 1.词的表示 
- 独热编码(离散符号）：向量之间是相互正交的，无法体现出向量之间的相似性
- 同义词词典：例如wordnet,但也有制作和更新的工作量庞大；部分词和词组不包含；对于形容词和动词效果不好 等缺点
- 词向量表示（也叫词嵌入）：在向量空间表示每一个词，意思相近的词用相近的向量表示

### 2.词向量
- 稀疏向量表示：词共现矩阵-在词之间连续出现次数进行计数，面临稀疏性问题、向量维数随着词典大小线性增长。解决：SVD、PCA降维，但是计算量大
- 稠密向量表示：奇异值分解、**受神经网络启发的模型（S-G，CBOW）**；elmo和bert
- 神经网络词嵌入的优点-短稠密向量 ：短向量便于机器学习使用，稠密向量具有更好的泛化能力，可以通过有监督的学习包含更多的含义

### 3.word2vec：学习词嵌入的一种框架
- **基本思想**

>利用大量的文本语料、每一个词汇中的词被表示为一个向量
>在遍历文本的过程中，每一个当前位置的词称为中心词，中心词周围的称为上下文词
>利用中心词和上下文词的相似性，计算**中心词**在给定**上下文词后**(或者相反)的概率
>不断更新词向量使得上述概率最大化

- **skip-grams算法**
基于 中心词预测上下文词

给定中心词c，上下文o出现的概率可以用$u_{o}$和 $v_{c}$ 的余弦相似性表示，并作归一化处理
![Image Name](https://cdn.kesci.com/upload/image/q67kqgdc2.png?imageView2/0/w/640/h/640)
![Image Name](https://cdn.kesci.com/upload/image/q67lbakjge.png?imageView2/0/w/640/h/640)
![Image Name](https://cdn.kesci.com/upload/image/q67lkcqsbb.png?imageView2/0/w/640/h/640)

在最优化过程中，最耗时的是计算softmax-- 引入负采样机制，基本思想：针对一个正样本（中心词c+上下文o）和多对负样本（中心词c+一个随机词）训练一个二元逻辑回归，损失函数相应地发生了改变

![Image Name](https://cdn.kesci.com/upload/image/q67lxryxad.png?imageView2/0/w/640/h/640)
对于每个词都有2个词嵌入u和v，可以仅使用v，或者将u和v相加，或者拼接在一起构成一个双倍的词嵌入

- **CBOW算法**
连续词袋算法，上下文词预测中心词


- **评测**
内部任务评测：除非能和实际任务建立明确的联系，否则不能确定是否真的有用
外部任务评测：如果效果不好，不能确定是因为词向量本身还是使用方式
词向量类比：根据词嵌入加减后与另一个词嵌入之间的余弦距离是否符合直观的语义来评价词嵌入的好坏。a与b相减后与c的余弦距离

![Image Name](https://cdn.kesci.com/upload/image/q67mlump1y.png?imageView2/0/w/320/h/320)



### 4.word2vec使用与可视化


![Image Name](https://cdn.kesci.com/upload/image/q67o6uj4d7.png?imageView2/0/w/640/h/640)

![Image Name](https://cdn.kesci.com/upload/image/q67o7niocd.png?imageView2/0/w/640/h/640)

![Image Name](https://cdn.kesci.com/upload/image/q67o81zpf1.png?imageView2/0/w/640/h/640)

![Image Name](https://cdn.kesci.com/upload/image/q67o8d9kko.png?imageView2/0/w/640/h/640)

![Image Name](https://cdn.kesci.com/upload/image/q67o8qv5ry.png?imageView2/0/w/640/h/640)

![Image Name](https://cdn.kesci.com/upload/image/q67o93pqfy.png?imageView2/0/w/640/h/640)

```
##读入语料 ##Text8Corpus 是一个英文语料库，单词清洗和分词
sentences= gensim.models.word2vec.Text8Corpus('/kaggle/input/word2vec-data/text8/text8')

##训练word2vec
model = gensim.models.word2vec.Word2Vec(sentences,size=300)
model.save('text8.w2w')

##装载词向量
model = gensim.models.Word2Vec.load("text8.w2v")
# 装载词向量
all_word_vector = model[model.wv.vocab]

start_word = 'apple'
topn = 50
pca = sklearn.decomposition.PCA(n_components=3)
pca.fit(all_word_vector)
# 收集与start_word最相似的词向量
similar_word_list = [start_word] + [pair[0] for pair in model.most_similar(start_word, topn=topn)]
similar_word_vector =  [model[word] for word in similar_word_list]
# 降维
decomposed_vector = pca.transform(similar_word_vector)


# 设置坐标图中画出的点的坐标，文本标注的位置和颜色
trace = plotly.graph_objs.Scatter3d(
    x=decomposed_vector[:, 0],
    y=decomposed_vector[:, 1],
    z=decomposed_vector[:, 2],
    mode="markers+text",
    text=similar_word_list,
    textposition="bottom center",
    marker=dict(
        color=[256 - int(numpy.linalg.norm(decomposed_vector[i] - decomposed_vector[0])) for i in range(len(similar_word_list))]
    )
)
layout = plotly.graph_objs.Layout(
    title="Top " + str(topn) + " Word Most Similar With \"" + start_word + "\""
)
data = [trace]
figure = plotly.graph_objs.Figure(data=data, layout=layout)
graph_name = "word2vec.html"
# 绘图
plotly.offline.plot(figure, filename=graph_name, auto_open=False)

##查看与之相对最相似的词
model.most_similar(positive=['does','have'], negative=['do'])
model.most_similar(positive=['woman', 'king'], negative=['man'])
```

### 文本分类
#### 1.文本表示
- **词序列** ：需要使用序列模型进行处理
- **稀疏向量特征** ：可用于大多数分类算法，最常见的是词袋模型（忽略了词之间的顺序关系）
- **稠密特征向量**
	由词嵌入计算得到：对每个单词计算一个向量，用较短的向量表示一个单词
	然后对一句话中的所有单词进行汇总，得到该句子的向量
	例如：对句中词汇的词向量取平均，或者对这些词向量跑LSTM（更加有意义）

#### 2.稀疏特征的向量表示（基于词袋模型）

the fat cat sat on the mat 
- **词频**:在文档中出现的次数
	$f_{fat}(x)=1,f_{cat}(x)=1,f_{the}(x)=2,f_{car}(x)=0$
- **词存在**：词频的简化，判断词存在与否，得到的是2元向量
	$f_{fat}(x)=1,f_{cat}(x)=1,f_{the}(x)=1,f_{car}(x)=0$
- **词频的变换，例如逆文本频率指数**
词频除以出点过的文档数，然后取对数。
一个词如果只在极少的文档中出现，那么这个单词对于该文本来说，很有信息量。
如果一个单词在所有文档中都出现，比如一些虚词，即使出现的频率很高，也不具有分类的信息。
![Image Name](https://bkimg.cdn.bcebos.com/pic/4afbfbedab64034f258b7c13a2c379310a551d64?x-bce-process=image/resize,m_lfit,w_268,limit_1)

#### 3.文本分类模型
- **生成模型（建模p(x,y)）** : 建模文档与类别的联合概率分布
	- 朴素贝叶斯
- **判别式模型（建模p(y|x)）** : 给定句子的条件下，类别的条件概率分布
	- 逻辑回归、SVM、NN、决策树
	
- **评价指标**
 - 测试数据：一组独立于训练数据，用作分类性能评价的数据
 - 指标：
	 	准确率（类别不均衡问题、无法考虑类别间的相对重要性与不同误差的代价、测试数据本身的随机性（得到不同的衡量值）） 
	精度与召回：都是针对目标类别的评价指标