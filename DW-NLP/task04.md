### Contextual Word Embeddings
- 如何表示单词的意义：
>one-hot -> word embedding :word embedding 用一个简单的概率分布，来预测背景中所有的上下文单词.向量中的每个维度都包含了单词的某种语义
- 一词多义现象：
>上述词表征的方法，无法解决一次多义现象，即：一个在结构外形上一样的单词用一个词向量表示。
> 现实要求则希望出现：1)每个单词语义拥有一个embedding 2）上下文越接近的则embedding相似度越高
> 这就是**Contextual Word Embeddings**，实现方法就是ELMO:Embeddings from language model 



### ELMO:RNN-based language models 
在之前word2vec及GloVe的工作中，每个词对应一个vector，对于多义词无能为力，或者随着语言环境的改变，这些vector不能准确的表达相应特征。ELMo的作者认为好的词表征模型应该同时兼顾两个问题：一是词语用法在语义和语法上的复杂特点；二是随着语言环境的改变，这些用法也应该随之改变。

ELMo算法过程为：

1. 先在大语料上以language model为目标训练出bidirectional LSTM模型；
2. 然后利用LSTM产生词语的表征；

ELMo模型包含多layer的bidirectional LSTM，可以这么理解:

高层的LSTM的状态可以捕捉词语意义中和语境相关的那方面的特征(比如可以用来做语义的消歧)，而低层的LSTM可以找到语法方面的特征(比如可以做词性标注)。

#### Bidirectional language models

![](https://pirctures.oss-cn-beijing.aliyuncs.com/img/1.png)

ELMo模型有两个比较关键的公式：

![gongshi_1](https://pirctures.oss-cn-beijing.aliyuncs.com/img/gongshi_1.png)

![image-20200607195155431](https://pirctures.oss-cn-beijing.aliyuncs.com/img/image-20200607195155431.png)

这里可以看出预测句子概率
$$
p(t1,t2,...,tn)
$$
有两个方向：正方向和反方向。

(t1,t2,...tn)是一系列的tokens 。

设输入token的表示为![image-20200607201058404](https://pirctures.oss-cn-beijing.aliyuncs.com/img/image-20200607201058404.png),在每一个位置k，每一层LSTM上都输出相应的context-dependent的表征![image-20200607201217757](https://pirctures.oss-cn-beijing.aliyuncs.com/img/image-20200607201217757.png)，这里 j 代表LSTM的某层layer，例如顶层的LSTM的输出可以表示为：![image-20200607201432357](https://pirctures.oss-cn-beijing.aliyuncs.com/img/image-20200607201432357.png)，我们对最开始的两个概率求对数极大似然估计，即：

![image-20200607201838599](https://pirctures.oss-cn-beijing.aliyuncs.com/img/image-20200607201838599.png)

这里的Θ*x*代表token embedding, Θ*s*代表softmax layer的参数。

#### word feature

对于每一个token，一个L层的biLM模型要计算出共2*L*+1个表征： 

![image-20200607202244219](https://pirctures.oss-cn-beijing.aliyuncs.com/img/image-20200607202244219.png)第二个“=”可以理解为：

当 j=0 时，代表token层。当 j>0 时，同时包括两个方向的hidden表征。

### GPT

这部分介绍GPT模型([原文](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)).

GPT的训练分为两个阶段：1）无监督预训练语言模型；2）各个任务的微调。

模型结构图：

![](https://pirctures.oss-cn-beijing.aliyuncs.com/img/2.png)

#### 1.无监督pretrain

使用语言模型最大化下面的式子：

![image-20200607204843715](https://pirctures.oss-cn-beijing.aliyuncs.com/img/image-20200607204843715.png)

其中 *k* 是上下文窗口大小，*θ* 是语言模型参数，使用一个神经网络来模拟条件概率 *P*

在论文中，使用一个多层的transformer decoder来作为LM(语言模型)，可以看作transformer的变体。将transformer  decoder中Encoder-Decoder Attention层去掉作为模型的主体，然后将decoder的输出经过一个softmax层，来得到目标词的输出分布：

![image-20200607205533296](https://pirctures.oss-cn-beijing.aliyuncs.com/img/image-20200607205533296.png)

这里![image-20200607205656843](https://pirctures.oss-cn-beijing.aliyuncs.com/img/image-20200607205656843.png)表示 *ui* 的上下文，![image-20200607205825579](https://pirctures.oss-cn-beijing.aliyuncs.com/img/image-20200607205825579.png)是词向量矩阵，![image-20200607205855307](https://pirctures.oss-cn-beijing.aliyuncs.com/img/image-20200607205855307.png)是位置向量矩阵。

#### 2.有监督finetune

在这一步，我们根据自己的任务去调整预训练语言模型的参数 *θ*，

![image-20200607210752637](https://pirctures.oss-cn-beijing.aliyuncs.com/img/image-20200607210752637.png)

最后优化的式子为：

![image-20200607210919284](https://pirctures.oss-cn-beijing.aliyuncs.com/img/image-20200607210919284.png)

在自己的任务中，使用遍历式的方法将结构化输入转换成预训练语言模型能够处理的有序序列：

![](https://pirctures.oss-cn-beijing.aliyuncs.com/img/3.png)

### BERT

Bert([原文](https://arxiv.org/abs/1810.04805
))是谷歌的大动作，公司AI团队新发布的BERT模型，在机器阅读理解顶级水平测试SQuAD1.1中表现出惊人的成绩：全部两个衡量指标上全面超越人类，并且还在11种不同NLP测试中创出最佳成绩，包括将GLUE基准推至80.4％（绝对改进7.6％），MultiNLI准确度达到86.7% （绝对改进率5.6％）等。可以预见的是，BERT将为NLP带来里程碑式的改变，也是NLP领域近期最重要的进展。

BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder。模型的主要创新点都在pre-train方法上，即用了Masked LM和Next Sentence Prediction两种方法分别捕捉词语和句子级别的representation。

BERT采用了Transformer Encoder的模型来作为语言模型，Transformer模型来自于经典论文《Attention is all you need》, 完全抛弃了RNN/CNN等结构，而完全采用Attention机制来进行input-output之间关系的计算，如下图中左半边部分所示：

![](https://pirctures.oss-cn-beijing.aliyuncs.com/img/4.png)

Bert模型结构如下：

![](https://pirctures.oss-cn-beijing.aliyuncs.com/img/5.png)


```python

```
