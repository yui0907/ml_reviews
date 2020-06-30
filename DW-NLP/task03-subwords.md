### Subword Models

#### 语言学的一部分

##### 人类语言的声音：语音学和音韵学

- 语音学：研究对象是声音流
- 语音学：假设人类语音中存在一些基本分类单位，例如音素，以及其它特征。

##### 形态学：词的一部分

- 将词素（morphemes）视为最小的语义单位。

  - [[un [[fortun（e）] ROOT ate] STEM] STEM ly] WORD

- 深度学习很少研究形态学

- 不同语言的书写形式具有很大差异，例如中文词语之间没有空格，而英文词语之间会通过空格隔开。

  

#### 字符级模型 

- 字符模型需要克服这些困难：
  - 数量巨大的开放单词表，包括词语形态的变换（Rich morphology）、音译现象（Transliteration），以及非正式表达（informal spelling），例如缩写，网络用语等。
- 两种实现方式：
  - 构建的是一个对单词有效的系统，其中仍然有文字，在此基础上，还能得到任何字符序列的单词表示。模型需要识别看起来熟悉的字符序列部分。
    - 未未知的单词生成词嵌入
    - 用未知的单词，得到相似的单词。共享相似的词嵌入。
    - 解决未登录词问题。
  - 将语言分解为字符序列进行处理

#### 纯字符级别模型 

- **字符级别模型的问题**：序列变得更长(约是之前7倍);因为没有太多的信息，数据之间的联系更小，需要在更远的地方往回传播：导致训练起来十分慢。(虽然能够获得较好的效果)

- **进展1**：Fully Character-Level Neural Machine  Translation without Explicit Segmentation-2017

  - 编码器如下图；解码器则只是一个  char-level GRU

  ![1593495688565](pic\1593495688565.png)s

  - 实现：
    - 对输入的单词首先做一个字符级别的embeddings
    - 用核大小为3，4，5的卷积层扫描，得到3，4，5个字符的表达；
    - 以步幅5进行最大池化，得到egment-embedding
    - 将这些embedding送入4层Highway Network
    - 再通过一个单层的双向GRU，得到最终的encoder的output。
    - 最后经过一个character-level的GRU解码器，得到最终结果。
  - 成果：
    - 更好的分数
    - 训练更快 

- **进展2**：Stronger character results with depth in LSTM seq2seq model-2018

  - 比较基于单词核基于字符的模型

    ![1593497042411](pic\1593497042411.png)

#### 子词模型 

- 介于字符级核单词级的一种表示
  - 与word-level相同的结构，但是使用更小的单元word pieces来代替单词；
  - hybrid 结构, 主要部分依然是基于word, 但是其他的一些部分（unknown部分）用characters。

#### BPE:Byte Pair Encoding 

- 重复将出现频率最高的n-gram的pair作为新的n-gram加入词汇库中,，直到达到我们的要求。

#### Hybrid character and word-level models

- 大部分时候都使用word-level的模型来做translate，只有在遇到rare or unseen的words的时候才会使用character-level的模型协助。

  ![1593498873546](pic\1593498873546.png)

- 比如该例子中，若cute是一个out of vocabulary的单词，我们就需要使用character-level的模型去处理。在decode过程中，如果发现<unk>，说明需要character-level的decode, 最后的损失函数是word-level部分和character-level部分的加权叠加。

- 同时，在decoding过程中，在word-level部分和character-level部分均使用了beam search的方法，选取topK可能性的字符或单词。

#### FastText 

- 在word2vec方法中我们基于word-level的模型来得到每一个单词的embedding,但是对于含有许多OOV单词的文本库word2vec的效果并不好

- FastText方法就是汲取了subword的思想，它将每个单词转变为对于character的n-gram和该单词本身的集合。

- 例如，对于单词 “<where>”，以及n=3。则集合可以表示为 ${<wh,whe,her,ere,re>,<where>}$(其中<>代表单词的开始与结束)。

- 对于每个单词 ![[公式]](https://www.zhihu.com/equation?tex=w) ,其对应集合可用 ![[公式]](https://www.zhihu.com/equation?tex=G_%7Bw%7D) 来表示。设该集合每个n-gram表示为 ![[公式]](https://www.zhihu.com/equation?tex=z_g) ,则每个单词可以表示为其所有n-gram矢量和的形式，则center word和context word 间的相似度可表示为：
  $$
  s(w,c)=\sum_{g\in G_{w}}Z_{g}^TV_{c}
  $$

- 就可以使用原有的word2vec算法来训练得到对应单词的embedding。其保证了算法速度快的同时，解决了OOV的问题，是很好的算法。

