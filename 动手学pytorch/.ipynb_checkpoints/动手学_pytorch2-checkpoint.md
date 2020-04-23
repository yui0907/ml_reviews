## 模型选择、过拟合和欠拟合
- **模型选择**
	1.**验证数据集**
	从严格意义上讲，测试集只能在所有超参数和模型参数选定后使用一次。不可以使用测试数据选择模型，如调参。由于无法从训练误差估计泛化误差，因此也不应只依赖训练数据选择模型。鉴于此，我们可以预留一部分在训练数据集和测试数据集以外的数据来进行模型选择。这部分数据被称为验证数据集，简称验证集（validation set）。例如，我们可以从给定的训练集中随机选取一小部分作为验证集，而将剩余部分作为真正的训练集。
	2.**K折交叉验证**   
	由于验证数据集不参与模型训练，当训练数据不够用时，预留大量的验证数据显得太奢侈。一种改善的方法是K折交叉验证（K-fold cross-validation）。在K折交叉验证中，我们把原始训练数据集分割成K个不重合的子数据集，然后我们做K次模型训练和验证。每一次，我们使用一个子数据集验证模型，并使用其他K-1个子数据集来训练模型。在这K次训练和验证中，每次用来验证模型的子数据集都不同。最后，我们对这K次训练误差和验证误差分别求平均。


- **过拟合和欠拟合** 
	模型训练中经常出现的两类典型问题：
一类是模型无法得到较低的训练误差，我们将这一现象称作欠拟合（underfitting）；
另一类是模型的训练误差远小于它在测试数据集上的误差，我们称该现象为过拟合（overfitting）。
在实践中，我们要尽可能同时应对欠拟合和过拟合。
	1.**模型复杂度**
	给定一个由标量数据特征$x$和对应的标量标签$y$组成的训练数据集，多项式函数拟合的目标是找一个$K$阶多项式函数
$$
 \hat{y} = b + \sum_{k=1}^K x^k w_k 
$$
来近似 $y$。在上式中，$w_k$是模型的权重参数，$b$是偏差参数。与线性回归相同，多项式函数拟合也使用平方损失函数。特别地，一阶多项式函数拟合又叫线性函数拟合。
给定训练数据集，模型复杂度和误差之间的关系：
![Image Name](https://cdn.kesci.com/upload/image/q5jc27wxoj.png?imageView2/0/w/960/h/960)
	2.**训练数据集大小**
	影响欠拟合和过拟合的另一个重要因素是训练数据集的大小。一般来说，如果训练数据集中样本数过少，特别是比模型参数数量（按元素计）更少时，过拟合更容易发生。此外，泛化误差不会随训练数据集里样本数量增加而增大。因此，在计算资源允许的范围之内，我们通常希望训练数据集大一些，特别是在模型复杂度较高时，例如层数较多的深度学习模型。
	
- **解决方法**
	1.**正则化**
	通过为模型损失函数添加惩罚项使学出的模型参数值较小，是应对过拟合的常用手段。
	$L_2$范数正则化令**权重$w_1$和$w_2$先自乘小于1的数，再减去不含惩罚项的梯度**。因此，$L_2$范数正则化又叫**权重衰减**。权重衰减通过惩罚绝对值较大的模型参数为需要学习的模型增加了限制，这可能对过拟合有效。
	2.**丢弃法**
	多层感知机中神经网络图描述了一个单隐藏层的多层感知机。其中输入个数为4，隐藏单元个数为5，且隐藏单元$h_i$（$i=1, \ldots, 5$）的计算表达式为
$$
 h_i = \phi\left(x_1 w_{1i} + x_2 w_{2i} + x_3 w_{3i} + x_4 w_{4i} + b_i\right) 
$$
这里$\phi$是激活函数，$x_1, \ldots, x_4$是输入，隐藏单元$i$的权重参数为$w_{1i}, \ldots, w_{4i}$，偏差参数为$b_i$。当对该隐藏层使用丢弃法时，该层的隐藏单元将有一定概率被丢弃掉。设丢弃概率为$p$，那么有$p$的概率$h_i$会被清零，有$1-p$的概率$h_i$会除以$1-p$做拉伸。丢弃概率是丢弃法的超参数。具体来说，设随机变量$\xi_i$为0和1的概率分别为$p$和$1-p$。使用丢弃法时我们计算新的隐藏单元$h_i'$
$$
 h_i' = \frac{\xi_i}{1-p} h_i 
$$
由于$E(\xi_i) = 1-p$，因此
$$
 E(h_i') = \frac{E(\xi_i)}{1-p}h_i = h_i 
$$
即丢弃法不改变其输入的期望值。让我们对之前多层感知机的神经网络中的隐藏层使用丢弃法，一种可能的结果如图所示，其中$h_2$和$h_5$被清零。这时输出值的计算不再依赖$h_2$和$h_5$，在反向传播时，与这两个隐藏单元相关的权重的梯度均为0。由于在训练中隐藏层神经元的丢弃是随机的，即$h_1, \ldots, h_5$都有可能被清零，输出层的计算无法过度依赖$h_1, \ldots, h_5$中的任一个，从而在训练模型时起到正则化的作用，并可以用来应对过拟合。在测试模型时，我们为了拿到更加确定性的结果，一般不使用丢弃法
![Image Name](https://cdn.kesci.com/upload/image/q5jd69in3m.png?imageView2/0/w/960/h/960)


```python
def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()
    
    return mask * X / keep_prob
    
drop_prob1, drop_prob2 = 0.2, 0.5
def net(X, is_training=True):
    X = X.view(-1, num_inputs)
    H1 = (torch.matmul(X, W1) + b1).relu()
    if is_training:  # 只在训练模型时使用丢弃法
        H1 = dropout(H1, drop_prob1)  # 在第一层全连接后添加丢弃层
    H2 = (torch.matmul(H1, W2) + b2).relu()
    if is_training:
        H2 = dropout(H2, drop_prob2)  # 在第二层全连接后添加丢弃层
    return torch.matmul(H2, W3) + b3
```

## 梯度消失和梯度爆炸
深度模型有关数值稳定性的典型问题是消失（vanishing）和爆炸（explosion）。
- **当神经网络的层数较多时，模型的数值稳定性容易变差。**
	假设一个层数为$L$的多层感知机的第$l$层$\boldsymbol{H}^{(l)}$的权重参数为$\boldsymbol{W}^{(l)}$，输出层$\boldsymbol{H}^{(L)}$的权重参数为$\boldsymbol{W}^{(L)}$。为了便于讨论，不考虑偏差参数，且设所有隐藏层的激活函数为恒等映射（identity mapping）$\phi(x) = x$。给定输入$\boldsymbol{X}$，多层感知机的第$l$层的输出$\boldsymbol{H}^{(l)} = \boldsymbol{X} \boldsymbol{W}^{(1)} \boldsymbol{W}^{(2)} \ldots \boldsymbol{W}^{(l)}$。此时，如果层数$l$较大，$\boldsymbol{H}^{(l)}$的计算可能会出现衰减或爆炸。举个例子，假设输入和所有层的权重参数都是标量，如权重参数为0.2和5，多层感知机的第30层输出为输入$\boldsymbol{X}$分别与$0.2^{30} \approx 1 \times 10^{-21}$（消失）和$5^{30} \approx 9 \times 10^{20}$（爆炸）的乘积。当层数较多时，梯度的计算也容易出现消失或爆炸。
	
- **随机初始化模型参数的原因**
	假设输出层只保留一个输出单元$o_1$（删去$o_2$和$o_3$以及指向它们的箭头），且隐藏层使用相同的激活函数。如果将每个隐藏单元的参数都初始化为相等的值，那么在正向传播时每个隐藏单元将根据相同的输入计算出相同的值，并传递至输出层。在反向传播中，每个隐藏单元的参数梯度值相等。因此，这些参数在使用基于梯度的优化算法迭代后值依然相等。之后的迭代也是如此。在这种情况下，无论隐藏单元有多少，隐藏层本质上只有1个隐藏单元在发挥作用。因此，正如在前面的实验中所做的那样，我们通常将神经网络的模型参数，特别是权重参数，进行随机初始化。
	1. PyTorch的默认随机初始化 
	 在线性回归的简洁实现中，我们使用`torch.nn.init.normal_()`使模型`net`的权重参数采用正态分布的随机初始化方式。不过，PyTorch中`nn.Module`的模块参数都采取了较为合理的初始化策略（不同类型的layer具体采样的哪一种初始化方法的可参考[源代码](https://github.com/pytorch/pytorch/tree/master/torch/nn/modules)），因此一般不用我们考虑
	2. Xavier随机初始化
	假设某全连接层的输入个数为$a$，输出个数为$b$，Xavier随机初始化将使该层中权重参数的每个元素都随机采样于均匀分布
$$
U\left(-\sqrt{\frac{6}{a+b}}, \sqrt{\frac{6}{a+b}}\right).
$$
它的设计主要考虑到，模型参数初始化后，每层输出的方差不该受该层输入个数影响，且每层梯度的方差也不该受该层输出个数影响。

## 循环神经网络进阶

### GRU
RNN存在的问题：梯度较容易出现衰减或爆炸（BPTT）  
GRU⻔控循环神经⽹络：捕捉时间序列中时间步距离较⼤的依赖关系 ,解决长期记忆和反向传播中的梯度等问题。
• 重置⻔有助于捕捉时间序列⾥短期的依赖关系；  
• 更新⻔有助于捕捉时间序列⾥⻓期的依赖关系。 

- **GRU的输入输出结构**
GRU的输入输出结构与普通的RNN是一样的。
有一个当前的输入$x^t$ ，和上一个节点传递下来的隐状态（hidden state）$h^{t-1}$，这个隐状态包含了之前节点的相关信息。
结合$x^t$ 和$h^{t-1}$，GRU会得到当前隐藏节点的输出$y^t$ 和传递给下一个节点的隐状态$h^t$。
![Image Name](https://cdn.kesci.com/upload/image/q5srwnnmv7.png?imageView2/0/w/360/h/360)

- **GRU的内部结构**
首先，我们先通过上一个传输下来的状态 $h^{t-1}$ 和当前节点的输入$x^t$ 来获取两个门控状态。如下图所示，其中$r$控制重置的门控（reset gate），$z$为控制更新的门控（update gate）。
![Image Name](https://cdn.kesci.com/upload/image/q5ss7lod3r.png?imageView2/0/w/360/h/360)

得到门控信号之后，首先使用重置门控来得到“重置”之后的数据$h^{t-1'}=h^{t-1}⊙r$ ，再将$h^{t-1'}$与输入 $x^r$ 进行拼接，再通过一个tanh激活函数来将数据放缩到-1~1的范围内。即得到如下图所示的$h'$ 。
![Image Name](https://cdn.kesci.com/upload/image/q5ssjfuxhf.png?imageView2/0/w/960/h/960)

这里的$h'$主要是包含了当前输入的$x^t$数据。有针对性地对$h'$添加到当前的隐藏状态，相当于”记忆了当前时刻的状态“
![Image Name](https://cdn.kesci.com/upload/image/q5ssnlsje7.png?imageView2/0/w/960/h/960)

最后介绍GRU最关键的一个步骤**更新记忆**阶段。

在这个阶段，我们同时进行了遗忘了记忆两个步骤。我们使用了先前得到的更新门控$z$（update gate）。
更新表达式：$h^t = z ⊙h^{t-1}+(1-z)⊙h^{'}$

首先再次强调一下，门控信号（z）的范围为0~1。门控信号越接近1，代表”记忆“下来的数据越多；而越接近0则代表”遗忘“的越多。

GRU很聪明的一点就在于，我们使用了同一个门控$z$就同时可以进行遗忘和选择记忆（LSTM则要使用多个门控）。

**$z ⊙h^{t-1}$**：表示对原本隐藏状态的选择性“遗忘”。这里的$z$可以想象成遗忘门（forget gate），忘记$h^{t-1}$维度中一些不重要的信息。

**$(1-z)⊙h'$** ： 表示对包含当前节点信息的$h'$进行选择性”记忆“。与上面类似，这里的$(1-z)$同理会忘记$h'$ 维度中的一些不重要的信息。或者，这里我们更应当看做是对h'$维度中的某些信息进行选择。

**$h^t = z ⊙h^{t-1}+(1-z)⊙h^{'}$**:结合上述，这一步的操作就是忘记传递下来的 $h^{t-1}$中的某些维度信息，并加入当前节点输入的某些维度信息。
可以看到，这里的遗忘$z$和选择$(1-z)$是联动的。也就是说，对于传递进来的维度信息，我们会进行选择性遗忘，则遗忘了多少权重 （z），我们就会使用包含当前输入的$h'$中所对应的权重进行弥补(1-z)。以保持一种”恒定“状态。




```python
##初始化参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)

def get_params():  
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32) #正态分布
        return torch.nn.Parameter(ts, requires_grad=True)
    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))
     
    W_xz, W_hz, b_z = _three()  # 更新门参数
    W_xr, W_hr, b_r = _three()  # 重置门参数
    W_xh, W_hh, b_h = _three()  # 候选隐藏状态参数
    
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q])

def init_gru_state(batch_size, num_hiddens, device):   #隐藏状态初始化
    return (torch.zeros((batch_size, num_hiddens), device=device), )
    
##GRU模型
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z)
        R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r)
        H_tilda = torch.tanh(torch.matmul(X, W_xh) + R * torch.matmul(H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)
```

### LSTM
长短期记忆（Long short-term memory, LSTM）是一种特殊的RNN，主要是为了解决长序列训练过程中的梯度消失和梯度爆炸问题。简单来说，就是相比普通的RNN，LSTM能够在更长的序列中有更好的表现。

遗忘门:控制上一时间步的记忆细胞 
输入门:控制当前时间步的输入  
输出门:控制从记忆细胞到隐藏状态  
记忆细胞：⼀种特殊的隐藏状态的信息的流动 

- **LSTM结构**

![Image Name](https://cdn.kesci.com/upload/image/q5stl6xcdz.png?imageView2/0/w/960/h/960)


![Image Name](https://cdn.kesci.com/upload/image/q5stmqibvz.png?imageView2/0/w/960/h/960)

以上，就是LSTM的内部结构。通过门控状态来控制传输状态，记住需要长时间记忆的，忘记不重要的信息；而不像普通的RNN那样只能够“呆萌”地仅有一种记忆叠加方式。对很多需要“长期记忆”的任务来说，尤其好用。

但也因为引入了很多内容，导致参数变多，也使得训练难度加大了很多。因此很多时候我们往往会使用效果和LSTM相当但参数更少的GRU来构建大训练量的模型。


```python
##初始化参数、隐层、模型定义
def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))
          
    W_xi, W_hi, b_i = _three()  # 输入门参数
    W_xf, W_hf, b_f = _three()  # 遗忘门参数
    W_xo, W_ho, b_o = _three()  # 输出门参数
    W_xc, W_hc, b_c = _three()  # 候选记忆细胞参数
          
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q])
          
          
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), 
            torch.zeros((batch_size, num_hiddens), device=device))
                  
      
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid(torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i)
        F = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)
        O = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
        C_tilda = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * C.tanh()
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, C)
```

### 深度循环神经网络  

![Image Name](https://cdn.kesci.com/upload/image/q5jk3z1hvz.png?imageView2/0/w/320/h/320)

$$
\boldsymbol{H}_t^{(1)} = \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh}^{(1)} + \boldsymbol{H}_{t-1}^{(1)} \boldsymbol{W}_{hh}^{(1)} + \boldsymbol{b}_h^{(1)})\\
\boldsymbol{H}_t^{(\ell)} = \phi(\boldsymbol{H}_t^{(\ell-1)} \boldsymbol{W}_{xh}^{(\ell)} + \boldsymbol{H}_{t-1}^{(\ell)} \boldsymbol{W}_{hh}^{(\ell)} + \boldsymbol{b}_h^{(\ell)})\\
\boldsymbol{O}_t = \boldsymbol{H}_t^{(L)} \boldsymbol{W}_{hq} + \boldsymbol{b}_q
$$

### 双向循环神经网络 

![Image Name](https://cdn.kesci.com/upload/image/q5j8hmgyrz.png?imageView2/0/w/320/h/320)

$$ 
\begin{aligned} \overrightarrow{\boldsymbol{H}}_t &= \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh}^{(f)} + \overrightarrow{\boldsymbol{H}}_{t-1} \boldsymbol{W}_{hh}^{(f)} + \boldsymbol{b}_h^{(f)})\\
\overleftarrow{\boldsymbol{H}}_t &= \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh}^{(b)} + \overleftarrow{\boldsymbol{H}}_{t+1} \boldsymbol{W}_{hh}^{(b)} + \boldsymbol{b}_h^{(b)}) \end{aligned} $$
$$
\boldsymbol{H}_t=(\overrightarrow{\boldsymbol{H}}_{t}, \overleftarrow{\boldsymbol{H}}_t)
$$
$$
\boldsymbol{O}_t = \boldsymbol{H}_t \boldsymbol{W}_{hq} + \boldsymbol{b}_q
$$

## 机器翻译
机器翻译（MT）：将一段文本从一种语言自动翻译为另一种语言，用神经网络解决这个问题通常称为神经机器翻译（NMT）。
主要特征：输出是单词序列而不是单个单词。 
困难：输出序列的长度可能与源序列的长度不同。
### 数据处理
1.数据清洗：处理乱码与空格。
2.分词：将字符串转换成单词组成的列表
3.建立词典，将单词组成的列表编程单词id组成的列表，这里会得到如下几样东西
		> 去重后词典，及其中单词对应的索引列表
		> 还可以得到给定索引找到其对应的单词的列表，以及给定单词得到对应索引的字典。
		> 原始语料所有词对应的词典索引的列表
4.padding:一个batch中所有句子输入长度保持一致
	```
	def pad(line, max_len, padding_token):
			if len(line) > max_len:
					return line[:max_len]
			return line + [padding_token] * (max_len - len(line))
	```
5.加载数据：数据生成器

### Encoder-Decoder
encoder：输入到隐藏状态  
decoder：隐藏状态到输出
对话系统、生成式任务、翻译

####  Sequence to Sequence模型
seq2seq模型基于编码器-解码器体系结构，通过序列输入生成序列输出。 
编码器和解码器都使用递归神经网络（RNN）处理可变长度的序列输入。 
编码器的隐藏状态直接用于初始化解码器的隐藏状态，以将信息从编码器传递到解码器。

训练  
![Image Name](https://cdn.kesci.com/upload/image/q5jc7a53pt.png?imageView2/0/w/640/h/640)
预测
![Image Name](https://cdn.kesci.com/upload/image/q5jcecxcba.png?imageView2/0/w/640/h/640)
具体结构：
![Image Name](https://cdn.kesci.com/upload/image/q5jccjhkii.png?imageView2/0/w/500/h/500)

Encoder
```
class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.num_hiddens=num_hiddens
        self.num_layers=num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size,num_hiddens, num_layers, dropout=dropout)
   
    def begin_state(self, batch_size, device):
        return [torch.zeros(size=(self.num_layers, batch_size, self.num_hiddens),  device=device),
                torch.zeros(size=(self.num_layers, batch_size, self.num_hiddens),  device=device)]
    def forward(self, X, *args):
        X = self.embedding(X) # X shape: (batch_size, seq_len, embed_size)
        X = X.transpose(0, 1)  # RNN needs first axes to be time
        # state = self.begin_state(X.shape[1], device=X.device)
        out, state = self.rnn(X)
        # The shape of out is (seq_len, batch_size, num_hiddens).
        # state contains the hidden state and the memory cell
        # of the last time step, the shape is (num_layers, batch_size, num_hiddens)
        return out, state
```
Decoder
```
class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size,num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens,vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).transpose(0, 1)
        out, state = self.rnn(X, state)
        # Make the batch to be the first dimension to simplify loss computation.
        out = self.dense(out).transpose(0, 1)
        return out, state
```

### 损失函数
解码器的输出是一个和词典维度相同的向量，其每个值对应与向量索引位置对应词的分数，一般是选择分数最大的那个词作为最终的输出。
在计算损失函数之前，要把padding去掉，因为padding的部分不参与计算
#### 序列屏蔽
序列有效长度保留，无效长度填充为特定value
```
def SequenceMask(X, X_len,value=0):
    maxlen = X.size(1)
    mask = torch.arange(maxlen)[None, :].to(X_len.device) < X_len[:, None]   
    X[~mask]=value
    return X
```
#### 损失函数MaskedSoftmaxCELoss
在交叉熵的基础上加上SequenceMask
```
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    # pred shape: (batch_size, seq_len, vocab_size)
    # label shape: (batch_size, seq_len)
    # valid_length shape: (batch_size, )
    def forward(self, pred, label, valid_length):
        # the sample weights shape should be (batch_size, seq_len)
        weights = torch.ones_like(label)
        weights = SequenceMask(weights, valid_length).float()
        self.reduction='none'
        output=super(MaskedSoftmaxCELoss, self).forward(pred.transpose(1,2), label)
        return (output*weights).mean(dim=1)
```

###  Beam Search搜索算法
greedy search:只考虑当前时刻的局部最优解，没有考虑前后语义是否连贯（非全局最优解)
维特比算法:选择整体分数最高的句子（搜索空间太大）
集束搜索：结合了greedy search和维特比算法
![Image Name](https://cdn.kesci.com/upload/image/q5jcia86z1.png?imageView2/0/w/640/h/640)


## 注意力机制
### seq2seq的不足
解码器在各个时间步依赖相同的背景变量（context vector）来获取输⼊序列信息。当编码器为循环神经⽹络时，背景变量来⾃它最终时间步的隐藏状态。将源序列输入信息以循环单位状态编码，然后将其传递给解码器以生成目标序列。然而这种结构存在着问题，尤其是RNN机制实际中存在长程梯度消失的问题，对于较长的句子，我们很难寄希望于将输入的序列转化为定长的向量而保存所有的有效信息，**所以随着所需翻译句子的长度的增加，这种结构的效果会显著下降**。
与此同时，解码的目标词语可能只与原输入的部分词语有关，而并不是与所有的输入有关。例如，当把“Hello world”翻译成“Bonjour le monde”时，“Hello”映射成“Bonjour”，“world”映射成“monde”。在seq2seq模型中，解码器只能隐式地从编码器的最终状态中选择相应的信息。然而，注意力机制可以将这种选择过程显式地建模。

### 注意力机制框架

Attention 是一种通用的带权池化方法，输入由两部分构成：询问（query）和键值对（key-value pairs）。$𝐤_𝑖∈ℝ^{𝑑_𝑘}, 𝐯_𝑖∈ℝ^{𝑑_𝑣}$. Query  $𝐪∈ℝ^{𝑑_𝑞}$ , attention layer得到输出与value的维度一致 $𝐨∈ℝ^{𝑑_𝑣}$. 对于一个query来说，attention layer 会与每一个key计算注意力分数并进行权重的归一化，输出的向量$o$则是value的加权求和，而每个key计算的权重与value一一对应。

为了计算输出，我们首先假设有一个函数$\alpha$ 用于计算query和key的相似性，然后可以计算所有的 attention scores $a_1, \ldots, a_n$ by
$$
a_i = \alpha(\mathbf q, \mathbf k_i).
$$
我们使用 softmax函数 获得注意力权重：
$$
b_1, \ldots, b_n = \textrm{softmax}(a_1, \ldots, a_n).
$$
最终的输出就是value的加权求和：
$$
\mathbf o = \sum_{i=1}^n b_i \mathbf v_i.
$$
![Image Name](https://cdn.kesci.com/upload/image/q5km4ooyu2.PNG?imageView2/0/w/960/h/960)

不同的attetion layer的区别在于score函数的选择，在本节的其余部分，我们将讨论两个常用的注意层 Dot-product Attention 和 Multilayer Perceptron Attention


### 点积注意力
The dot product 假设query和keys有相同的维度, 即 $\forall i, 𝐪,𝐤_𝑖 ∈ ℝ_𝑑 $. 通过计算query和key转置的乘积来计算attention score,通常还会除去 $\sqrt{d}$ 减少计算出来的score对维度𝑑的依赖性，如下
$$
𝛼(𝐪,𝐤)=⟨𝐪,𝐤⟩/ \sqrt{d} 
$$
假设 $ 𝐐∈ℝ^{𝑚×𝑑}$ 有 $m$ 个query，$𝐊∈ℝ^{𝑛×𝑑}$ 有 $n$ 个keys. 我们可以通过矩阵运算的方式计算所有 $mn$ 个score：
$$
𝛼(𝐐,𝐊)=𝐐𝐊^𝑇/\sqrt{d}
$$
 现在让我们实现这个层，它支持一批查询和键值对。此外，它支持作为正则化随机删除一些注意力权重.
 ```
 class DotProductAttention(nn.Module): 
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # query: (batch_size, #queries, d)
    # key: (batch_size, #kv_pairs, d)
    # value: (batch_size, #kv_pairs, dim_v)
    # valid_length: either (batch_size, ) or (batch_size, xx)
    def forward(self, query, key, value, valid_length=None):
        d = query.shape[-1]
        # set transpose_b=True to swap the last two dimensions of key
        
        scores = torch.bmm(query, key.transpose(1,2)) / math.sqrt(d)
        attention_weights = self.dropout(masked_softmax(scores, valid_length))
        print("attention_weight\n",attention_weights)
        return torch.bmm(attention_weights, value)
```

### 多层感知机注意力
在多层感知器中，我们首先将 query and keys 投影到  $ℝ^ℎ$ .为了更具体，我们将可以学习的参数做如下映射 
$𝐖_𝑘∈ℝ^{ℎ×𝑑_𝑘}$ ,  $𝐖_𝑞∈ℝ^{ℎ×𝑑_𝑞}$ , and  $𝐯∈ℝ^h$ . 将score函数定义
$$
𝛼(𝐤,𝐪)=𝐯^𝑇tanh(𝐖_𝑘𝐤+𝐖_𝑞𝐪)
$$
. 
然后将key 和 value 在特征的维度上合并（concatenate），然后送至 a single hidden layer perceptron 这层中 hidden layer 为  ℎ  and 输出的size为 1 .隐层激活函数为tanh，无偏置.
```
class MLPAttention(nn.Module):  
    def __init__(self, units,ipt_dim,dropout, **kwargs):
        super(MLPAttention, self).__init__(**kwargs)
        # Use flatten=True to keep query's and key's 3-D shapes.
        self.W_k = nn.Linear(ipt_dim, units, bias=False)
        self.W_q = nn.Linear(ipt_dim, units, bias=False)
        self.v = nn.Linear(units, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_length):
        query, key = self.W_k(query), self.W_q(key)
        #print("size",query.size(),key.size())
        # expand query to (batch_size, #querys, 1, units), and key to
        # (batch_size, 1, #kv_pairs, units). Then plus them with broadcast.
        features = query.unsqueeze(2) + key.unsqueeze(1)
        #print("features:",features.size())  #--------------开启
        scores = self.v(features).squeeze(-1) 
        attention_weights = self.dropout(masked_softmax(scores, valid_length))
        return torch.bmm(attention_weights, value)
```

### 引入注意力机制的Seq2seq模型
下图展示encoding 和decoding的模型结构，在时间步为t的时候。此刻attention layer保存着encodering看到的所有信息——即encoding的每一步输出。在decoding阶段，解码器的$t$时刻的隐藏状态被当作query，encoder的每个时间步的hidden states作为key和value进行attention聚合. Attetion model的输出当作成上下文信息context vector，并与解码器输入$D_t$拼接起来一起送到解码器：

![Image Name](https://cdn.kesci.com/upload/image/q5km7o8z93.PNG?imageView2/0/w/800/h/800)

$$
Fig1具有注意机制的seq-to-seq模型解码的第二步
$$
下图展示了seq2seq机制的所以层的关系，下面展示了encoder和decoder的layer结构

![Image Name](https://cdn.kesci.com/upload/image/q5km8dihlr.PNG?imageView2/0/w/800/h/800)
$$
Fig2具有注意机制的seq-to-seq模型中层结构
$$
#### 解码器
   由于带有注意机制的seq2seq的编码器不变，所以在此处我们只关注解码器。我们添加了一个MLP注意层(MLPAttention)，它的隐藏大小与解码器中的LSTM层相同。然后我们通过从编码器传递三个参数来初始化解码器的状态:
   
- the encoder outputs of all timesteps：encoder输出的各个状态，被用于attetion layer的memory部分，有相同的key和values

- the hidden state of the encoder’s final timestep：编码器最后一个时间步的隐藏状态，被用于初始化decoder 的hidden state

- the encoder valid length: 编码器的有效长度，借此，注意层不会考虑编码器输出中的填充标记（Paddings）

   在解码的每个时间步，我们使用解码器的最后一个RNN层的输出作为注意层的query。然后，将注意力模型的输出与输入嵌入向量连接起来，输入到RNN层。虽然RNN层隐藏状态也包含来自解码器的历史信息，但是attention model的输出显式地选择了enc_valid_len以内的编码器输出，这样attention机制就会尽可能排除其他不相关的信息。
```
class Seq2SeqAttentionDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention_cell = MLPAttention(num_hiddens,num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size+ num_hiddens,num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens,vocab_size)

    def init_state(self, enc_outputs, enc_valid_len, *args):
        outputs, hidden_state = enc_outputs
#         print("first:",outputs.size(),hidden_state[0].size(),hidden_state[1].size())
        # Transpose outputs to (batch_size, seq_len, hidden_size)
        return (outputs.permute(1,0,-1), hidden_state, enc_valid_len)
        #outputs.swapaxes(0, 1)
        
    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_len = state
        #("X.size",X.size())
        X = self.embedding(X).transpose(0,1)
#         print("Xembeding.size2",X.size())
        outputs = []
        for l, x in enumerate(X):
#             print(f"\n{l}-th token")
#             print("x.first.size()",x.size())
            # query shape: (batch_size, 1, hidden_size)
            # select hidden state of the last rnn layer as query
            query = hidden_state[0][-1].unsqueeze(1) # np.expand_dims(hidden_state[0][-1], axis=1)
            # context has same shape as query
#             print("query enc_outputs, enc_outputs:\n",query.size(), enc_outputs.size(), enc_outputs.size())
            context = self.attention_cell(query, enc_outputs, enc_outputs, enc_valid_len)
            # Concatenate on the feature dimension
#             print("context.size:",context.size())
            x = torch.cat((context, x.unsqueeze(1)), dim=-1)
            # Reshape x to (1, batch_size, embed_size+hidden_size)
#             print("rnn",x.size(), len(hidden_state))
            out, hidden_state = self.rnn(x.transpose(0,1), hidden_state)
            outputs.append(out)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.transpose(0, 1), [enc_outputs, hidden_state,
                                        enc_valid_len]
```
我们得到了相同的解码器输出形状，但是状态结构改变了。

## 卷积神经网络基础
### 二维互相关运算
二维互相关（cross-correlation）运算的输入是一个二维输入数组和一个二维核（kernel）数组，输出也是一个二维数组，其中核数组通常称为卷积核或过滤器（filter）。卷积核的尺寸通常小于输入数组，卷积核在输入数组上滑动，在每个位置上，卷积核与该位置处的输入子数组按元素相乘并求和，得到输出数组中相应位置的元素。图1展示了一个互相关运算的例子，阴影部分分别是输入的第一个计算区域、核数组以及对应的输出。

![Image Name](https://cdn.kesci.com/upload/image/q5nfdbhcw5.png?imageView2/0/w/640/h/640)
$$
\begin{array}{l}{0 \times 0+1 \times 1+3 \times 2+4 \times 3=19} \\ {1 \times 0+2 \times 1+4 \times 2+5 \times 3=25} \\ {3 \times 0+4 \times 1+6 \times 2+7 \times 3=37} \\ {4 \times 0+5 \times 1+7 \times 2+8 \times 3=43}\end{array}
$$

```
def corr2d(X, K):
    H, W = X.shape
    h, w = K.shape
    Y = torch.zeros(H - h + 1, W - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y
```
### 二维卷积层
二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏置来得到输出。卷积层的模型参数包括卷积核和标量偏置。
```
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```
### 互相关运算与卷积运算

卷积层得名于卷积运算，但卷积层中用到的并非卷积运算而是互相关运算。我们将核数组上下翻转、左右翻转，再与输入数组做互相关运算，这一过程就是卷积运算。由于卷积层的核数组是可学习的，所以使用互相关运算与使用卷积运算并无本质区别。

### 特征图与感受野

二维卷积层输出的二维数组可以看作是输入在空间维度（宽和高）上某一级的表征，也叫特征图（feature map）。影响元素$x$的前向计算的所有可能输入区域（可能大于输入的实际尺寸）叫做$x$的感受野（receptive field）。

以二维互相关运算为例，输入中阴影部分的四个元素是输出中阴影部分元素的感受野。我们将图中形状为$2 \times 2$的输出记为$Y$，将$Y$与另一个形状为$2 \times 2$的核数组做互相关运算，输出单个元素$z$。那么，$z$在$Y$上的感受野包括$Y$的全部四个元素，在输入上的感受野包括其中全部9个元素。可见，我们可以通过更深的卷积神经网络使特征图中单个元素的感受野变得更加广阔，从而捕捉输入上更大尺寸的特征。

### 填充和步幅
#### 填充

填充（padding）是指在输入高和宽的两侧填充元素（通常是0元素），图2里我们在原输入高和宽的两侧分别添加了值为0的元素。


![Image Name](https://cdn.kesci.com/upload/image/q5nfl6ejy4.png?imageView2/0/w/640/h/640)


如果原输入的高和宽是$n_h$和$n_w$，卷积核的高和宽是$k_h$和$k_w$，在高的两侧一共填充$p_h$行，在宽的两侧一共填充$p_w$列，则输出形状为：
$$
(n_h+p_h-k_h+1)\times(n_w+p_w-k_w+1)
$$
我们在卷积神经网络中使用奇数高宽的核，比如$3 \times 3$，$5 \times 5$的卷积核，对于高度（或宽度）为大小为$2 k + 

#### 步幅

在互相关运算中，卷积核在输入数组上滑动，每次滑动的行数与列数即是步幅（stride）。此前我们使用的步幅都是1，下图展示了在高上步幅为3、在宽上步幅为2的二维互相关运算。


![Image Name](https://cdn.kesci.com/upload/image/q5nflohnqg.png?imageView2/0/w/640/h/640)


一般来说，当高上步幅为$s_h$，宽上步幅为$s_w$时，输出形状为：

$$

\lfloor(n_h+p_h-k_h+s_h)/s_h\rfloor \times \lfloor(n_w+p_w-k_w+s_w)/s_w\rfloor

$$


如果$p_h=k_h-1$，$p_w=k_w-1$，那么输出形状将简化为$\lfloor(n_h+s_h-1)/s_h\rfloor \times \lfloor(n_w+s_w-1)/s_w\rfloor$。更进一步，如果输入的高和宽能分别被高和宽上的步幅整除，那么输出形状将是$(n_h / s_h) \times (n_w/s_w)$。

当$p_h = p_w = p$时，我们称填充为$p$；当$s_h = s_w = s$时，我们称步幅为$s$。1$的核，令步幅为1，在高（或宽）两侧选择大小为$k$的填充，便可保持输入与输出尺寸相同。


### 多输入通道和多输出通道

之前的输入和输出都是二维数组，但真实数据的维度经常更高。例如，彩色图像在高和宽2个维度外还有RGB（红、绿、蓝）3个颜色通道。假设彩色图像的高和宽分别是$h$和$w$（像素），那么它可以表示为一个$3 \times h \times w$的多维数组，我们将大小为3的这一维称为通道（channel）维。

#### 多输入通道

卷积层的输入可以包含多个通道，下图展示了一个含2个输入通道的二维互相关计算的例子。


![Image Name](https://cdn.kesci.com/upload/image/q5nfmdnwbq.png?imageView2/0/w/640/h/640)

假设输入数据的通道数为$c_i$，卷积核形状为$k_h\times k_w$，我们为每个输入通道各分配一个形状为$k_h\times k_w$的核数组，将$c_i$个互相关运算的二维输出按通道相加，得到一个二维数组作为输出。我们把$c_i$个核数组在通道维上连结，即得到一个形状为$c_i\times k_h\times k_w$的卷积核。

#### 多输出通道

卷积层的输出也可以包含多个通道，设卷积核输入通道数和输出通道数分别为$c_i$和$c_o$，高和宽分别为$k_h$和$k_w$。如果希望得到含多个通道的输出，我们可以为每个输出通道分别创建形状为$c_i\times k_h\times k_w$的核数组，将它们在输出通道维上连结，卷积核的形状即$c_o\times c_i\times k_h\times k_w$。

对于输出通道的卷积核，我们提供这样一种理解，一个$c_i \times k_h \times k_w$的核数组可以提取某种局部特征，但是输入可能具有相当丰富的特征，我们需要有多个这样的$c_i \times k_h \times k_w$的核数组，不同的核数组提取的是不同的特征。

### 卷积层与全连接层的对比
二维卷积层经常用于处理图像，与此前的全连接层相比，它主要有两个优势：

一是全连接层把图像展平成一个向量，在输入图像上相邻的元素可能因为展平操作不再相邻，网络难以捕捉局部信息。而卷积层的设计，天然地具有提取局部信息的能力。

二是卷积层的参数量更少。不考虑偏置的情况下，一个形状为$(c_i, c_o, h, w)$的卷积核的参数量是$c_i \times c_o \times h \times w$，与输入图像的宽高无关。假如一个卷积层的输入和输出形状分别是$(c_1, h_1, w_1)$和$(c_2, h_2, w_2)$，如果要用全连接层进行连接，参数数量就是$c_1 \times c_2 \times h_1 \times w_1 \times h_2 \times w_2$。使用卷积层可以以较少的参数数量来处理更大的图像。

### 池化

#### 二维池化层

池化层主要用于缓解卷积层对位置的过度敏感性。同卷积层一样，池化层每次对输入数据的一个固定形状窗口（又称池化窗口）中的元素计算输出，池化层直接计算池化窗口内元素的最大值或者平均值，该运算也分别叫做**最大池化或平均池化**。下图展示了池化窗口形状为$2\times 2$的最大池化。


![Image Name](https://cdn.kesci.com/upload/image/q5nfob3odo.png?imageView2/0/w/640/h/640)
$$
\begin{array}{l}{\max (0,1,3,4)=4} \\ {\max (1,2,4,5)=5} \\ {\max (3,4,6,7)=7} \\ {\max (4,5,7,8)=8}\end{array}
$$
二维平均池化的工作原理与二维最大池化类似，但将最大运算符替换成平均运算符。池化窗口形状为$p \times q$的池化层称为$p \times q$池化层，其中的池化运算叫作$p \times q$池化。

池化层也可以在输入的高和宽两侧填充并调整窗口的移动步幅来改变输出形状。池化层填充和步幅与卷积层填充和步幅的工作机制一样。

在处理多通道输入数据时，池化层对每个输入通道分别池化，但不会像卷积层那样将各通道的结果按通道相加。这意味着池化层的输出通道数与输入通道数相等。
```
X = torch.arange(32, dtype=torch.float32).view(1, 2, 4, 4)
pool2d = nn.MaxPool2d(kernel_size=3, padding=1, stride=(2, 1))
Y = pool2d(X)
```
平均池化层使用的是`nn.AvgPool2d`，使用方法与`nn.MaxPool2d`相同。

## LeNet 模型
LeNet分为卷积层块和全连接层块两个部分

卷积层块里的基本单位是卷积层后接平均池化层：卷积层用来识别图像里的空间模式，如线条和物体局部，之后的平均池化层则用来降低卷积层对位置的敏感性。

卷积层块由两个这样的基本单位重复堆叠构成。在卷积层块中，每个卷积层都使用$5 \times 5$的窗口，并在输出上使用sigmoid激活函数。第一个卷积层输出通道数为6，第二个卷积层输出通道数则增加到16。

全连接层块含3个全连接层。它们的输出个数分别是120、84和10，其中10为输出的类别个数。

```
#net
class Flatten(torch.nn.Module):  #展平操作
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Reshape(torch.nn.Module): #将图像大小重定型
    def forward(self, x):
        return x.view(-1,1,28,28)      #(B x C x H x W)
    
net = torch.nn.Sequential(     #Lelet                                                  
    Reshape(),
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), #b*1*28*28  =>b*6*28*28
    nn.Sigmoid(),                                                       
    nn.AvgPool2d(kernel_size=2, stride=2),                              #b*6*28*28  =>b*6*14*14
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),           #b*6*14*14  =>b*16*10*10
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),                              #b*16*10*10  => b*16*5*5
    Flatten(),                                                          #b*16*5*5   => b*400
    nn.Linear(in_features=16*5*5, out_features=120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)
#print
X = torch.randn(size=(1,1,28,28), dtype = torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
'''
Reshape output shape: 	 torch.Size([1, 1, 28, 28])
Conv2d output shape: 	 torch.Size([1, 6, 28, 28])
Sigmoid output shape: 	 torch.Size([1, 6, 28, 28])
AvgPool2d output shape: 	 torch.Size([1, 6, 14, 14])
Conv2d output shape: 	 torch.Size([1, 16, 10, 10])
Sigmoid output shape: 	 torch.Size([1, 16, 10, 10])
AvgPool2d output shape: 	 torch.Size([1, 16, 5, 5])
Flatten output shape: 	 torch.Size([1, 400])
Linear output shape: 	 torch.Size([1, 120])
Sigmoid output shape: 	 torch.Size([1, 120])
Linear output shape: 	 torch.Size([1, 84])
Sigmoid output shape: 	 torch.Size([1, 84])
Linear output shape: 	 torch.Size([1, 10])
'''
```

卷积层由于使用高和宽均为5的卷积核，从而将高和宽分别减小4，而池化层则将高和宽减半，但通道数则从1增加到16。全连接层则逐层减少输出个数，直到变成图像的类别数10。
![Image Name](https://cdn.kesci.com/upload/image/q5ndxi6jl5.png?imageView2/0/w/640/h/640)



卷积神经网络就是含卷积层的网络。
LeNet交替使用卷积层和最大池化层后接全连接层来进行图像分类。

## 深度卷积神经网络（AlexNet）

LeNet:  在大的真实数据集上的表现并不尽如⼈意。     
1.神经网络计算复杂。  
2.还没有⼤量深⼊研究参数初始化和⾮凸优化算法等诸多领域。  
  
机器学习的特征提取:手工定义的特征提取函数  
神经网络的特征提取：通过学习得到数据的多级表征，并逐级表⽰越来越抽象的概念或模式。  
  
神经网络发展的限制:数据、硬件

### AlexNet
首次证明了学习到的特征可以超越⼿⼯设计的特征，从而⼀举打破计算机视觉研究的前状。
**特征：**
1. 8层变换，其中有5层卷积和2层全连接隐藏层，以及1个全连接输出层。
2. 将sigmoid激活函数改成了更加简单的ReLU激活函数。
3. 用Dropout来控制全连接层的模型复杂度。
4. 引入数据增强，如翻转、裁剪和颜色变化，从而进一步扩大数据集来缓解过拟合。

![Image Name](https://cdn.kesci.com/upload/image/q5kv4gpx88.png?imageView2/0/w/640/h/640)

```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
         # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            #由于使用CPU镜像，精简网络，若为GPU镜像可添加该层
            #nn.Linear(4096, 4096),
            #nn.ReLU(),
            #nn.Dropout(0.5),

            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
        )

    def forward(self, img):

        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
```

### VGG使用重复元素的网络
VGG：通过重复使⽤简单的基础块来构建深度模型。  
Block:数个相同的填充为1、窗口形状为$3\times 3$的卷积层,接上一个步幅为2、窗口形状为$2\times 2$的最大池化层。  
卷积层保持输入的高和宽不变，而池化层则对其减半。

![Image Name](https://cdn.kesci.com/upload/image/q5l6vut7h1.png?imageView2/0/w/640/h/640)

```
def vgg_block(num_convs, in_channels, out_channels): #卷积层个数，输入通道数，输出通道数
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 这里会使宽高减半
    return nn.Sequential(*blk)

def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一个vgg_block都会使宽高减半
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    # 全连接层部分
    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(),
                                 nn.Linear(fc_features, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, 10)
                                ))
    return net
		
```
###  ⽹络中的⽹络（NiN） 
LeNet、AlexNet和VGG：先以由卷积层构成的模块充分抽取 空间特征，再以由全连接层构成的模块来输出分类结果。  
NiN：串联多个由卷积层和“全连接”层构成的小⽹络来构建⼀个深层⽹络。  
⽤了输出通道数等于标签类别数的NiN块，然后使⽤全局平均池化层对每个通道中所有元素求平均并直接⽤于分类。  

![Image Name](https://cdn.kesci.com/upload/image/q5l6u1p5vy.png?imageView2/0/w/960/h/960)

1×1卷积核作用   
1.放缩通道数：通过控制卷积核的数量达到通道数的放缩。  
2.增加非线性。1×1卷积核的卷积过程相当于全连接层的计算过程，并且还加入了非线性激活函数，从而可以增加网络的非线性。  
3.计算参数少

NiN重复使⽤由卷积层和代替全连接层的1×1卷积层构成的NiN块来构建深层⽹络。  
NiN去除了容易造成过拟合的全连接输出层，而是将其替换成输出通道数等于标签类别数 的NiN块和全局平均池化层。   
NiN的以上设计思想影响了后⾯⼀系列卷积神经⽹络的设计。  


```
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU())
    return blk
class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2), 
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    GlobalAvgPool2d(), 
    # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
    d2l.FlattenLayer())
X = torch.rand(1, 1, 224, 224)
for name, blk in net.named_children(): 
    X = blk(X)
    print(name, 'output shape: ', X.shape)
'''
0 output shape:  torch.Size([1, 96, 54, 54])
1 output shape:  torch.Size([1, 96, 26, 26])
2 output shape:  torch.Size([1, 256, 26, 26])
3 output shape:  torch.Size([1, 256, 12, 12])
4 output shape:  torch.Size([1, 384, 12, 12])
5 output shape:  torch.Size([1, 384, 5, 5])
6 output shape:  torch.Size([1, 384, 5, 5])
7 output shape:  torch.Size([1, 10, 5, 5])
8 output shape:  torch.Size([1, 10, 1, 1])
9 output shape:  torch.Size([1, 10])
'''
```


### GoogLeNet :Networks with Parallel Concatenations
1. 由Inception基础块组成。  
2. Inception块相当于⼀个有4条线路的⼦⽹络。它通过不同窗口形状的卷积层和最⼤池化层来并⾏抽取信息，并使⽤1×1卷积层减少通道数从而降低模型复杂度。   
3. 可以⾃定义的超参数是每个层的输出通道数，我们以此来控制模型复杂度。 



![Image Name](https://cdn.kesci.com/upload/image/q5weakogtb.png?imageView2/0/w/640/h/640)



完整模型结构  

![Image Name](https://cdn.kesci.com/upload/image/q5l6x0fyyn.png?imageView2/0/w/640/h/640)


```
class Inception(nn.Module):
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出
```


```python
参考：
https://zhuanlan.zhihu.com/p/32481747
https://zhuanlan.zhihu.com/p/32085405
```


```python

```
