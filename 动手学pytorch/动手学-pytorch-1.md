---
title: 动手学 pytorch-实现线性回归、softmax、感知机、RNN
date: 2020-02-13 23:43:39
catalog: true
doc :true
tags:
---

## tensor 基础


### 创建tensor

#### 创建特定tensor：
  - **`torch.from_numpy(a)`**
    numpy向量转tensor

  - **`torch.tensor([2,2])`**
  - **`torch.FloatTensor([2,2.])`**
  - **`torch.FloatTensor([[1,2],[3,4]])`**
	列表转tensor

  - **`torch.empty(size)`**
    返回形状为size的空tensor
    
  - **`torch.zeros(size)`**
    全部是0的tensor
    
  - **`torch.zeros_like(input)`**
    返回跟input的tensor一个size的全零tensor
    
  - **`torch.ones(size)`**
    全部是1的tensor
    
  - **`torch.ones_like(input)`**
    返回跟input的tensor一个size的全一tensor
    
  - **`torch.arange(start=0, end, step=1)`**
    返回一个从start到end的序列，可以只输入一个end参数，就跟python的range()一样了。实际上PyTorch也有range()，但是这个要被废掉了，替换成arange了
    
  - **`torch.full(size, fill_value)`**
    这个有时候比较方便，把fill_value这个数字变成size形状的张量
    
#### 随机采样生成：

  - **`torch.round(size)`**
     在[0,1]内的均匀分布的随机数
  - **`torch.rand_like(input)`**
    返回跟input的tensor一样size的0-1随机数
  - **`torch.randn(size)`**
    返回标准正太分布N(0,1)的随机数
  - **`torch.normal(mean, std, out=None)`**
    正态分布。这里注意，mean和std都是tensor，返回的形状由mean和std的形状决定，一般要求两者形状一样。如果，mean缺失，则默认为均值0，如果std缺失，则默认标准差为1.
  
	
### torch 基本操作  

#### 索引：

  - **`index_select`**
	
	```
	x=torch.rand(4,3,28,28) ###4张图片
	x.index_select(0,torch.tensor([0,1,2]))#第一个参数为轴，第二个参数为tensor类型的索引
	x.index_select(0,torch.arange(3))#效果同上句

	```
  - **`torch.masked_select`**
  
	```
	x=torch.rand(3,4)
	mask=x.ge(0.5)#会把x中大于0.5的置为一，其他置为0，类似于阈值化操作。
	y=torch.masked_select(x,mask)#将mask中值为1的元素取出来，比如mask有3个位置值为1
	```
  - **`torch.take`**
  
	```
	x=torch.tensor([[1,2,3],[4,5,6]])
	torch.take(x,torch.tensor([0,2,6]))
	#则最后结果为tensor([1,3,6]),也就是说会先将tensor压缩成一维向量，再按照索引取元素。
	```
#### 切片:(start : end : step)  
	```
	img=torch.rand(4,3,28,28)#4张图片
	
	img[1]#获取第二张图片
	img[0,0].shape#获取第一张图片的第一个通道的图片形状
	img[0,0,2,4]#返回像素灰度值标量
	
	img[:2]#获得img[0]和img[1]
	#img[:2,:1]==img[:2,:1,:,:]
	img[2:]#获得img[2],img[3],img[4]三张图片
	img[-1:]#获得img[4]
	img[:,:,::2,::2]#对图片进行隔行（列）采样
	
	#还有一种索引中的...操作，有自动填充的功能,一般用于维数很多时使用。
	img[0,...]
	#img.shape的结果是torch.Size([4,28,28]),这是和img[0,:]或者img[0]是一样的。
	img[0,...,0:28:2]
	#此时由于写了最右边的索引，中间的...等价于:,:，即img[0,:,:,0:28:2]
	```
#### 维度变换（view/reshape/squeeze/transpose/expand/permute）:

  - **`view()和reshape效果一样,都是改变shape`**
	
	```
	a = torch.rand(4, 1, 28, 28)
	print(a.shape)
	print(a.reshape(4 * 1, 28, 28).shape)
	print(a.reshape(4, 1 * 28 * 28).shape)
	'''
	torch.Size([4, 1, 28, 28])
	torch.Size([4, 28, 28])
	torch.Size([4, 784])
	'''
	```
  - **`增加维度`**
	正的索引是在那个维度原本的位置前面插入这个新增加的维度，负的索引是在那个位置之后插入
    
	```
	print(a.shape)
	print(a.unsqueeze(0).shape)  # 在0号维度位置插入一个维度
	print(a.unsqueeze(-1).shape)  # 在最后插入一个维度
	print(a.unsqueeze(3).shape)  # 在3号维度位置插入一个维度
	'''
	torch.Size([4, 1, 28, 28])
	torch.Size([1, 4, 1, 28, 28])
	torch.Size([4, 1, 28, 28, 1])
	torch.Size([4, 1, 28, 1, 28])
	'''
	```
  - **`删减维度`**
	删减维度实际上是一个压榨的过程，直观地看是把那些多余的[]给去掉，也就是只是去删除那些size=1的维度
    ```
	a = torch.Tensor(1, 4, 1, 9)
	print(a.shape)
	print(a.squeeze().shape) # 能删除的都删除掉
	print(a.squeeze(0).shape) # 尝试删除0号维度,ok
	print(a.squeeze(2).shape) # 尝试删除2号维度,ok
	print(a.squeeze(3).shape) # 尝试删除3号维度,3号维度是9不是1,删除失败
	'''
	torch.Size([1, 4, 1, 9])
	torch.Size([4, 9])
	torch.Size([4, 1, 9])
	torch.Size([1, 4, 9])
	torch.Size([1, 4, 1, 9])
	'''
	```
  - **`维度扩展(expand)`**  
	expand就是在某个size=1的维度上改变size，改成更大的一个大小，实际就是在每个size=1的维度上的标量的广播操作
    ```
	b = torch.rand(32)
	f = torch.rand(4, 32, 14, 14)

	# 想要把b加到f上面去

	# 先进行维度增加
	b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
	print(b.shape)

	# 再进行维度扩展
	b = b.expand(4, -1, 14, 14)  # -1表示这个维度保持不变,这里写32也可以
	print(b.shape)
	
	'''
	torch.Size([1, 32, 1, 1])
	torch.Size([4, 32, 14, 14])
	'''

	```
  - **`维度重复(repeat)`**  
	repeat就是将每个位置的维度都重复至指定的次数，以形成新的Tensor。repeat会重新申请内存空间
	```
	b = torch.rand(1,32,1,1)
	print(b.shape)

	# 维度重复,32这里不想进行重复,所以就相当于"重复至1次"
	b = b.repeat(4, 1, 14, 14)
	print(b.shape)
	'''
	torch.Size([1, 32, 1, 1])
	torch.Size([4, 32, 14, 14])
	'''
	```
   - **`转置`**     
	只适用于dim=2的Tensor
	```
	c = torch.Tensor(2, 4)
	print(c.t().shape)
	'''
	torch.Size([4, 2])
	'''
	```
   - **`维度交换transpose`** 
	注意这种交换使得存储不再连续，再执行一些reshape的操作肯定是执行不了的，所以要调用一下contiguous()使其变成连续的维度
	```
	d = torch.Tensor(6, 3, 1, 2)
	print(d.transpose(1, 3).contiguous().shape)  # 1号维度和3号维度交换
	'''
	torch.Size([6, 2, 1, 3])
	'''
	```
	下面这个例子比较一下每个位置上的元素都是一致的，来验证一下这个交换->压缩shape->展开shape->交换回去是没有问题的
	```
	e = torch.rand(4, 3, 6, 7)
	e2 = e.transpose(1, 3).contiguous().reshape(4, 7 * 6 * 3).reshape(4, 7, 6, 3).transpose(1, 3)
	print(e2.shape)
	# 比较下两个Tensor所有位置上的元素是否都相等
	print(torch.all(torch.eq(e, e2)))
	'''
	torch.Size([4, 3, 6, 7])
	tensor(1, dtype=torch.uint8)
	'''
	```
   - **`permute`** 
	如果四个维度表示上节的[batch,channel,h,w][batch,channel,h,w]      [batch,channel,h,w][batch,channel,h,w]，如果想把channelchannel      channelchannel放到最后去，形成[batch,h,w,channel][batch,h,w,channel]      [batch,h,w,channel][batch,h,w,channel]，那么如果使用前面的维度交换，至少要交换两次（先13交换再12交换）。而使用permute可以直接指定维度新的所处位置，方便很多
	```
	h = torch.rand(4, 3, 6, 7)
	print(h.permute(0, 2, 3, 1).shape)
	'''
	torch.Size([4, 6, 7, 3])
	'''
	```
#### tensor的拼接与拆分
   (cat/stack/spilit/chunk) 
   ```
   #cat拼接
	a=torch.rand(4,3,18,18)
	b=torch.rand(5,3,18,18)
	c=torch.rand(4,1,18,18)
	d=a.copy()

	torch.cat([a,b],dim=0)#拼接得到(9,3,18,18)的数据
	#若为2维数据，dim=0则是竖向拼接，dim=0就是横向拼接。dim所指维度可以不同，但其他维度形状必须一致
	torch.cat([a,c],dim=1)#就会得到(4,4,18,18)的数据。
	#stack增维度拼接
	torch.stack([a,b],dim=0)#得到形状为(2,4,3,18,18)。使用时列表内对象的形状需要一致。
	#split拆分

	#根据欲拆分长度：
	a1,a2=a.split(2,dim=0)#拆分长度为2.对第0维按照2个一份进行拆分。拆分获得两个形状为(2,3,18,18)的张量。
	a1,a2=a.split([3,1],dim=0)#不同长度拆分。获得(3,3,18,18)和(1,3,18,18)两个形状的张量。
	#chunk拆分

	#根据欲拆分数量
	a1,a2,a3,a4=a.chunk(4,dim=0)#将张量依第0维拆分成4个(1,3,18,18)的张量。等效于a.split(1,dim=0)
   ```
#### tensor的运算
	```
	# +等价于torch.add()
	
	#乘法
	torch.mm(a, b) ##mm只能进行矩阵乘法,不可以是tensor,也就是输入的两个tensor维度只能是 (n×m) 和 (m×p)
	torch.mul(a, b) ##是矩阵a和b对应位相乘，a和b的维度必须相等，比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是(1, 2)的矩阵
	torch.matmul(a,b) ##可以进行张量乘法, 输入可以是高维
	torch.bmm(a,b) ## 是两个三维张量相乘, 两个输入tensor维度是 (b×n×m) 和 (b×m×p) , 第一维b代表batch size，输出为 (b×n×p)
	
	#平方
	a=torch.full([2,2],2)#创建一个(2,2)的全2矩阵
	a.pow(2)#a的每个元素都平方
	a**2#等价于上一句
	
	#开方
	a.sqrt()#平方根
	a**0.5#等价于上一句
	
	#exp,log
	a.torch.exp(torch.ones(2,2))
	torch.log(a)
	
	#近似
	#floor()向下取整，ceil()向上取整，round()四舍五入。
	
	#取整取小数
	#trunc()取整，frac()取小数
	
	#clamp取范围
	a=torch.tensor([[3,5],[6,8]])
	a.clamp(6)#得到[[6,6],[6,8]],小于6的都变为6
	a.clamp(5,6)#得到[[5,5],[6,6]],小于下限变为下限，大于上限变为上限。
	```
#### tensor的统计属性
	```
	#范数
	#求多少p范数只需要在norm(p)的参数中修改p即可
	a.norm(1)#求a的一范数，范数也可以加dim=
	
	#求最大值和最小值与其相关的索引
	a.min()
	a.max()
	a.argmax()#会得到索引值，返回的永远是一个标量，多维张量会先拉成向量再求得其索引。拉伸的过程为每一行加起来变成一整行，而不是matlab中的列拉成一整列。
	a.argmin()
	a.argmax(dim=1)#如果不想获取拉伸后的索引值就需要在指定维度上进行argmax，比如如果a为(2，2)的矩阵，那么这句话的结果就可能是[1,1],表示第一行第一个在此行最大，第二行第一个在此行最大。
	
	#累加总和
	a.sum()
	
	#累乘综合
	a.prod()
	
	#dim,keepdim
	#假设a的形状为(4,10)
	a.max(dim=1)#结果会得到一个(4)的张量，表示4个样本中每个样本10个特征的最大值组成的张量。(max换成argmax也是同理)。
	a.max(dim=1,keepdim=True)#同时返回a.argmax(dim=1)得到的结果，以保持维度数目和原来一致。
	
	#top-k,k-th
	a.topk(5)#返回张量a前5个最大值组成的向量
	a.topk(5,dim=1,largest=False)#关闭largest求最小的5个
	a.kthvalue(8,dim=1)#返回第八小的值
	
	#比较操作
	#都是进行element-wise操作
	torch.eq(a,b)#返回的是张量
	torch.equal(a,b)#返回的是True或者False
	```
#### Tensor的高阶操作
  - **`where`** 
	用C=torch.where(condition,A,B)其中A,B,C,condition是shape相同的Tensor，C中的某些元素来自A，某些元素来自B，这由condition中相应位置的元素是1还是0来决定
	```
	cond = torch.tensor([[0.6, 0.1], [0.2, 0.7]])
	a = torch.tensor([[1, 2], [3, 4]])
	b = torch.tensor([[4, 5], [6, 7]])
	c = torch.where(cond > 0.5, a, b) ##条件满足取a,条件不满足取b
	print(c) 
	'''
	tensor([[1, 5],
        [6, 4]])
	'''
	```
  - **`gather`**
	torch.gather(input, dim, index, out=None)
	沿给定轴dim，将输入索引张量index指定位置的值进行聚合
	 对一个3维张量，输出可以定义为：
                **`out[i][j][k] = tensor[index[i][j][k]][j][k] # dim=0`**
                **` out[i][j][k] = tensor[i][index[i][j][k]][k] # dim=1`**
                **`out[i][j][k] = tensor[i][j][index[i][j][k]] # dim=2`**
	dim=1,按行操作，那么是列索引
	dim=2,按列操作，那么是行索引
	
	```
	a = torch.Tensor([[1,2],[3,4]])
	b = torch.gather(a,1,torch.LongTensor([[0,0],[1,0]]))  
	'''
	tensor([[1., 1.],  
			[4., 3.]])
	[1,2]取[0,0]->[1,1],[3,4]取[1,0]即[4,3]
	'''
	b = torch.gather(a,2,torch.LongTensor([[1,1],[1,0]]))
	'''
	[1,1]取[3,4],[1,0]取[3,2]
	tensor([[3., 4.],
        [3., 2.]])
	'''

	```
#### pytorch自动求导autograd:

	autograd包是PyTorch中所有神经网络的核心。它为Tensors上的所有操作提供自动微分。它是一个自定义的框架，这意味着以代码运行方式定义后向传播，并且每次迭代都可以不同。

  - **`Tensor类`**
	torch.Tensor是包的核心类。如果将其属性.requires_grad设置为Ture，则会开始跟踪针对tensor的所有操作。完成计算后，可以调用.backward()来自动计算所有梯度。该张量的梯度将累积到.grad属性中。
	.detach()：停止tensor历史记录的跟踪，它将其与计算历史记录分离，并防止将来的计算被跟踪。
	要停止跟踪历史记录（和使用内存），我们还可以将代码块使用with torch.no_grad():包装起来，这在评估模型时特别有用，因为模型在训练阶段具有requires_grad = True的可训练参数有利于调参，但在评估阶段我们不再需要梯度。
  
  - **`function类`**
    Function类也是autograd一个非常重要的类，Tensor 和 Function 互相连接并构建一个非循环图，它保存整个完整的计算过程的历史信息。每个张量都有一个 .grad_fn 属性保存着创建了张量的 Function 的引用，（如果用户自己创建张量，则grad_fn 是 None ）。
    如果你想计算导数，你可以调用 Tensor.backward()。如果 Tensor 是标量（即它包含一个元素数据），则不需要指定任何参数backward()，但是如果它有更多元素，则需要指定一个gradient 参数来指定张量的形状。


#### pytorch流程

```
1: 准备数据(注意数据格式不同）
2: 定义网络结构model
3: 定义损失函数
4: 定义优化算法 optimizer
5: 训练-pytorch
迭代训练：
	5.1:准备好tensor形式的输入数据和标签(可选)
	5.2:前向传播计算网络输出output和计算损失函数loss
	5.3:反向传播更新参数
		5.3.1:将上次迭代计算的梯度值清0
			optimizer.zero_grad()
		5.3.2:反向传播，计算梯度值
			loss.backward()
		5.3.3:更新权值参数
			optimizer.step()
6: 在测试集上测试-pytorch
    遍历测试集，自定义metric
7: 保存网络（可选） 具体实现参考上面代码
```

## 线性回归
```
损失函数：均方误差
优化函数：（小批量）随机梯度下降
```
  - **`从零开始的实现核心代码`**
	- **`1.数据读取生成器`**
	```
	def data_iter(batch_size,features,labels):
		num_examples = len(features)
		indices = list(range(num_examples))
		random.shuffle(indices)
		for i in range(0,num_examples,batch_size):
			j = torch.LongTensor(indices[i:min(i+batch_size,num_examples)])
			yield features.index_select(0,j),labels.index_select(0,j)
	```
	- **`2.模型初始化参数`**
	```
	w = torch.tensor(np.random.normal(0,0.01,(num_inputs,1)),dtype=torch.float32)
	b = torch.zeros(1,dtype=torch.float32)

	w.requires_grad_(requires_grad =True)
	b.requires_grad_(requires_grad =True)
	```
	- **`3.定义模型和损失函数、优化函数`**
	```
	def linreg(X,w,b):
		return torch.mm(X,w)+b
		
	def squared_loss(y_hat,y):
		return (y_hat-y.view(-1,1))**2/2
		
	def sgd(params,lr,batch_size):
		for param in params:
			param.data -=lr*param.grad / batch_size
	```
	- **`4.模型训练`**
	```
	for epoch in range(num_epochs):
		for X,y in data_iter(batch_size,features,labels):
			l = loss(net(X,w,b),y).sum()
			l.backward() ##反向传播
			
			sgd([w,b],lr,batch_size) ##参数优化
			w.grad.data.zero_()
			b.grad.data.zero_()
		train_l =loss(net(features,w,b),labels)
		print('epoch %d ,loss %f'%(epoch+1,train_l.mean()))	
	```
  - **`pytorch实现`**
	- **`1.数据读取`**
	```
	dataset = Data.TensorDataset(features,labels)

	data_iter = Data.DataLoader(
		dataset = dataset,
		batch_size = batch_size,
		shuffle =True,
		num_workers =2,
		)
	```
	- **`2.定义模型`**
	```
	##多层网络生成法
	net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # other layers can be added here
    )
	```
	- **`3.模型参数初始化`**
	```
	from torch.nn import init
	init.normal_(net[0].weight, mean=0.0, std=0.01)
	init.constant_(net[0].bias, val=0.0) 
	```
	- **`4.损失函数和优化函数`**
	```
	loss = nn.MSELoss()
	import torch.optim as optim
	optimizer = optim.SGD(net.parameters(), lr=0.03)   # built-in random gradient descent function 
	```
	- **`5.模型训练`**
	```
	for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # reset gradient, equal to net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
	```

 ## softmax  
 ```
 是一个单层神经网络，输出时通过softmax operator将输出值变换成值为正且和为1的概率分布，把预测概率最大的类别作为输出类别
 损失函数：交叉熵损失函数，关心对正确类别的预测概率，因为只要其值足够大，就可以确保分类结果正确。（平方损失则过于严格）
 ```
   - **`从零开始的实现核心代码`**
        - **`1.参数初始化`**
	```
	W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
	b = torch.zeros(num_outputs, dtype=torch.float)
	W.requires_grad_(requires_grad=True)
	b.requires_grad_(requires_grad=True)
	```
	- **`2.定义模型`**
	```
	def softmax(X):
		X_exp = X.exp()
		partition = X_exp.sum(dim=1,keepdim=True)
		return X_exp/partition
	def net(X):
		return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
	```
	-  **`3.损失函数和准确率`**
	```
	def cross_entry(y_hat,y):
		return - torch.log(y_hat.gather(1,y_view(-1, 1)))
	def accuracy(y_hat, y):
		return (y_hat.argmax(dim=1) == y).float().mean().item()
	
	```
  - **`pytorch实现`**
	- **`1.定义模型`**
	```
	net = nn.Sequential(
        # FlattenLayer(),
        # nn.Linear(num_inputs, num_outputs)
        OrderedDict([
          ('flatten', FlattenLayer()),
          ('linear', nn.Linear(num_inputs, num_outputs))])
        )
	```
	- **`2.参数初始化`**
	```
	init.normal_(net.linear.weight, mean=0, std=0.01)
	init.constant_(net.linear.bias, val=0)
	```
	- **`3.损失函数和优化函数`**
	```
	loss = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
	```
 ## 感知机
  ```
  多层感知机就是含有至少一个隐藏层的由全连接层组成的神经网络，且每个隐藏层的输出通过激活函数进行变换。
  关于激活函数的选择：
	ReLu函数是一个通用的激活函数，目前在大多数情况下使用。但是，ReLU函数只能在隐藏层中使用。
	用于分类器时，sigmoid函数及其组合通常效果更好。由于梯度消失问题，有时要避免使用sigmoid和tanh函数。
	在神经网络层数较多的时候，最好使用ReLu函数，ReLu函数比较简单计算量少，而sigmoid和tanh函数计算量大很多。
	在选择激活函数的时候可以先选用ReLu函数如果效果不理想可以尝试其他激活函数。
  ```
  - **`从零开始的实现核心代码`**
    - **`1.参数初始化`**
	```
	W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float) ##隐藏层参数
	b1 = torch.zeros(num_hiddens, dtype=torch.float)
	W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float) ##输出层参数
	b2 = torch.zeros(num_outputs, dtype=torch.float)

	params = [W1, b1, W2, b2]
	for param in params:
		param.requires_grad_(requires_grad=True)
	```
	- **`2.激活函数与网络、损失函数`**
	```
	def relu(X):
		return torch.max(input=X, other=torch.tensor(0.0))
		
	def net(X):
		X = X.view((-1, num_inputs))
		H = relu(torch.matmul(X, W1) + b1)
		return torch.matmul(H, W2) + b2
		
	loss = torch.nn.CrossEntropyLoss()
	```
  - **`pytorch实现`**
	- **`1.参数初始化和模型`**
	```
	net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs), 
        )
    
	for params in net.parameters():
		init.normal_(params, mean=0, std=0.01)
	```
	- **`2.损失与优化`**
	```
	loss = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
	```
## 文本预处理

文本是一类序列数据，一篇文章可以看作是字符或单词的序列，本节将介绍文本数据的常见预处理步骤，预处理通常包括四个步骤：

1.读入文本
2.分词
3.建立字典，将每个词映射到一个唯一的索引（index）
4.将文本从词的序列转换为索引的序列，方便输入模型

  - **`文本预处理pytorch`**
	- **`1.分词`**
	将一个句子划分成若干个词（token），转换为一个词的序列
	```
	def tokensize(sentences,token='word'):
		#token:做哪一个级别的分词
		if token == 'word':
			return [sentence.split(' ') for sentence in sentences]
		elif token == 'char':
			return [list(sentence) for sentence in sentences]
		else:
			print('ERROR: unkown token type' + token)
	```

	前面介绍的分词方式非常简单，它至少有以下几个缺点:

	1. 标点符号通常可以提供语义信息，但是我们的方法直接将其丢弃了
	2. 类似“shouldn't", "doesn't"这样的词会被错误地处理
	3. 类似"Mr.", "Dr."这样的词会被错误地处理

	我们可以通过引入更复杂的规则来解决这些问题，但是事实上，有一些现有的工具可以很好地进行分词，如：[spaCy](https://spacy.io/)和[NLTK](https://www.nltk.org/)。
	```
	text = "Mr. Chen doesn't agree with my suggestion."
	import spacy
	nlp = spacy.load('en_core_web_sm')
	doc = nlp(text)
	print([token.text for token in doc])
	'''
	['Mr.', 'Chen', 'does', "n't", 'agree', 'with', 'my', 'suggestion', '.']
	'''
	
	from nltk.tokenize import word_tokenize
	from nltk import data
	data.path.append('/home/kesci/input/nltk_data3784/nltk_data')
	print(word_tokenize(text))
	'''
	['Mr.', 'Chen', 'does', "n't", 'agree', 'with', 'my', 'suggestion', '.']
	'''
	```
	
	- **`2.建立字典`**
	为了方便模型处理，我们需要将字符串转换为数字。因此我们需要先构建一个字典（vocabulary），将每个词映射到一个唯一的索引编号。
	
	```
	class Vocab(object):
    ##去重、删除部分、添加特殊token、映射
		def __init__(self, tokens, min_freq=0, use_special_tokens=False):
			##min_freq阈值，小于它忽略掉
			# :tokens为语料库上分词后所有的词 
			counter = count_corpus(tokens)   ##词频字典<key,value> <词，词频>
			self.token_freqs = list(counter.items()) ##拿出2元组，构造一个列表
			self.idx_to_token = [] ##记录需要维护的词
			
			if use_special_tokens:
				# padding, begin of sentence, end of sentence, unknown
				self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
				self.idx_to_token += ['<pad>', '<bos>', '<eos>', '<unk>'] 
				###pad短句补长所用,bos、eos标志开始结束，eos，unk新事物未出现在语料库--未登录词
			else:
				self.unk = 0
				self.idx_to_token += ['<unk>']
			self.idx_to_token += [token for token, freq in self.token_freqs
							if freq >= min_freq and token not in self.idx_to_token]
			self.token_to_idx = dict()##从词到索引号的映射
			for idx, token in enumerate(self.idx_to_token):
				self.token_to_idx[token] = idx

		def __len__(self):  ###字典大小
			return len(self.idx_to_token)

		def __getitem__(self, tokens):  ###
			if not isinstance(tokens, (list, tuple)):
				return self.token_to_idx.get(tokens, self.unk)##直接从字典中寻找
			return [self.__getitem__(token) for token in tokens]

		def to_tokens(self, indices):  ###给定索引，返回对应的词
			if not isinstance(indices, (list, tuple)):
				return self.idx_to_token[indices]
			return [self.idx_to_token[index] for index in indices]

	##统计词频
	def count_corpus(sentences):
		tokens = [tk for st in sentences for tk in st]
		return collections.Counter(tokens)  # 返回一个字典，记录每个词的出现次数
	```
	- **`3.将词转为索引`**
	使用字典，我们可以将原文本中的句子从单词序列转换为索引序列
	```
	for i in range(8, 10):
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])

	##words: ['the', 'time', 'traveller', 'for', 'so', 'it', 'will', 'be', 'convenient', 'to', 'speak', 'of', 'him', '']
	##indices: [1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0]
	##words: ['was', 'expounding', 'a', 'recondite', 'matter', 'to', 'us', 'his', 'grey', 'eyes', 'shone', 'and']
	##indices: [20, 21, 22, 23, 24, 16, 25, 26, 27, 28, 29, 30]
	
	```
	
## 语言模型


一段自然语言文本可以看作是一个离散时间序列，给定一个长度为$T$的词的序列$w_1, w_2, \ldots, w_T$，
语言模型的目标就是评估该序列是否合理，即计算该序列的概率：
$$
P(w_1, w_2, \ldots, w_T).
$$
假设序列$w_1, w_2, \ldots, w_T$中的每个词是依次生成的，我们有
$$
\begin{align*}
P(w_1, w_2, \ldots, w_T)
&= \prod_{t=1}^T P(w_t \mid w_1, \ldots, w_{t-1})\\
&= P(w_1)P(w_2 \mid w_1) \cdots P(w_T \mid w_1w_2\cdots w_{T-1})
\end{align*}
$$
例如，一段含有4个词的文本序列的概率
$$
P(w_1, w_2, w_3, w_4) =  P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_1, w_2) P(w_4 \mid w_1, w_2, w_3).
$$
语言模型的参数就是词的概率以及给定前几个词情况下的条件概率。设训练数据集为一个大型文本语料库，如维基百科的所有条目，词的概率可以通过该词在训练数据集中的相对词频来计算，例如，$w_1$的概率可以计算为：
$$
\hat P(w_1) = \frac{n(w_1)}{n}
$$
其中$n(w_1)$为语料库中以$w_1$作为第一个词的文本的数量，$n$为语料库中文本的总数量。
类似的，给定$w_1$情况下，$w_2$的条件概率可以计算为：
$$
\hat P(w_2 \mid w_1) = \frac{n(w_1, w_2)}{n(w_1)}
$$
其中$n(w_1, w_2)$为语料库中以$w_1$作为第一个词，$w_2$作为第二个词的文本的数量。

  - **`1.n元语法`**
       序列长度增加，计算和存储多个词共同出现的概率的复杂度会呈指数级增加。$n$元语法通过马尔可夫假设简化模型，马尔科夫假设是指一个词的出现只与前面$n$个词相关，即$n$阶马尔可夫链（Markov chain of order $n$），如果$n=1$，那么有$P(w_3 \mid w_1, w_2) = P(w_3 \mid w_2)$。基于$n-1$阶马尔可夫链，我们可以将语言模型改写为
	$$
	P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^T P(w_t \mid w_{t-(n-1)}, \ldots, w_{t-1}) .
	$$
	以上也叫$n$元语法（$n$-grams），它是基于$n - 1$阶马尔可夫链的概率语言模型。例如，当$n=2$时，含有4个词的文本序列的概率就可以改写为：
	$$
	\begin{align*}
	P(w_1, w_2, w_3, w_4)
	&= P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_1, w_2) P(w_4 \mid w_1, w_2, w_3)\\
	&= P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_2) P(w_4 \mid w_3)
	\end{align*}
	$$
	当$n$分别为1、2和3时，我们将其分别称作一元语法（unigram）、二元语法（bigram）和三元语法（trigram）。例如，长度为4的序列$w_1, w_2, w_3, w_4$在一元语法、二元语法和三元语法中的概率分别为
	$$
	\begin{aligned}
	P(w_1, w_2, w_3, w_4) &=  P(w_1) P(w_2) P(w_3) P(w_4) ,\\
	P(w_1, w_2, w_3, w_4) &=  P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_2) P(w_4 \mid w_3) ,\\
	P(w_1, w_2, w_3, w_4) &=  P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_1, w_2) P(w_4 \mid w_2, w_3) .
	\end{aligned}
	$$
	当$n$较小时，$n$元语法往往并不准确。例如，在一元语法中，由三个词组成的句子“你走先”和“你先走”的概率是一样的。然而，当$n$较大时，$n$元语法需要计算并存储大量的词频和多词相邻频率。
	
	> **缺点：1.参数空间大、计算开销大，2.数据稀疏，齐夫定律：单词的词频与单词的排名成反比**
	
  - **`2.语言模型pytorch小例（中文）`**
    - **`建立字符索引`**
	
	```
	def load_data_jay_lyrics():
		with open('/home/kesci/input/jaychou_lyrics4703/jaychou_lyrics.txt') as f:
			corpus_chars = f.read()
		corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
		corpus_chars = corpus_chars[0:10000]
		idx_to_char = list(set(corpus_chars))   # 去重，得到索引到字符的映射
		char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)]) # 字符到索引的映射
		vocab_size = len(char_to_idx)
		corpus_indices = [char_to_idx[char] for char in corpus_chars] # 将每个字符转化为索引，得到一个索引的序列
		return corpus_indices, char_to_idx, idx_to_char, vocab_size
	```
  - **`3.时序数据采样`**
	在训练中我们需要每次随机读取小批量样本和标签。与之前章节的实验数据不同的是，时序数据的一个样本通常包含连续的字符。假设时间步数为5，样本序列为5个字符，即“想”“要”“有”“直”“升”。
	**该样本的标签序列为这些字符分别在训练集中的下一个字符**，即“要”“有”“直”“升”“机”，即$X$=“想要有直升”，$Y$=“要有直升机”。
	
	现在我们考虑序列“想要有直升机，想要和你飞到宇宙去”，如果时间步数为5，有以下可能的样本和标签：
	* $X$：“想要有直升”，$Y$：“要有直升机”
	* $X$：“要有直升机”，$Y$：“有直升机，”
	* $X$：“有直升机，”，$Y$：“直升机，想”
	* ...
	* $X$：“要和你飞到”，$Y$：“和你飞到宇”
	* $X$：“和你飞到宇”，$Y$：“你飞到宇宙”
	* $X$：“你飞到宇宙”，$Y$：“飞到宇宙去”

	可以看到，如果序列的长度为$T$，时间步数为$n$，那么一共有$T-n$个合法的样本，但是这些样本有大量的重合，我们通常采用更加高效的采样方式。我们有两种方式对时序数据进行采样，分别是随机采样和相邻采样。

	- **`随机采样`**
		每次从数据里随机采样一个小批量。其中批量大小`batch_size`是每个小批量的样本数，`num_steps`是每个样本所包含的时间步数。
		在随机采样中，每个样本是原始序列上任意截取的一段序列，相邻的两个随机小批量在原始序列上的位置不一定相毗邻。
		```
		import torch
		import random
		def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
			# 减1是因为对于长度为n的序列，X最多只有包含其中的前n - 1个字符
			num_examples = (len(corpus_indices) - 1) // num_steps  # 下取整，得到不重叠情况下的样本个数
			# 每个样本的第一个字符在corpus_indices中的下标
			example_indices = [i * num_steps for i in range(num_examples)]  
			random.shuffle(example_indices)

			def _data(i):
				# 返回从i开始的长为num_steps的序列
				return corpus_indices[i: i + num_steps]
			if device is None:
				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			
			for i in range(0, num_examples, batch_size):
				# 每次选出batch_size个随机样本
				batch_indices = example_indices[i: i + batch_size]  # 当前batch的各个样本的首字符的下标
				X = [_data(j) for j in batch_indices]
				Y = [_data(j + 1) for j in batch_indices]
				yield torch.tensor(X, device=device), torch.tensor(Y, device=device)
		```
		
	- **`随机采样`**
		在相邻采样中，相邻的两个随机小批量在原始序列上的位置相毗邻
		```
		def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
			if device is None:
				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			corpus_len = len(corpus_indices) // batch_size * batch_size  # 保留下来的序列的长度
			corpus_indices = corpus_indices[: corpus_len]  # 仅保留前corpus_len个字符
			indices = torch.tensor(corpus_indices, device=device)
			indices = indices.view(batch_size, -1)  # resize成(batch_size, )
			batch_num = (indices.shape[1] - 1) // num_steps
			for i in range(batch_num):
				i = i * num_steps
				X = indices[:, i: i + num_steps]
				Y = indices[:, i + 1: i + num_steps + 1]
				yield X, Y
		```
		
## 循环神经网络基础
循环神经网络是基于当前的输入与过去的输入序列，预测序列的下一个字符。循环神经网络引入一个隐藏变量$H$，用$H_{t}$表示$H$在时间步$t$的值。$H_{t}$的计算基于$X_{t}$和$H_{t-1}$，可以认为$H_{t}$记录了到当前字符为止的序列信息，利用$H_{t}$对序列的下一个字符进行预测。
![Image Name](https://cdn.kesci.com/upload/image/q5jkm0v44i.png?imageView2/0/w/640/h/640)
  - **`结构`**
	假设$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$是时间步$t$的小批量输入，$\boldsymbol{H}_t  \in \mathbb{R}^{n \times h}$是该时间步的隐藏变量，则：
	$$
	\boldsymbol{H}_t = \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hh}  + \boldsymbol{b}_h).
	$$
	其中，$\boldsymbol{W}_{xh} \in \mathbb{R}^{d \times h}$，$\boldsymbol{W}_{hh} \in \mathbb{R}^{h \times h}$，$\boldsymbol{b}_{h} \in \mathbb{R}^{1 \times h}$，$\phi$函数是非线性激活函数。由于引入了$\boldsymbol{H}_{t-1} \boldsymbol{W}_{hh}$，$H_{t}$能够捕捉截至当前时间步的序列的历史信息，就像是神经网络当前时间步的状态或记忆一样。由于$H_{t}$的计算基于$H_{t-1}$，上式的计算是循环的，使用循环计算的网络即循环神经网络（recurrent neural network）。
	在时间步$t$，输出层的输出为：
	$$
	\boldsymbol{O}_t = \boldsymbol{H}_t \boldsymbol{W}_{hq} + \boldsymbol{b}_q.
	$$
	其中$\boldsymbol{W}_{hq} \in \mathbb{R}^{h \times q}$，$\boldsymbol{b}_q \in \mathbb{R}^{1 \times q}$。
	
  - **`从零开始实现循环神经网络`**
	- **`one-hot`**
		需要将字符表示成向量，这里采用one-hot向量。假设词典大小是$N$，每次字符对应一个从$0$到$N-1$的唯一的索引，则该字符的向量是一个长度为$N$的向量，若字符的索引是$i$，则该向量的第$i$个位置为$1$，其他位置为$0$。下面分别展示了索引为0和2的one-hot向量，向量长度等于词典大小。
	```
	def one_hot(x, n_class, dtype=torch.float32):
		##n_class :字典的大小
		result = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)  # shape: (n, n_class)
		result.scatter_(1, x.long().view(-1, 1), 1)  # result[i, x[i, 0]] = 1，将每一行的x[i,0]改写为1
		return result
	```
	   我们每次采样的小批量的形状是（批量大小, 时间步数）。下面的函数将这样的小批量变换成数个形状为（批量大小, 词典大小）的矩阵，矩阵个数等于时间步数。也就是说，时间步$t$的输入为$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$，其中$n$为批量大小，$d$为词向量大小，即one-hot向量长度（词典大小）。
	```
	def to_onehot(X, n_class):
		return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]
	```
	- **`初始化模型参数`**
	```
	num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
	# num_inputs: d
	# num_hiddens: h, 隐藏单元的个数是超参数
	# num_outputs: q

	def get_params():
		def _one(shape):
			param = torch.zeros(shape, device=device, dtype=torch.float32)
			nn.init.normal_(param, 0, 0.01)
			return torch.nn.Parameter(param)

		# 隐藏层参数
		W_xh = _one((num_inputs, num_hiddens))
		W_hh = _one((num_hiddens, num_hiddens))
		b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device))
		# 输出层参数
		W_hq = _one((num_hiddens, num_outputs))
		b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device))
		return (W_xh, W_hh, b_h, W_hq, b_q)
	```
	- **`定义模型`**
	完成前向计算
	函数`rnn`用循环的方式依次完成循环神经网络每个时间步的计算
	```
	def rnn(inputs, state, params):
		# inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
		W_xh, W_hh, b_h, W_hq, b_q = params
		H, = state ##存储状态，RNN只有一个，但是LSTM不止一个，为了复用，定义为元组
		outputs = []##各个时间步的输出
		for X in inputs:
			H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h) ##更新
			Y = torch.matmul(H, W_hq) + b_q
			outputs.append(Y)
		return outputs, (H,)
	```
	函数init_rnn_state初始化隐藏变量的初始状态，这里的返回值是一个元组。
	```
	def init_rnn_state(batch_size, num_hiddens, device):
		return (torch.zeros((batch_size, num_hiddens), device=device), )
	```
	- **`剪裁梯度`**
	循环神经网络中较容易出现梯度衰减或梯度爆炸，这会导致网络几乎无法训练。裁剪梯度（clip gradient）是一种应对梯度爆炸的方法。
	假设我们把所有模型参数的梯度拼接成一个向量 $\boldsymbol{g}$，并设裁剪的阈值是$\theta$。裁剪后的梯度的$L_2$范数不超过$\theta$
	$$
	 \min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right)\boldsymbol{g}
	$$
	
	
	```
	def grad_clipping(params, theta, device):
		norm = torch.tensor([0.0], device=device)
		for param in params:
			norm += (param.grad.data ** 2).sum()
		norm = norm.sqrt().item() ##||g||
		if norm > theta:
			for param in params:
				param.grad.data *= (theta / norm)
	```
	- **`定义预测函数`**
		以下函数基于前缀`prefix`（含有数个字符的字符串）来预测接下来的`num_chars`个字符。我们将循环神经单元`rnn`设置成了函数参数，这样在后面小节介绍其他循环神经网络时能重复使用这个函数。
	
	```
	def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
					num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
		state = init_rnn_state(1, num_hiddens, device)
		output = [char_to_idx[prefix[0]]]   # output记录prefix加上预测的num_chars个字符
		for t in range(num_chars + len(prefix) - 1):
			# 将上一时间步的输出作为当前时间步的输入
			X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
			# 计算输出和更新隐藏状态
			(Y, state) = rnn(X, state, params)
			# 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
			if t < len(prefix) - 1:
				output.append(char_to_idx[prefix[t + 1]])##前len(prefix)只需添加
			else:
				output.append(Y[0].argmax(dim=1).item()) ##后面的，通过预测得到
		return ''.join([idx_to_char[i] for i in output])
	```
	- **`困惑度`**

	我们通常使用困惑度（perplexity）来评价语言模型的好坏。“softmax回归”中交叉熵损失函数的定义。困惑度是对交叉熵损失函数做指数运算后得到的值。特别地，

	* 最佳情况下，模型总是把标签类别的概率预测为1，此时困惑度为1；
	* 最坏情况下，模型总是把标签类别的概率预测为0，此时困惑度为正无穷；
	* 基线情况下，模型总是预测所有类别的概率都相同，此时困惑度为类别个数。

	显然，任何一个有效模型的困惑度必须小于类别个数。在本例中，困惑度必须小于词典大小`vocab_size`。
	
	- **`定义模型训练函数`**
	跟之前模型训练函数相比，这里的模型训练函数有以下几点不同：
	1. 使用困惑度评价模型。
	2. 在迭代模型参数前裁剪梯度。
	3. 对时序数据采用不同采样方法将导致隐藏状态初始化的不同。
	
	```
	def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss() ##损失函数

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:  # 否则需要使用detach函数从计算图分离隐藏状态
                for s in state:
                    s.detach_()
            # inputs是num_steps个形状为(batch_size, vocab_size)的矩阵
            inputs = to_onehot(X, vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # Y的形状是(batch_size, num_steps)，转置后再变成形状为
            # (num_steps * batch_size,)的向量，这样跟输出的行一一对应
            y = torch.flatten(Y.T)
            # 使用交叉熵损失计算平均分类误差
            l = loss(outputs, y.long())
            
            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)  # 裁剪梯度
            d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, device, idx_to_char, char_to_idx))
	```
	
