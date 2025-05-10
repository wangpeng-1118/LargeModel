# Abstract

LLM在生成长文本时，进行 LLM 推理服务会面临挑战：模型的中间状态（即 KV 缓存）占用的内存非常大，并且随着序列长度和批大小的增加而线性增长。

InfiniGen:一个专为长文本生成设计的全新 KV 缓存管理框架，可与现代的**基于卸载（offloading）机制的推理系统**协同工作

key insight of InfiniGen:用于计算 Transformer 下一层注意力的关键 token，可以通过在当前层输入上执行==最小化“预演”操作(minimal rehearsal)==，并结合下一层部分查询权重与 KV 缓存内容进行推测得到。

系统只需预取关键的 KV 缓存项，而无需将全部缓存加载

# Introduction

随着用户对更长序列和更大 batch size 的需求不断增长，KV 缓存带来的内存压力将变得更加突出。

InfiniGen:为长文本生成设计的KV缓存管理框架，旨在与现代卸载式推理系统协同工作。

两个设计理念：

* 通过在第 i 层中对注意力计算进行最小化预演，从而推测出对生成下一个 token 起关键作用的 KV 缓存项，并丢弃非关键项。
* 利用 CPU 大容量内存，在 CPU 中维护 KV 缓存池，从而确保在生成长文本时始终能够识别出关键的 KV 值，动态移除不经常使用的标记的KV Cache。

特别是，InfiniGen通过离线操作模型权重，使推测更加高效和精确，==通过倾斜Transformer架构的查询和键矩阵，强调某些重要的列。==

在预填充阶段，当推理请求的提示和输入最初被处理时，InfiniGen会生成后续解码阶段（即输出生成阶段）使用的部分权重。在解码阶段的第i - 1层，InfiniGen使用第i - 1层的注意力输入、部分查询权重和第i层的部分键缓存来推测下一层（第i层）的注意力模式。

contribution：

* 提出了InfiniGen，这是一个动态KV缓存管理框架，通过智能管理CPU内存中的KV缓存，与现代基于卸载的LLM服务系统协同工作（利用了CPU出内存大的优势）
* 提出了一种新的KV缓存预取技术与短暂修剪，推测后续的注意力层的注意力模式，并带来了KV缓存的主要部分，而在CPU内存中保留其余部分（预取需要用到的少部分KV Cache）
* 我们在一个现代的基于卸载的推理系统上实现了InfiniGen，并证明它大大优于现有的KV缓存管理方法，实现了高达3.00倍的性能，同时还提供了更好的模型精度。

# background

输入张量的维度N * D,N为输入的token的数量，D是张量的维度（模型的维度）。先进行层归一化，再输入到注意力层。通过不同的权重矩阵，每个token对应生成Q，K，V。矩阵被重塑为H * D * d，H是注意头的数量，d是头的维度，D = H * d。

在本工作中，我们将解码阶段中的每次标记生成称为一次迭代（iteration）。

**outlier**：指某些参数值（或激活值）特别大或特别小，远离平均值，跟其他值差很多的“异常数值”。可能出现在weights和activations中，在低比特量化中，是精度损失的主要来源。

#  Motivation

现代LLM服务系统，如DeepSpeed和FlexGen，已经支持将模型权重或KV缓存卸载到CPU内存。当涉及到基于卸载的推理系统时，KV缓存大小变得更加成问题，这是由于CPU和GPU之间的低PCIe带宽，它成为了新的关键瓶颈

尽管通过量化压缩KV缓存可以潜在地减少基于卸载的系统中的数据传输开销，但这并不是一个根本性的解决方案，因为量化并没有解决KV缓存问题的根本原因，即KV条目随着序列长度的线性增长

**Challenges in KV Cache Management**: 减轻KV缓存从CPU到GPU的传输开销的根本方法是通过识别计算注意力分数的关键键和值来减少要加载的KV缓存的量

先前的关于管理KV缓存的工作并不能有效地解决基于卸载的推理系统中的以下挑战

* Dynamic nature of attention patterns across iterations(迭代间注意力模式的动态性)：在当前迭代中被认为不重要的标记可能在后续迭代中变得重要。因此，H2O在序列长度超过KV缓存预算时开始与动态注意力模式作斗争，导致余弦相似度低于Optimal情况
* **Adjusting the number of KV entries across layers（跨层调整KV条目数量）**:不同层对KV缓存的需求不同，我们需要跨不同层动态调整参与注意力计算的关键标记数量，以有效利用KV缓存预算
* **Adjusting the number of KV entries across queries（跨查询调整KV条目数量）**：

# InfiniGen Design

## Overview

InfiniGen的核心设计理念是利用丰富的CPU内存容量，以增加在KV缓存中识别重要token的窗口大小。

并不会将整个KV Cach保存在GPU中，二十仅加载少数几个重要token的key和value，动态地丢弃其他不重要的KV Cache。

InfiniGen 不是直接通过列求和找 token，而是通过列求和找“最重要的维度（列）”，然后**再在这些维度上看哪些 token（行）值最大**，从而判断最关键的 token。

因为 Transformer 的 Attention 是**点积**，如果某些维度（列）在 Q 和 K 中都很大，就会对 Attention 分数贡献非常大；

只使用上面的方法求出来的列进行attention权重的计算，推测是通过在前一层执行当前层的minimal rehearsal来完成的。然后在decode阶段选择权重位于$[max - \alpha, max]$的token进行计算

我们计算
$$
Q' = X \cdot W_Q \cdot V = X \cdot U \Sigma V^T V
$$
注意到：
$$
V^T \cdot V = I
$$
所以上式等价于：
$$
Q' = X \cdot U \Sigma
$$
这就完成了一个“坐标空间旋转”，**把原来的 Q 向量投影到了一个新的坐标系中**，而这个坐标系是：**按照奇异值从大到小排序的方向（重要性排序的方向）**



# 个人总结

## 需要优化的point

在大模型中，我们使用transformer框架，在预测的过程中会产生大量的KV Cache，在需要生成长文本的任务中，这些缓存的数据量甚至会超过模型的权重。在类似FlexGen中，使用offloading的方式将大部分的KV Cache存储在CPU的内存中并结合TopK attention，这样当需要使用到某些KV Cache的时候再将其传到GPU的内存中。但是此时的一个瓶颈在于PCIe带宽，当我们需要某些KV Cache进行权重计算的时候，GPU可能会会有很长一段时间在等待KV Cache从CPU传到GPU的内存中，时间上有很大的消耗。已知的方法有Quantization，Unimportant Token Eviction，这些都会导致传入到模型的数据丢失。

（本文假设GPU中的内存存放的时weights和avtivation，并且GPU的内存并没有被占满）

## Motivation

为了提升效率，可以将KV Cache传入到GPU memory过程和GPU进行上一层的attention+FFN的计算并行，应该尽可能保证传输的时间比GPU处理上一层的数据的时间要短。

本文在Attention操作中，在计算V的权重时进行了优化，使用PCA中的方法使用这一层中的权重中数据对下一层的Q K进行降维，这样就大大减少了下一层计算权重时需要传输到GPU内存的数据减少。



## 背景知识

在一个多维的空间中，有n个数据点，每个数据的维度为d，假设这些数据组成一个n*d维的矩阵M，使用一个对角矩阵对M进行右乘，得到的矩阵相对于M的区别在于对数据的每个维度的大小进行了放缩，体现在坐标系上面就是对数据进行了拉伸操作。

类似，让一个正交矩阵右乘M，得到的结果体现在坐标上面就是将数据集中的每个点围绕着坐标原点进行了旋转操作。所以在神经网络中某个矩阵乘以一个正交矩阵后，并不会丢失数据。

**主成分分析**：将储存数据的矩阵X进行SVD分解
$$
X = U \Sigma V^T
$$
然后使用得到的矩阵$V$,的k个维度的列向量右乘原矩阵$X$,得到将X进行降维后的数据，如果乘以完成的矩阵矩阵$V$,相当于对矩阵进行旋转/翻转，并不会损失原本的信息。为了使得降维后的数据相对于元数据损失的数据最小，我们因该选择$V$中最大的k个奇异值对应的列。$X$分成的三个矩阵在坐标系中可以看作是将一个坐标系进行拉伸然后再进行一次旋转/翻转。当对$X$右乘矩阵$V$之后的结果可以看成是对一个正交基的拉伸，此时对角线矩阵中值越大的维度对应的元数据中的信息量就越大。

**动机**：作者观察到，对query和key矩阵进行skew操作，使得矩阵中少数通道的值比其他通道要大得多，而attentio中计算V的权重的时候是对单个token的Q和其他token的K进行点乘，所以我们可以使用skew后的矩阵的少数通道的值计算两个token之间的权重。

**UVM（Unified Virtual Memory）**：提供GPU和CPU内存统一管理的机制。因为 CPU 和 GPU 之间的带宽（比如 PCIe 3.0）远远低于 GPU 内存带宽。**当数据太大时，UVM会频繁触发 Page Fault（缺页中断）**

**Full Cache**：使用完整的KV Cache

**w/o Skewing**：不使用Skewing，但是只使用一部分Cache

**w/ Skewing**：使用skewing后选出来的更有代表性的Top列

### 标准评测数据集

**COPA（Choice of Plausible Alternatives）**：给一个场景，要从两个选项中选出最合理的因果关系

**Open Book QA（Open Book Question Answering）**：结合常识和小学科学知识回答选择题

**WinoGrande（Winograd Challenge）**：两个人，两个物体，要根据句子推理谁是谁

**PIQA（Physical Interaction QA）**：物理常识推理，询问最合理的物理行为，比如怎么摆放怎么使用工具

**RTE（Recognizing Textual Entailment）**：文本蕴含推理，给两句话，判断第二句是不是第一句推理出来的结果

























