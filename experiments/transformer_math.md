# Decoder-only Transformer 架构的数学重构：以矩阵阶数为不变量

> 本文是 `experiments/transformer.md` 的深化版。原文给出了 Decoder-only Transformer 的骨架与数据流；本文补上每一步的严格数学推导，并把**矩阵的形状（阶数）**作为贯穿全文的不变量。阅读时请始终问自己三个问题：
>
> 1. 这个操作的输入、输出形状是什么？
> 2. 它是否破坏了残差流的宽度 $d$？
> 3. 它是在哪个轴上混合信息：batch $B$、序列 $N$、特征 $d$，还是词表 $V$？

---

## 0. 记号与“形状不变量”

| 符号 | 含义 | 本项目典型值 |
| :--- | :--- | :--- |
| $B$ | batch size | 32 |
| $N$ | 序列长度（context length） | 256 |
| $d$ | 模型宽度 $d_{model}$ | 768 |
| $h$ | 注意力头数 | 6 |
| $d_k$ | 每个头的维度，$d_k=d/h$ | 128 |
| $d_{ff}$ | FFN 隐藏维度 | 2048 |
| $V$ | 词表大小 | 50257 |
| $L$ | Transformer 层数 | 6 |

本文把原文中的 token 数 $n$ 记作 $N$，把嵌入维度记作 $d$。

**核心不变量：**

$$
H^{(\ell)} \in \mathbb{R}^{B\times N\times d},
\qquad \ell=0,1,\dots,L.
$$

也就是说，**残差流（residual stream）的形状从输入到最后一层始终不变**。Transformer 的所有“花样”都发生在残差流内部：注意力在 $N$ 轴上混合 token，FFN 在 $d$ 轴上混合特征，多头把 $d$ 临时切成 $h\times d_k$，FFN 把 $d$ 临时撑到 $d_{ff}$，但最终都必须回到 $B\times N\times d$。

这条不变量不仅是理解架构的钥匙，也是排查代码 shape 错误的第一原则。

---

## 1. 语言模型：从联合分布到交叉熵

### 1.1 自回归分解

给定一个 token 序列

$$
x_{1:N}=(x_1,x_2,\dots,x_N),\qquad x_t\in\{1,2,\dots,V\},
$$

语言模型用链式法则把联合概率分解为条件概率的乘积：

$$
p_\theta(x_{1:N})=\prod_{t=1}^{N}p_\theta(x_t\mid x_{<t}),
$$

其中 $x_{<t}=(x_1,\dots,x_{t-1})$。这就是“next-token prediction”的数学定义。模型不需要一次性预测整个序列，只需要在每一步给定前缀，输出下一个 token 的分布。

### 1.2 交叉熵与 logits

模型在位置 $t$ 输出一个 logits 向量

$$
z_t=f_\theta(x_{<t})\in\mathbb{R}^{V}.
$$

经过 softmax 得到概率分布：

$$
p_\theta(x_t=v\mid x_{<t})
=
\frac{\exp(z_{t,v})}{\sum_{u=1}^{V}\exp(z_{t,u})}.
$$

负对数似然（negative log-likelihood, NLL）为

$$
\mathcal{L}(\theta)
=
-\frac{1}{N}\sum_{t=1}^{N}\log p_\theta(x_t\mid x_{<t}).
$$

对单个位置，记目标 token 为 $y_t=x_t$，则

$$
\ell_t
=
-z_{t,y_t}
+
\log\sum_{u=1}^{V}\exp(z_{t,u}).
$$

这就是交叉熵（cross-entropy）损失。训练时所有位置并行计算，但因果掩码保证位置 $t$ 的预测只依赖 $x_{\le t}$。实现上通常把输入序列右移一位作为标签：位置 $t$ 的 logits 对应目标 $x_{t+1}$。

### 1.3 softmax 的雅可比与梯度

设 $a=\operatorname{softmax}(s)$，即

$$
a_i=\frac{e^{s_i}}{\sum_{k=1}^{V}e^{s_k}}.
$$

对 $s_j$ 求偏导：

$$
\frac{\partial a_i}{\partial s_j}
=
\frac{\delta_{ij}e^{s_i}\sum_k e^{s_k}-e^{s_i}e^{s_j}}
{\left(\sum_k e^{s_k}\right)^2}
=
a_i(\delta_{ij}-a_j).
$$

写成矩阵形式：

$$
J_a=\frac{\partial a}{\partial s}
=
\operatorname{diag}(a)-aa^\top
\in\mathbb{R}^{V\times V}.
$$

对于交叉熵 $\ell=-\log a_{y}$，有

$$
\frac{\partial \ell}{\partial s_j}
=
-\frac{1}{a_y}\frac{\partial a_y}{\partial s_j}
=
-\frac{1}{a_y}a_y(\delta_{yj}-a_j)
=
a_j-\delta_{yj}.
$$

因此

$$
\nabla_s\ell=a-y,
\qquad y=\operatorname{onehot}(y).
$$

这个极其简洁的结果是整条反向传播链的起点：**logits 的梯度 = 预测分布减去真实 one-hot 分布**。

### 1.4 模型的输入输出阶数

从矩阵视角看，整个模型是一个函数

$$
f_\theta:\{1,\dots,V\}^{B\times N}\longrightarrow\mathbb{R}^{B\times N\times V}.
$$

输入是离散 token 的二维网格，输出是三维 logits 张量。中间的隐藏状态始终是

$$
H\in\mathbb{R}^{B\times N\times d}.
$$

词表维度 $V$ 只在两个地方出现：**输入端的 token embedding** 和 **输出端的 LM Head**。这是后文反复使用的一个结构性事实。

---

## 2. 嵌入层：把离散 token 提升到连续空间

### 2.1 one-hot 与查表

设词嵌入矩阵

$$
E\in\mathbb{R}^{V\times d},
\qquad
e_v=E[v,:]\in\mathbb{R}^{d}.
$$

对 token $x_t$，构造 one-hot 向量 $o_t\in\{0,1\}^{V}$，则

$$
x_t^{(0)}=E^\top o_t=e_{x_t}\in\mathbb{R}^{d}.
$$

把 $N$ 个 one-hot 向量按行堆叠成 $O\in\mathbb{R}^{N\times V}$，得到矩阵形式：

$$
X^{(0)}=OE\in\mathbb{R}^{N\times d}.
$$

加入 batch 后：

$$
X^{(0)}\in\mathbb{R}^{B\times N\times d}.
$$

所以 embedding 层本质上是一个**查表**：one-hot 乘矩阵等价于取矩阵的某一行。参数量为

$$
Vd.
$$

### 2.2 权重绑定（weight tying）

LM Head 把隐藏状态映射回词表 logits：

$$
Z=H^{(L)}W_{LM},
\qquad
W_{LM}\in\mathbb{R}^{d\times V},
\qquad
Z\in\mathbb{R}^{B\times N\times V}.
$$

如果令

$$
W_{LM}=E^\top,
$$

则

$$
Z=H^{(L)}E^\top,
\qquad
z_{t,v}=\langle h_t,e_v\rangle.
$$

也就是说，token $v$ 的 logit 是隐藏向量 $h_t$ 与词向量 $e_v$ 的内积。输入嵌入和输出分类共享同一套几何空间，这就是**权重绑定（weight tying）**。

**为什么小模型必须绑定？**

当前配置下：

$$
Vd=50257\times 768=38{,}597{,}376,
$$

而模型总参数约为 $81.1$M，嵌入层占了

$$
\frac{38.6\text{M}}{81.1\text{M}}\approx 47.6\%.
$$

如果解绑，参数量会直接增加 $38.6$M，总参数变成 $119.7$M，而多出来的参数几乎全在输出层，极易过拟合。绑定相当于一种强正则化，同时把总参数从 $119.7$M 降到 $81.1$M（减少约 $32\%$）。

**为什么大模型可以解绑？**

GPT-3 的 $Vd\approx 50257\times 12288\approx 617$M，而总参数约 $175$B：

$$
\frac{617\text{M}}{175\text{B}}\approx 0.35\%.
$$

此时解绑的边际成本极低，独立输出层可以增加模型容量。**同一个架构决策，在小模型和大模型上会得出相反结论**，这是 scaling 改变设计取舍的第一个例子。

---

## 3. 位置编码：在不改变宽度的前提下注入顺序

### 3.1 加性绝对位置编码

最直接的做法是学习一个位置嵌入矩阵

$$
P\in\mathbb{R}^{N_{max}\times d},
\qquad
P_t=P[t,:]\in\mathbb{R}^{d},
$$

然后把它加到 token 嵌入上：

$$
X^{(0)}=OE+P_{1:N}.
$$

形状仍然是 $B\times N\times d$。这里体现了一个重要的工程原则：**位置编码必须与 token 嵌入同形**，否则会破坏残差流的宽度 $d$。

如果采用拼接（concatenation），形状会变成 $B\times N\times 2d$，后续所有投影矩阵都要改成 $2d$ 的输入，残差流宽度也会改变。加法则保持不变量：

$$
\mathbb{R}^{B\times N\times d}+\mathbb{R}^{1\times N\times d}
\longrightarrow
\mathbb{R}^{B\times N\times d}.
$$

### 3.2 RoPE 的二维推导

RoPE（Rotary Position Embedding）不把位置加到输入上，而是在注意力内部旋转 query 和 key。它要求 $d_k$ 为偶数，把每一对相邻维度 $(2i-1,2i)$ 看成一个二维平面。

定义第 $i$ 对维度的频率

$$
\theta_i=\text{base}^{-2(i-1)/d_k},
\qquad i=1,2,\dots,\frac{d_k}{2},
$$

通常 $\text{base}=10000$。位置 $m$ 在第 $i$ 个平面上的旋转矩阵为

$$
R_{m,i}
=
\begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i)\\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}
\in\mathbb{R}^{2\times 2}.
$$

把所有平面对角拼起来，得到

$$
R_m=\operatorname{diag}(R_{m,1},R_{m,2},\dots,R_{m,d_k/2})
\in\mathbb{R}^{d_k\times d_k}.
$$

设未旋转的 query、key 为

$$
\tilde q_m=W_Qx_m,\qquad \tilde k_n=W_Kx_n,
$$

旋转后为

$$
q_m=R_m\tilde q_m,\qquad k_n=R_n\tilde k_n.
$$

关键性质是旋转矩阵正交：

$$
R_m^\top R_m=I,
\qquad
R_m^\top R_n=R_{n-m}.
$$

因此

$$
\langle q_m,k_n\rangle
=
\tilde q_m^\top R_m^\top R_n\tilde k_n
=
\tilde q_m^\top R_{n-m}\tilde k_n.
$$

**旋转矩阵带来的位置效应只依赖于相对位置 $n-m$，而不依赖于绝对位置 $m,n$ 本身**。这就是 RoPE 的核心数学动机。

同时，旋转不改变向量长度：

$$
\|q_m\|_2=\|R_m\tilde q_m\|_2=\|\tilde q_m\|_2.
$$

所以 RoPE 只改变方向，不改变模长。

### 3.3 为什么只旋转 Q、K

注意力分数是

$$
S_{ij}=\frac{\langle q_i,k_j\rangle}{\sqrt{d_k}}.
$$

位置信息只需要影响“谁和谁匹配”，也就是 $q_i$ 与 $k_j$ 的内积。因此旋转 $Q$ 和 $K$ 就足够了。

$V$ 不参与打分，它是被加权平均的内容。如果也对 $V_j$ 做旋转，输出会变成

$$
o_i=\sum_{j\le i}A_{ij}R_j\tilde v_j,
$$

这会让输出依赖于绝对位置 $j$，而且会改变内容的表示基。RoPE 的设计是**只在打分阶段注入位置，在内容聚合阶段保持原样**。

### 3.4 复数形式与代码映射

二维旋转等价于复数乘法。把一对维度写成

$$
z=q_{2i-1}+i\,q_{2i},
$$

则旋转 $m\theta_i$ 等价于乘以

$$
e^{im\theta_i}=\cos(m\theta_i)+i\sin(m\theta_i).
$$

代码 `src/rope.py` 正是这样做的：

1. 用 `precompute_freqs_cis` 预计算 $e^{im\theta_i}$，形状为 $(N,d_k/2)$；
2. 把 $Q,K$ 的最后一维两两分组，视为复数；
3. 逐位置乘以 $e^{im\theta_i}$；
4. 转回实数表示。

形状始终是

$$
Q,K\in\mathbb{R}^{B\times h\times N\times d_k}
\longrightarrow
\mathbb{R}^{B\times h\times N\times d_k}.
$$

RoPE 不增加任何可学习参数，也不改变张量阶数。

### 3.5 位置编码的维度账本

| 方式 | 作用位置 | 参数形状 | 是否改变 $d$ | 是否依赖绝对位置 |
| :--- | :--- | :--- | :--- | :--- |
| 学习式绝对位置编码 | 输入嵌入相加 | $N_{max}\times d$ | 否 | 是 |
| RoPE | 注意力内部旋转 $Q,K$ | 无参数（预计算频率） | 否 | 否，只依赖相对位置 |

---

## 4. 自注意力：唯一的跨 token 混合算子

### 4.1 投影与多头拆分

设输入

$$
X\in\mathbb{R}^{N\times d}.
$$

完整的 QKV 投影矩阵为

$$
W_Q,W_K,W_V\in\mathbb{R}^{d\times d}.
$$

多头注意力把这 $d$ 个输出通道分成 $h$ 组，每组 $d_k=d/h$ 维：

$$
W_Q=[W_Q^{(1)},\dots,W_Q^{(h)}],
\qquad
W_Q^{(r)}\in\mathbb{R}^{d\times d_k}.
$$

对第 $r$ 个头：

$$
Q^{(r)}=XW_Q^{(r)},\qquad
K^{(r)}=XW_K^{(r)},\qquad
V^{(r)}=XW_V^{(r)},
$$

形状均为

$$
Q^{(r)},K^{(r)},V^{(r)}\in\mathbb{R}^{N\times d_k}.
$$

代码中 `c_attn = nn.Linear(d, 3d)`，等价于把三个矩阵横向拼成

$$
W_{QKV}=[W_Q,W_K,W_V]\in\mathbb{R}^{d\times 3d},
$$

一次矩阵乘法得到 $XW_{QKV}\in\mathbb{R}^{N\times 3d}$，再沿最后一维切成三块。**一次大矩阵乘法比三次小矩阵乘法更高效**，这是工程实现上的常见技巧。

> 注：PyTorch 的 `nn.Linear` 权重按 $(d_{out},d_{in})$ 存储，数学上等价于本文行向量约定下的转置。本文统一用 $XW$ 的行向量写法，阅读代码时只需记住这个转置关系。

### 4.2 缩放点积的方差推导

对单个头，注意力分数为

$$
S^{(r)}
=
\frac{Q^{(r)}K^{(r)\top}}{\sqrt{d_k}}
\in\mathbb{R}^{N\times N}.
$$

为什么除以 $\sqrt{d_k}$？做一个理想化假设：$q,k\in\mathbb{R}^{d_k}$ 的每个分量独立、均值为 0、方差为 1，则

$$
s=\langle q,k\rangle=\sum_{i=1}^{d_k}q_ik_i.
$$

均值为

$$
\mathbb{E}[s]=\sum_{i=1}^{d_k}\mathbb{E}[q_i]\mathbb{E}[k_i]=0.
$$

方差为

$$
\operatorname{Var}(s)
=
\sum_{i=1}^{d_k}\operatorname{Var}(q_ik_i)
=
\sum_{i=1}^{d_k}\mathbb{E}[q_i^2]\mathbb{E}[k_i^2]
=
d_k.
$$

所以

$$
\operatorname{std}(s)=\sqrt{d_k}.
$$

除以 $\sqrt{d_k}$ 后，分数的方差回到 1。如果不缩放，$d_k$ 越大，logits 的数值范围越宽，softmax 越容易饱和：

$$
a_i(1-a_i)\to 0,
$$

梯度消失，训练不稳定。这就是缩放因子的数学理由。

### 4.3 因果掩码

Decoder-only 模型必须保证位置 $i$ 只能看到 $j\le i$。定义掩码矩阵

$$
M_{ij}=
\begin{cases}
0, & j\le i,\\
-\infty, & j>i.
\end{cases}
$$

把掩码加到分数上：

$$
\tilde S^{(r)}=S^{(r)}+M.
$$

因为

$$
\exp(-\infty)=0,
$$

所以 softmax 后

$$
A_{ij}^{(r)}=0,\qquad j>i.
$$

每一行只剩下前缀 $j\le i$ 的权重，且行和为 1：

$$
\sum_{j=1}^{N}A_{ij}^{(r)}=1.
$$

这就是自回归性质的矩阵表达：**注意力权重矩阵是下三角的**。

> 实现中通常用很大的负数（如 $-10^9$）代替 $-\infty$，避免数值溢出或 NaN；数学上仍视为 $-\infty$。本项目的 `F.scaled_dot_product_attention(is_causal=True)` 会在融合算子内部自动处理因果掩码。

### 4.4 softmax 与凸组合

对每一行做 softmax：

$$
A^{(r)}=\operatorname{softmax}_{\text{row}}\left(S^{(r)}+M\right)
\in\mathbb{R}^{N\times N}.
$$

输出为

$$
O^{(r)}=A^{(r)}V^{(r)}
\in\mathbb{R}^{N\times d_k}.
$$

逐行看：

$$
o_i^{(r)}
=
\sum_{j=1}^{N}A_{ij}^{(r)}v_j^{(r)}
=
\sum_{j\le i}A_{ij}^{(r)}v_j^{(r)}.
$$

因为 $A_{ij}\ge 0$ 且 $\sum_j A_{ij}=1$，所以 $o_i$ 是前缀 value 向量的**凸组合（convex combination）**。这给了注意力一个非常直观的解释：

> 注意力就是在做一次数据相关的加权平均；权重由 query 和 key 的相似度决定。

从核方法的角度看，这是 Nadaraya-Watson 核回归（因果情况下求和上限为 $i$）：

$$
o_i=\frac{\sum_j\kappa(q_i,k_j)v_j}{\sum_j\kappa(q_i,k_j)},
\qquad
\kappa(q_i,k_j)=\exp\left(\frac{\langle q_i,k_j\rangle}{\sqrt{d_k}}\right).
$$

softmax 只是把这个核权重归一化。

### 4.5 多头拼接与输出投影

每个头输出 $O^{(r)}\in\mathbb{R}^{N\times d_k}$。把所有头沿特征维拼接：

$$
O=[O^{(1)},O^{(2)},\dots,O^{(h)}]
\in\mathbb{R}^{N\times hd_k}
=
\mathbb{R}^{N\times d}.
$$

然后做输出投影：

$$
Y=OW_O,
\qquad
W_O\in\mathbb{R}^{d\times d},
\qquad
Y\in\mathbb{R}^{N\times d}.
$$

**注意 $W_O$ 的作用**：如果没有 $W_O$，各头只是简单拼接，直到 FFN 才发生交互；$W_O$ 让不同头的信息在 $d$ 维空间里线性混合，学习“哪些头的组合有用”。

### 4.6 排列等变性与位置信息的必要性

先忽略位置编码和因果掩码。设 $P\in\{0,1\}^{N\times N}$ 是置换矩阵，满足 $P^\top P=I$。对输入做置换 $X\mapsto PX$：

$$
S(PX)
=
\frac{(PX)W_QW_K^\top(PX)^\top}{\sqrt{d_k}}
=
P\,S(X)\,P^\top.
$$

逐行 softmax 与行置换可交换，因此

$$
A(PX)=P\,A(X)\,P^\top.
$$

输出为

$$
O(PX)
=
A(PX)(PX)W_V
=
P\,A(X)\,P^\top P\,XW_V
=
P\,O(X).
$$

所以没有位置编码时，注意力对 token 的排列是**等变的（permutation equivariant）**：换一下 token 顺序，输出也只是跟着换顺序。模型把输入当成一个**集合**，而不是序列。

因果掩码打破了完全的置换对称性：它规定了“只能看前缀”，因此“哪些 token 可见”被编码了。但掩码本身不编码**距离**：在固定 query 位置的前提下，前缀内部 token 的相对顺序仍然不可区分。要区分“相邻”和“相距很远”，仍然需要 RoPE 或绝对位置编码。这就是位置编码不可省略的原因。

### 4.7 参数量与计算量

单个头的投影矩阵形状为 $d\times d_k$，三个矩阵共

$$
3dd_k.
$$

$h$ 个头合起来：

$$
3hdd_k=3d^2.
$$

输出投影：

$$
d^2.
$$

所以注意力层参数（bias=False）为

$$
3d^2+d^2=4d^2.
$$

计算量方面，给定 $N$ 个 token，分数矩阵 $S$ 是 $N\times N$，两个矩阵乘法（$QK^\top$ 和 $AV$）的复杂度为

$$
O(N^2d).
$$

这是 Transformer 最核心的瓶颈来源。

---

## 5. 前馈网络：逐 token 的特征混合

### 5.1 GELU

GELU（Gaussian Error Linear Unit）定义为

$$
\operatorname{GELU}(x)=x\,\Phi(x),
$$

其中 $\Phi$ 是标准正态分布的累积分布函数：

$$
\Phi(x)=\frac{1}{2}\left(1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right).
$$

常用近似为

$$
\operatorname{GELU}(x)
\approx
0.5x\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}\left(x+0.044715x^3\right)\right)\right).
$$

它的导数为

$$
\operatorname{GELU}'(x)=\Phi(x)+x\,\phi(x),
$$

其中 $\phi(x)=\frac{1}{\sqrt{2\pi}}e^{-x^2/2}$ 是标准正态密度。GELU 在负值区域不是硬截断，而是平滑地趋近于 0，因此比 ReLU 更平滑。

GELU-FFN 为

$$
F(X)=\operatorname{GELU}(XW_1+b_1)W_2+b_2,
$$

其中

$$
W_1\in\mathbb{R}^{d\times d_{ff}},
\qquad
W_2\in\mathbb{R}^{d_{ff}\times d}.
$$

形状变化：

$$
\mathbb{R}^{N\times d}
\xrightarrow{W_1}
\mathbb{R}^{N\times d_{ff}}
\xrightarrow{\operatorname{GELU}}
\mathbb{R}^{N\times d_{ff}}
\xrightarrow{W_2}
\mathbb{R}^{N\times d}.
$$

参数量（含 bias）：

$$
dd_{ff}+d_{ff}+d_{ff}d+d
=
2dd_{ff}+d_{ff}+d.
$$

### 5.2 SwiGLU

SwiGLU 使用门控（gating）：

$$
F(X)
=
\left(\operatorname{SiLU}(XW_{gate})\odot(XW_{up})\right)W_{down},
$$

其中

$$
W_{gate},W_{up}\in\mathbb{R}^{d\times d_{ff}},
\qquad
W_{down}\in\mathbb{R}^{d_{ff}\times d},
$$

$\odot$ 是逐元素乘法（Hadamard product），

$$
\operatorname{SiLU}(x)=x\,\sigma(x)=\frac{x}{1+e^{-x}}.
$$

形状：

$$
XW_{gate},XW_{up}\in\mathbb{R}^{N\times d_{ff}},
$$

逐元素相乘后仍是 $\mathbb{R}^{N\times d_{ff}}$，最后 $W_{down}$ 把它压回 $\mathbb{R}^{N\times d}$。

SwiGLU 的参数量（bias=False）为

$$
3dd_{ff}.
$$

**为什么是 $d_{ff}\approx\frac{8}{3}d$？**

GELU-FFN 的参数量是

$$
2d\cdot 4d=8d^2.
$$

为了让 SwiGLU 的参数量与之接近：

$$
3dd_{ff}=8d^2
\quad\Longrightarrow\quad
d_{ff}=\frac{8}{3}d.
$$

本项目 $d=768$：

$$
d_{ff}=\frac{8}{3}\times 768=2048,
$$

正好是 256 的倍数，便于 GPU 高效计算。代码中还会把 $d_{ff}$ 向上取整到 256 的倍数。

### 5.3 FFN 作为 key-value memory

把 GELU-FFN 逐分量展开：

$$
h_j=\phi\left(\langle x,k_j\rangle+b_{1,j}\right),
\qquad
k_j=W_1[:,j]\in\mathbb{R}^{d},
$$

$$
y=\sum_{j=1}^{d_{ff}}h_jv_j+b_2,
\qquad
v_j=W_2[j,:]\in\mathbb{R}^{d}.
$$

所以 FFN 可以看成有 $d_{ff}$ 个“记忆槽”的 key-value 存储器：

- 每个 key $k_j\in\mathbb{R}^{d}$ 与输入 $x$ 做内积，决定这个槽被激活多少；
- 每个 value $v_j\in\mathbb{R}^{d}$ 是被写回残差流的内容；
- 激活函数 $\phi$ 控制写入强度。

SwiGLU 只是把标量门控换成了更复杂的门控：

$$
h_j=\operatorname{SiLU}(\langle x,k_j^{gate}\rangle)\cdot\langle x,k_j^{up}\rangle,
$$

但“$d_{ff}$ 个 key-value 槽”的结构不变。这个视角解释了为什么 FFN 的参数量远大于注意力：它承担了模型的大部分“知识存储”。

### 5.4 为什么 FFN 是逐位置的

对任意 token 位置 $t$：

$$
F(X)_{t,:}=F(X_{t,:}).
$$

第 $t$ 行的输出只依赖第 $t$ 行的输入，不同位置之间没有交互。因此 FFN 可以对所有 token 完全并行，且其参数量与 $N$ 无关。

**这是注意力与 FFN 的根本分工：**

| 算子 | 混合的轴 | 是否跨 token | 复杂度 |
| :--- | :--- | :--- | :--- |
| 自注意力 | 序列轴 $N$ | 是 | $O(N^2d)$ |
| FFN | 特征轴 $d$ | 否，逐位置 | $O(Ndd_{ff})$ |
| 归一化 | 特征轴 $d$ | 否，逐位置 | $O(Nd)$ |

---

## 6. 归一化与残差：让深层网络可训练

### 6.1 LayerNorm

对单个 token 的 $d$ 维向量 $x\in\mathbb{R}^{d}$，LayerNorm 计算

$$
\mu=\frac{1}{d}\sum_{j=1}^{d}x_j,
\qquad
\sigma^2=\frac{1}{d}\sum_{j=1}^{d}(x_j-\mu)^2,
$$

$$
\hat x_j=\frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}},
\qquad
y_j=\gamma_j\hat x_j+\beta_j.
$$

形状始终是 $\mathbb{R}^{d}\to\mathbb{R}^{d}$。

**不变性：**

- 平移不变：对任意常数 $c$，

$$
\operatorname{LN}(x+c\mathbf{1})=\operatorname{LN}(x).
$$

因为均值也平移了 $c$，偏差 $x_j-\mu$ 不变。

- 缩放不变：对任意 $\alpha>0$，

$$
\operatorname{LN}(\alpha x)=\operatorname{LN}(x).
$$

因为分子和分母同时缩放 $\alpha$。

LayerNorm 的雅可比矩阵为

$$
J_{\mathrm{LN}}
=
\frac{1}{\sigma}
\left(
I-\frac{1}{d}\mathbf{1}\mathbf{1}^\top
-\frac{1}{d}\hat x\hat x^\top
\right)
\in\mathbb{R}^{d\times d}.
$$

这个矩阵的秩最多是 $d-1$（输入中的平移/均值方向被消掉），解释了为什么 LayerNorm 会去除输入的均值信息。

### 6.2 RMSNorm

RMSNorm 去掉了减均值和偏置：

$$
\operatorname{RMS}(x)=\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\epsilon},
$$

$$
y_j=\gamma_j\frac{x_j}{\operatorname{RMS}(x)}.
$$

它只有 $d$ 个参数 $\gamma$，而 LayerNorm 有 $2d$ 个参数 $(\gamma,\beta)$。

**尺度不变性：** 对 $\alpha>0$，

$$
\operatorname{RMS}(\alpha x)=\alpha\operatorname{RMS}(x),
$$

所以

$$
\operatorname{RMSNorm}(\alpha x)=\operatorname{RMSNorm}(x).
$$

但它**没有平移不变性**，因为减均值会改变 $x_j^2$ 的分布。RMSNorm 的假设是：对深层网络而言，重新居中（re-centering）并不重要，控制尺度（re-scaling）才是关键。LLaMA、Mistral、Gemma 等模型都采用 RMSNorm。

RMSNorm 的雅可比为

$$
J_{\mathrm{RMSNorm}}
=
\frac{1}{\operatorname{RMS}(x)}
\operatorname{diag}(\gamma)
\left(
I-\frac{1}{d}\hat x\hat x^\top
\right),
\qquad
\hat x=\frac{x}{\operatorname{RMS}(x)}.
$$

### 6.3 Pre-Norm 与 Post-Norm 的梯度推导

**Post-Norm（原论文）：**

$$
h_{l+1}
=
\operatorname{Norm}\left(h_l+F_l(h_l)\right).
$$

**Pre-Norm（现代 LLM 主流）：**

$$
h_{l+1}
=
h_l+F_l\left(\operatorname{Norm}(h_l)\right).
$$

把 Pre-Norm 展开：

$$
h_L
=
h_0+\sum_{l=0}^{L-1}F_l\left(\operatorname{Norm}(h_l)\right).
$$

对 $h_0$ 求导：

$$
\frac{\partial h_L}{\partial h_0}
=
\prod_{l=0}^{L-1}\left(I+G_l\right),
\qquad
G_l=\frac{\partial F_l(\operatorname{Norm}(h_l))}{\partial h_l}.
$$

（矩阵不可交换，乘积按 $l=L-1,\dots,0$ 的顺序相乘。）展开乘积：

$$
\prod_{l=0}^{L-1}(I+G_l)
=
I+\sum_{l}G_l+\sum_{l_1<l_2}G_{l_2}G_{l_1}+\cdots.
$$

**恒等项 $I$ 始终存在**。即使所有 $G_l$ 都很小，梯度也可以沿着恒等路径直接传回浅层。这就是残差连接（residual connection）让深层网络可训练的核心原因。

Post-Norm 的梯度是

$$
\frac{\partial h_{l+1}}{\partial h_l}
=
J_{\mathrm{Norm}}\left(I+G_l\right).
$$

每一层都多乘一个归一化雅可比 $J_{\mathrm{Norm}}$。深层连乘时，这些矩阵的特征值可能反复缩小或放大，导致梯度消失或爆炸。Pre-Norm 把归一化移到子层内部，残差路径保持纯净，因此在很深的模型（几十到上百层）中更稳定。

### 6.4 残差流：$d$ 维共享总线

把 $h_l\in\mathbb{R}^{B\times N\times d}$ 想象成一条宽度为 $d$ 的“总线”：

- 注意力从总线上读取信息，在 $N$ 轴上混合后，写回总线；
- FFN 从总线上读取信息，在 $d$ 轴上变换后，写回总线；
- 归一化在写入前调整尺度；
- 每一层都只是在这条总线上做加法。

这解释了为什么所有子层的输出都必须是 $B\times N\times d$：**它们都在往同一条总线上写东西**。任何形状不匹配都会破坏这条总线。

### 6.5 Dropout：训练时的随机掩码

Dropout 不改变形状，只改变数值。训练时对每个元素独立采样

$$
m_i\sim\operatorname{Bernoulli}(1-p),
$$

并做

$$
y_i=\frac{m_i x_i}{1-p}.
$$

期望为

$$
\mathbb{E}[y_i]=\frac{(1-p)x_i}{1-p}=x_i,
$$

所以它是无偏的。缩放因子 $1/(1-p)$ 保证训练和推理的期望一致。

推理时 dropout 退化为恒等映射。这就是为什么推理前必须调用 `model.eval()`：否则随机掩码会让输出不稳定。Transformer 通常在三处使用 dropout：注意力权重、残差分支输出、FFN 输出。

---

## 7. 完整前向：形状账本

### 7.1 公式

输入 token IDs：

$$
I\in\{1,\dots,V\}^{B\times N}.
$$

嵌入：

$$
X^{(0)}=\operatorname{Emb}(I)\in\mathbb{R}^{B\times N\times d}.
$$

如果使用学习式绝对位置编码：

$$
X^{(0)}\leftarrow X^{(0)}+P_{1:N}.
$$

如果是 RoPE，位置编码不在这里加，而是在注意力内部旋转 $Q,K$。

对第 $l$ 层（$l=1,\dots,L$）：

$$
U=\operatorname{Norm}_1\left(H^{(l-1)}\right),
$$

$$
A=H^{(l-1)}+\operatorname{MHA}(U),
$$

$$
H^{(l)}=A+\operatorname{FFN}\left(\operatorname{Norm}_2(A)\right).
$$

所有 $H^{(l)},U,A$ 的形状都是

$$
\mathbb{R}^{B\times N\times d}.
$$

最后一层归一化：

$$
H=\operatorname{Norm}_f\left(H^{(L)}\right)\in\mathbb{R}^{B\times N\times d}.
$$

LM Head：

$$
Z=HW_{LM}\in\mathbb{R}^{B\times N\times V}.
$$

softmax：

$$
p=\operatorname{softmax}(Z)\in\mathbb{R}^{B\times N\times V}.
$$

### 7.2 形状账本

| 阶段 | 张量 | 形状 | 说明 |
| :--- | :--- | :--- | :--- |
| 输入 | token IDs | $B\times N$ | 离散索引 |
| 嵌入 | $X^{(0)}$ | $B\times N\times d$ | 查表 |
| QKV 投影 | $XW_{QKV}$ | $B\times N\times 3d$ | 一次大矩阵乘 |
| 分头 | $Q,K,V$ | $B\times h\times N\times d_k$ | $d=hd_k$ |
| RoPE | $Q,K$ | $B\times h\times N\times d_k$ | 形状不变 |
| 分数 | $S$ | $B\times h\times N\times N$ | 唯一的 $N^2$ 结构（分数/权重） |
| 掩码 | $M$ | $N\times N$ | 广播到 batch/head |
| 权重 | $A$ | $B\times h\times N\times N$ | 行和为 1 |
| 各头输出 | $O^{(1)},\dots,O^{(h)}$ | $B\times h\times N\times d_k$ | 每个头 $d_k$ 维，凸组合 |
| 拼接 | $O$ | $B\times N\times d$ | $h d_k=d$ |
| 输出投影 | $Y$ | $B\times N\times d$ | $W_O$ 混合各头 |
| FFN 隐藏 | $XW_{gate}$ 等 | $B\times N\times d_{ff}$ | 临时升维 |
| FFN 输出 | $F(X)$ | $B\times N\times d$ | 降回 $d$ |
| 最终归一化 | $H$ | $B\times N\times d$ | 形状不变 |
| LM Head | $Z$ | $B\times N\times V$ | 回到词表 |
| softmax | $p$ | $B\times N\times V$ | 概率分布 |

### 7.3 两个“膨胀—收缩”点

整个网络中，只有两个地方会暂时离开 $d$ 维残差流：

1. **多头拆分**：

$$
d\longrightarrow h\times d_k,
\qquad
d_k=\frac{d}{h}.
$$

每个头在低维子空间里独立计算，拼接后回到 $d$。

2. **FFN 升维**：

$$
d\longrightarrow d_{ff}\longrightarrow d.
$$

通常 $d_{ff}\approx 4d$ 或 $\frac{8}{3}d$，是模型参数的主要来源。

其他所有操作都保持 $B\times N\times d$。记住这两个点，就记住了 Transformer 的“形状地图”。

### 7.4 反向传播的阶数对称性

线性层

$$
Y=XW,
\qquad
X\in\mathbb{R}^{N\times d},
\quad
W\in\mathbb{R}^{d\times m},
\quad
Y\in\mathbb{R}^{N\times m}.
$$

损失对输入和权重的梯度为

$$
\frac{\partial L}{\partial X}
=
\frac{\partial L}{\partial Y}W^\top
\in\mathbb{R}^{N\times d},
$$

$$
\frac{\partial L}{\partial W}
=
X^\top\frac{\partial L}{\partial Y}
\in\mathbb{R}^{d\times m}.
$$

**梯度的形状与对应的激活/参数形状完全一致**。这是自动微分能工作的基础，也是检查实现是否正确的一个简单方法。

残差连接

$$
Y=X+F(X)
$$

的梯度为

$$
\frac{\partial L}{\partial X}
=
\frac{\partial L}{\partial Y}
+
J_F^\top\frac{\partial L}{\partial Y}.
$$

两项形状都是 $N\times d$，可以直接相加。

softmax + 交叉熵的梯度为

$$
\frac{\partial L}{\partial Z}
=
\frac{1}{BN}(p-y),
$$

形状 $B\times N\times V$（这里按 batch 内所有 token 平均；若只对一个序列平均，则为 $1/N$）。从 logits 到 token embedding，再一路回传到每一层的权重，所有梯度都保持与对应前向张量相同的阶数。

---

## 8. 参数计数：用形状做一次审计

本节先按当前代码的默认配置（LLaMA 风格：RoPE + RMSNorm + SwiGLU + 权重绑定）计数，再在 8.3 复现原文旧配置（GPT-3 风格：学习式绝对位置 + LayerNorm + GELU + bias）的精确计数。两套数字都能由同一套形状规则推出来。

### 8.1 通用公式

当前架构（bias=False，RMSNorm，SwiGLU，权重绑定）：

$$
P
=
\underbrace{Vd}_{\text{token embedding}}
+
\underbrace{L\left(4d^2+3dd_{ff}+2d\right)}_{\text{per layer}}
+
\underbrace{d}_{\text{final RMSNorm}}.
$$

逐项解释：

- $4d^2$：注意力。QKV 是 $3d^2$，输出投影是 $d^2$。
- $3dd_{ff}$：SwiGLU。gate、up、down 三个矩阵。
- $2d$：两层 RMSNorm 的 $\gamma$。
- $d$：最终 RMSNorm 的 $\gamma$。
- $Vd$：token embedding。LM Head 与它共享，所以不重复计数。

### 8.2 当前配置的精确计数

当前配置：

$$
V=50257,\quad d=768,\quad L=6,\quad d_{ff}=2048.
$$

逐项计算：

$$
Vd=50257\times 768=38{,}597{,}376.
$$

注意力：

$$
4d^2=4\times 768^2=4\times 589{,}824=2{,}359{,}296.
$$

FFN：

$$
3dd_{ff}=3\times 768\times 2048=4{,}718{,}592.
$$

两层 RMSNorm：

$$
2d=2\times 768=1{,}536.
$$

每层合计：

$$
2{,}359{,}296+4{,}718{,}592+1{,}536=7{,}079{,}424.
$$

六层合计：

$$
6\times 7{,}079{,}424=42{,}476{,}544.
$$

最终归一化：

$$
d=768.
$$

总计：

$$
38{,}597{,}376+42{,}476{,}544+768
=
\mathbf{81{,}074{,}688}.
$$

约 $81.1$M，与 README 中的“约 81M 参数”一致。

参数预算：

| 部分 | 参数量 | 占比 |
| :--- | ---: | ---: |
| Token embedding（与 LM Head 绑定） | 38,597,376 | 47.6% |
| 注意力（6 层合计） | 14,155,776 | 17.5% |
| FFN（6 层合计） | 28,311,552 | 34.9% |
| 归一化（6 层 + 最终） | 9,984 | 0.01% |
| **总计** | **81,074,688** | **100%** |

如果解绑 LM Head，总参数变为

$$
81{,}074{,}688+38{,}597{,}376=119{,}672{,}064.
$$

绑定节省了约 $32.3\%$ 的参数。

### 8.3 与旧配置的对照

原文中的 Mini-GPT 使用 $d=512$、GELU-FFN、LayerNorm、学习式绝对位置编码、带 bias、输出层与嵌入层绑定（weight tying）。其通用公式为

$$
P
=
Vd+N_{max}d
+
L\left(4d^2+9d+2dd_{ff}+d_{ff}\right)
+
2d.
$$

逐项来源：

- 注意力：$4(d^2+d)=4d^2+4d$；
- 两层 LayerNorm：$2\times 2d=4d$；
- FFN：$2dd_{ff}+d_{ff}+d$；
- 合并后每层为 $4d^2+9d+2dd_{ff}+d_{ff}$。

代入 $V=50257,d=512,N_{max}=256,L=6,d_{ff}=2048$：

$$
Vd=25{,}731{,}584,
$$

$$
N_{max}d=256\times 512=131{,}072,
$$

$$
4d^2=1{,}048{,}576,\quad
9d=4{,}608,\quad
2dd_{ff}=2{,}097{,}152,\quad
d_{ff}=2{,}048,
$$

每层：

$$
1{,}048{,}576+4{,}608+2{,}097{,}152+2{,}048=3{,}152{,}384.
$$

六层：

$$
6\times 3{,}152{,}384=18{,}914{,}304.
$$

总计：

$$
25{,}731{,}584+131{,}072+18{,}914{,}304+1{,}024
=
\mathbf{44{,}777{,}984}.
$$

与原文的 $44{,}777{,}984$ 完全一致。**同一个形状账本可以复现两套不同配置的精确参数量**，这说明参数计数不是背出来的，而是由矩阵阶数推出来的。

### 8.4 参数预算随规模的变化

| 模型 | $d$ | $L$ | $Vd$ | 每层注意力 $4d^2$ | 每层 FFN $8d^2$ | 总参数 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 本项目 | 768 | 6 | 38.6M | 2.36M | 4.72M | 81.1M |
| GPT-3 | 12288 | 96 | 617M | 604M | 1.208B | ~175B |

在小模型中，**嵌入层占主导**（本项目 47.6%），所以权重绑定是必要的；在大模型中，**FFN 占主导**（GPT-3 约 66%），嵌入层只占 0.35%，所以可以解绑、可以增大输出层容量。

这解释了一个常见的架构演化现象：

> 小模型拼的是“参数效率”，大模型拼的是“参数容量”。

---

## 9. KV-Cache：因果性的直接推论

### 9.1 为什么可以缓存

因果掩码保证

$$
o_i=\sum_{j\le i}A_{ij}v_j.
$$

位置 $i$ 的输出只依赖 $j\le i$ 的 key 和 value。生成第 $t+1$ 个 token 时，前面的 $K_{1:t}$ 和 $V_{1:t}$ 不会改变，因此可以缓存起来，避免重复计算。

注意：**只缓存 $K,V$，不缓存 $Q$**。原因是 query 只在自己的位置使用一次，而 key/value 会被所有未来位置反复读取。

### 9.2 增量解码的矩阵

预填充（prefill）阶段，输入 $N$ 个 token，计算

$$
K_{1:N},V_{1:N}\in\mathbb{R}^{B\times h\times N\times d_k}.
$$

解码第 $t+1$ 个 token 时：

$$
q_{t+1}\in\mathbb{R}^{B\times h\times 1\times d_k},
$$

$$
k_{t+1},v_{t+1}\in\mathbb{R}^{B\times h\times 1\times d_k}.
$$

拼接缓存：

$$
K_{1:t+1}=[K_{1:t};k_{t+1}]
\in\mathbb{R}^{B\times h\times (t+1)\times d_k},
$$

$$
V_{1:t+1}=[V_{1:t};v_{t+1}]
\in\mathbb{R}^{B\times h\times (t+1)\times d_k}.
$$

分数：

$$
S_{t+1}
=
\frac{q_{t+1}K_{1:t+1}^\top}{\sqrt{d_k}}
\in\mathbb{R}^{B\times h\times 1\times (t+1)}.
$$

权重：

$$
A_{t+1}
=
\operatorname{softmax}(S_{t+1})
\in\mathbb{R}^{B\times h\times 1\times (t+1)}.
$$

此时只有一个 query，所有缓存 key 都是过去，因此不需要因果掩码（代码中对应 `is_causal=False`）。输出：

$$
O_{t+1}
=
A_{t+1}V_{1:t+1}
\in\mathbb{R}^{B\times h\times 1\times d_k}.
$$

然后拼接多头、乘 $W_O$、残差、FFN，得到下一个 token 的 logits。

### 9.3 复杂度

| 阶段 | 时间 | 显存 |
| :--- | :--- | :--- |
| Prefill（$N$ 个 token） | $O(N^2d)$ | 分数矩阵 $O(N^2)$（Flash 可降到 $O(N)$） |
| Decode（每 token） | $O(Nd)$（注意力）+ $O(d^2+dd_{ff})$（投影/FFN） | KV cache $O(Nd)$ |
| 生成 $N$ 个 token 总计 | $O(N^2d+Nd^2+Ndd_{ff})$ | KV cache $O(Nd)$ |

KV cache 每层的标量数量为

$$
2Bhtd_k=2Btd.
$$

$L$ 层总共：

$$
2BLtd.
$$

换成字节数，乘以精度（fp16 为 2，fp32 为 4）。

### 9.4 GQA / MQA：减少 KV 头数

标准多头注意力中，$K,V$ 也有 $h$ 个头，cache 大小为 $2BLtd$。

GQA（Grouped-Query Attention）让多个 query 头共享一组 KV 头：

$$
h_{kv}<h.
$$

此时 cache 大小为

$$
2BLt\,h_{kv}d_k
=
2BLtd\cdot\frac{h_{kv}}{h}.
$$

压缩比为

$$
\frac{h}{h_{kv}}.
$$

MQA（Multi-Query Attention）是极端情况 $h_{kv}=1$，所有 query 头共享同一组 $K,V$。这能显著降低长上下文推理的显存带宽压力，但会损失一些表达能力。

### 9.5 数值示例

按单条序列（$B=1$）计算，本项目 $L=6,t=256,d=768$：

$$
2BLtd
=
2\times 1\times 6\times 256\times 768
=
2{,}359{,}296
$$

个标量。fp16 下约

$$
2{,}359{,}296\times 2\text{ bytes}\approx 4.72\text{ MB}.
$$

LLaMA-7B 量级（$B=1,L=32,d=4096,t=4096$）：

$$
2\times 32\times 4096\times 4096
=
1{,}073{,}741{,}824
$$

个标量，fp16 下约 $2.15$GB。

70B 量级（$B=1,L=80,d=8192,t=4096$，按标准 MHA 计算）：

$$
2\times 80\times 4096\times 8192
=
5{,}368{,}709{,}120
$$

个标量，fp16 下约 $10.7$GB。实际 LLaMA-2 70B 使用 GQA，$h_{kv}/h=8/64$，cache 约为 $10.7/8\approx1.34$GB。

这就是为什么长上下文推理中，KV cache 会成为显存的主要占用者。

---

## 10. Flash Attention：不改变数学的 IO 优化

### 10.1 在线 softmax

标准 softmax 需要先求出整行的最大值，再计算指数和。对于长度为 $N$ 的分数向量 $s\in\mathbb{R}^{N}$：

$$
m=\max_j s_j,
\qquad
\ell=\sum_{j=1}^{N}e^{s_j-m},
\qquad
a_j=\frac{e^{s_j-m}}{\ell}.
$$

输出：

$$
o=\sum_{j=1}^{N}a_jv_j.
$$

问题是要把 $s$ 和 $a$ 完整写进显存，复杂度 $O(N^2)$。Flash Attention 用**在线 softmax（online softmax）**分块计算。

维护三个运行量：

$$
m,\qquad \ell,\qquad o.
$$

初始时 $m=-\infty,\ell=0,o=0$。处理一个分数块 $s^{(b)}$ 及其 value 块 $V^{(b)}$：

$$
m'=\max\left(m,\max_{j\in b}s_j\right),
$$

$$
\ell'
=
e^{m-m'}\ell
+
\sum_{j\in b}e^{s_j-m'},
$$

$$
o'
=
\frac{
e^{m-m'}\ell\,o
+
\sum_{j\in b}e^{s_j-m'}v_j
}{
\ell'
}.
$$

处理完所有块后，$o$ 就是精确的 softmax 加权和。

### 10.2 为什么是精确的

在线 softmax 只利用了 softmax 的平移不变性：

$$
\operatorname{softmax}(s+c\mathbf{1})=\operatorname{softmax}(s).
$$

每次更新时，把旧的指数和按 $e^{m-m'}$ 重新缩放，等价于把旧的 $m$ 换成新的 $m'$。最终结果与一次性计算完全相同，只是数值上更稳定、显存上更省。

### 10.3 IO 与显存

标准注意力需要把 $S\in\mathbb{R}^{N\times N}$ 写入 HBM（高带宽显存），再读出来做 softmax，再写回，再读出来乘 $V$。Flash Attention 把 $S$ 分块放在 SRAM（片上高速缓存）中计算，避免 $N\times N$ 矩阵的反复读写。

- 计算量仍然是 $O(N^2d)$，没有减少；
- 显存从 $O(N^2)$ 降到 $O(N)$；
- 反向传播时不存储完整的 $A$，而是重新计算分块，用计算换显存。

**Flash Attention 不改变注意力的数学定义，只改变它的 IO 模式。** 这是理解它的关键。

---

## 11. 瓶颈：$N^2$ 从哪里来，何时真正致命

### 11.1 分数矩阵

每个头、每一层都要产生

$$
S\in\mathbb{R}^{N\times N}.
$$

如果显式存储，batch 为 $B$、头数为 $h$、层数为 $L$ 时，总元素数为

$$
B\,h\,L\,N^2.
$$

例如 $B=32,h=32,N=2048,L=1$：

$$
32\times 32\times 2048^2
\approx
4.29\times 10^9
$$

个元素。fp32 下约 $17$GB，**单层**就放不下。这就是注意力显存瓶颈的来源。

### 11.2 FLOPs 对比：注意力 vs FFN

预填充 $N$ 个 token 时：

- 注意力：两个 $N\times N$ 矩阵乘法，复杂度

$$
4N^2d.
$$

- SwiGLU FFN：三个矩阵乘法，复杂度

$$
6Ndd_{ff}.
$$

比值为

$$
\frac{\text{attention}}{\text{FFN}}
=
\frac{4N^2d}{6Ndd_{ff}}
=
\frac{2N}{3d_{ff}}.
$$

注意力在 FLOPs 上超过 FFN 的临界点是

$$
N>\frac{3}{2}d_{ff}.
$$

对本项目 $d_{ff}=2048$，临界点约为 $3072$，远大于 `block_size=256`。所以在当前上下文长度下，**FFN 才是 FLOPs 的主要来源**，注意力的 $N^2$ 更多是显存和 IO 问题。

对 GPT-3（$d_{ff}=49152$），临界点约为 $73728$。在 2048 的上下文下，FFN 仍然主导 FLOPs。

这提醒我们：**“$O(N^2)$ 是瓶颈”要分清楚是计算、显存还是 IO**。在短上下文训练中，FFN 往往更耗算力；在长上下文推理中，KV cache 和注意力 IO 才是主要矛盾。

### 11.3 长上下文的主要成本

长上下文推理时，每生成一个 token，都要读取全部历史 KV cache：

$$
O(Nd)
$$

的显存带宽。这个成本随 $N$ 线性增长，且无法像矩阵乘法那样充分利用 GPU 的并行度。因此，长上下文的核心优化往往不是减少 FLOPs，而是：

- 减少 KV cache 大小（GQA/MQA、量化）；
- 提高 KV cache 读取效率（PagedAttention、连续内存布局）；
- 减少需要读取的 token 数（稀疏注意力、检索）。

### 11.4 改进路线

| 方向 | 代表方法 | 改变的是什么 | 代价 |
| :--- | :--- | :--- | :--- |
| IO 优化 | Flash Attention | 不改变数学，减少 HBM 读写 | 实现复杂 |
| 推理缓存 | KV Cache | 复用历史 $K,V$ | 显存随 $N$ 增长 |
| 缓存管理 | PagedAttention | 分页管理 KV cache | 系统复杂度 |
| 减少 KV 头 | GQA / MQA | 共享 $K,V$ | 表达能力略降 |
| 稀疏注意力 | Sliding Window / Sparse | 只计算部分 $N^2$ 项 | 可能丢失长程依赖 |
| 线性注意力 | Kernelized Attention | 改变计算顺序 | 改变模型类 |
| 状态空间模型 | Mamba / SSM | 用递推替代注意力 | 生态与训练差异 |

### 11.5 线性注意力的推导

标准注意力：

$$
O=\operatorname{softmax}(QK^\top)V.
$$

如果去掉 softmax，换成特征映射 $\phi$：

$$
O=\phi(Q)\left(\phi(K)^\top V\right).
$$

（严格地说，归一化版本还要除以 $\phi(Q)\left(\phi(K)^\top\mathbf{1}\right)$；这里只保留核心的矩阵结合律。）注意矩阵乘法的结合律：

$$
\left[\phi(Q)\phi(K)^\top\right]V
=
\phi(Q)\left[\phi(K)^\top V\right].
$$

左边先算 $N\times N$，复杂度 $O(N^2d)$；右边先算

$$
\phi(K)^\top V\in\mathbb{R}^{d_\phi\times d_v},
$$

复杂度 $O(Nd_\phi d_v)$，再乘 $\phi(Q)$，复杂度 $O(Nd_\phi d_v)$。总计

$$
O(Nd_\phi d_v),
$$

对 $N$ 是线性的。

代价是：这不再是原来的 softmax 注意力，模型类发生了变化。线性注意力、RWKV、Mamba 等方法都在这个“用结构换效率”的谱系上。

---

## 12. 为什么是 Decoder-only，以及能力边界

### 12.1 架构选择的数学理由

**Encoder-Decoder：**

$$
p(y_{1:M}\mid x_{1:N})
=
\prod_{t=1}^{M}p(y_t\mid y_{<t},x_{1:N}).
$$

解码器有自注意力和交叉注意力。交叉注意力的 query 来自解码器，key/value 来自编码器：

$$
Q=Y W_Q\in\mathbb{R}^{M\times d},
\qquad
K=XW_K,\quad V=XW_V\in\mathbb{R}^{N\times d}.
$$

分数矩阵是

$$
QK^\top\in\mathbb{R}^{M\times N},
$$

**不是方阵**。这适合“输入序列到输出序列”长度不同的任务，但需要两套参数栈。

**Decoder-only：**

$$
p(x_{1:N})=\prod_{t=1}^{N}p(x_t\mid x_{<t}).
$$

没有独立的编码器，没有交叉注意力，只有自注意力：

$$
Q,K,V\in\mathbb{R}^{N\times d},
\qquad
QK^\top\in\mathbb{R}^{N\times N}.
$$

所有参数都在一个栈里。对于固定的参数预算，Decoder-only 可以把全部容量用于一个统一的下一 token 预测目标；任何条件任务都可以写成“给定前缀，续写后缀”。这是它成为通用范式的结构原因。

### 12.2 能力边界

| 边界 | 数学原因 | 表现 |
| :--- | :--- | :--- |
| 长上下文昂贵 | 分数矩阵 $N\times N$，KV cache $O(N)$ | 上下文越长，显存/带宽压力越大 |
| 固定窗口 | `block_size` 截断 | 窗口外的信息完全不可见 |
| 下一 token 目标 | 只优化 $p(x_t\mid x_{<t})$ | 可能拟合表面统计，产生幻觉 |
| 每 token 固定计算量 | 深度 $L$ 固定 | 不能按难度自适应增加计算 |
| 离散词表 | tokenization 的边界 | 子词切分影响推理与算术 |
| 无外部 grounding | 知识只来自参数与上下文 | 事实性受训练数据限制 |

这些边界不是“实现缺陷”，而是架构假设的直接后果。理解它们，才能理解为什么会出现 RAG、工具调用、长上下文扩展、CoT、MoE 等后续工作。

### 12.3 前沿方向

- **高效注意力**：Flash Attention、GQA/MQA、滑动窗口、稀疏注意力、线性注意力、SSM。
- **长上下文**：RoPE 频率缩放（NTK-aware、YaRN）、位置插值、检索增强。
- **稀疏化**：MoE（Mixture of Experts）把 FFN 拆成多个专家，每个 token 只激活少数专家，参数量与计算量解耦。
- **推理时计算**：Chain-of-Thought、self-consistency、工具调用，用更多 token 换取更深的条件计算。
- **后训练**：SFT、RLHF、DPO，把下一 token 预测的分布对齐到人类偏好。

---

## 13. 总结：一张图记住所有阶数

### 13.1 六条不变量

1. **残差流宽度不变**：

$$
H^{(\ell)}\in\mathbb{R}^{B\times N\times d},
\qquad
\forall \ell.
$$

2. **注意力是唯一的跨 token 混合算子**，其核心张量是

$$
S\in\mathbb{R}^{B\times h\times N\times N}.
$$

3. **FFN 是逐 token 的特征混合算子**，它把 $d$ 临时撑到 $d_{ff}$ 再收回。

4. **多头把 $d$ 切成 $h\times d_k$**：

$$
d=h d_k,
\qquad
d_k=\frac{d}{h}.
$$

5. **词表 $V$ 只出现在两端**：token embedding 和 LM Head。

6. **所有参数矩阵的内维都与相邻激活对齐**：

$$
XW,\quad
X\in\mathbb{R}^{N\times d},\quad
W\in\mathbb{R}^{d\times m}
\quad\Longrightarrow\quad
XW\in\mathbb{R}^{N\times m}.
$$

### 13.2 一句话记忆

> 外面是 $B,N$，里面是 $d$；  
> 注意力横着走（token 轴），FFN 竖着走（特征轴）；  
> 头把 $d$ 切开，FFN 把 $d$ 撑开再收回；  
> 词表只在两头，残差流始终是 $d$。

### 13.3 读懂任意 Transformer 变体的检查清单

遇到一个新的架构（LLaMA、Mistral、MoE、Mamba、RWKV……），按顺序问：

1. **残差流宽度是多少？** 哪一步改变了 $d$？
2. **哪里在 $N$ 轴上混合？** 是 full attention、sliding window、还是 recurrence？
3. **$N^2$ 的张量在哪里？** 是否被 Flash/稀疏/线性化消除？
4. **FFN 的 $d_{ff}$ 是多少？** 是 GELU、SwiGLU 还是 MoE？
5. **归一化放在子层前还是后？** 是 LayerNorm 还是 RMSNorm？
6. **位置信息如何注入？** 加性 PE、RoPE、ALiBi，还是根本不需要？
7. **推理时缓存什么？** 缓存 $K,V$ 还是整个状态？cache 的形状是什么？
8. **词表维度 $V$ 在哪里出现？** 是否权重绑定？

能回答这八个问题，基本就抓住了任何 Decoder-only 架构的骨架。

---

## 附录 A：与代码的对应关系

| 数学对象 | 代码位置 | 形状 |
| :--- | :--- | :--- |
| QKV 投影 $W_{QKV}$ | `src/model.py:135` `c_attn` | $d\times 3d$ |
| 输出投影 $W_O$ | `src/model.py:137` `c_proj` | $d\times d$ |
| 分头与 RoPE | `src/model.py:166-182` | $B\times h\times N\times d_k$ |
| Flash Attention | `src/model.py:206` `F.scaled_dot_product_attention` | $B\times h\times N\times d_k$ |
| 多头拼接 | `src/model.py:214` | $B\times N\times d$ |
| SwiGLU | `src/model.py:253-290` | $d\to d_{ff}\to d$ |
| RMSNorm | `src/model.py:75-104` | $d\to d$ |
| Pre-Norm Block | `src/model.py:296-316` | $B\times N\times d$ |
| LM Head / 权重绑定 | `src/model.py:365-366` | $V\times d$ |
| RoPE 频率预计算 | `src/rope.py:11-42` | $(N,d_k/2)$ |
| RoPE 应用 | `src/rope.py:45-85` | $B\times h\times N\times d_k$ |
| KV Cache 拼接 | `src/model.py:185-190` | $B\times h\times t\times d_k$ |

---

## 附录 B：关键公式速查

**自回归分解**

$$
p_\theta(x_{1:N})=\prod_{t=1}^{N}p_\theta(x_t\mid x_{<t}).
$$

**交叉熵梯度**

$$
\nabla_z\ell=\operatorname{softmax}(z)-y.
$$

**缩放点积注意力**

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

**多头注意力**

$$
\operatorname{MHA}(X)
=
\operatorname{Concat}(O^{(1)},\dots,O^{(h)})W_O.
$$

**RoPE 相对性**

$$
\langle R_m\tilde q,R_n\tilde k\rangle
=
\tilde q^\top R_{n-m}\tilde k.
$$

**SwiGLU**

$$
\operatorname{FFN}(X)
=
\left(\operatorname{SiLU}(XW_{gate})\odot XW_{up}\right)W_{down}.
$$

**RMSNorm**

$$
\operatorname{RMSNorm}(x)
=
\gamma\odot\frac{x}{\sqrt{\frac{1}{d}\sum_j x_j^2+\epsilon}}.
$$

**Pre-Norm Block**

$$
H^{(l)}
=
A+\operatorname{FFN}\left(\operatorname{Norm}_2(A)\right),
\qquad
A=H^{(l-1)}+\operatorname{MHA}\left(\operatorname{Norm}_1(H^{(l-1)})\right).
$$

**KV cache 大小（$L$ 层）**

$$
2BLtd.
$$

**预填充注意力复杂度**

$$
O(N^2d).
$$

**注意力与 FFN 的 FLOPs 比**

$$
\frac{4N^2d}{6Ndd_{ff}}
=
\frac{2N}{3d_{ff}}.
$$

---

## 参考资料

- Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
- Brown et al. *Language Models are Few-Shot Learners.* NeurIPS 2020.
- Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* 2021.
- Zhang & Sennrich. *Root Mean Square Layer Normalization.* NeurIPS 2019.
- Shazeer. *GLU Variants Improve Transformer.* 2020.
- Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.
- Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019.
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023.
- Katharopoulos et al. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML 2020.
- Touvron et al. *LLaMA: Open and Efficient Foundation Language Models.* 2023.
- Geva et al. *Transformer Feed-Forward Layers Are Key-Value Memories.* EMNLP 2021.
- Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022.
