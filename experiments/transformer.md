# An Introduction to LLMs

**Decoder-only transformer 架构原理：**

- 大语言模型是如何工作的
- Decoder-only 架构为什么会成为当前的通用范式，它的瓶颈又是什么
- 大模型的能力边界在哪里
- 大模型领域的前沿探索

## 楔子

2017年，Google 团队发表 *Attention is All You Need*，给出了注意力计算公式，提出了纯粹基于 self-attention 机制的 transformer 架构，可以实现极为高效的并行计算。

> Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

2020年，OpenAI 团队发表 *Language Models are Few-shot Leaners*。此前 OpenAI 研究团队就已经确定了 decoder-only 的生成式预训练方案。这篇文章里，虽然 GPT-3 在模型架构设计上依然相对保守，但他们在参数规模上进行极大程度的扩展（175B），最终验证了语言模型的 scaling law，提出了 in-context learning 这种全新范式。

> Brown, Tom, et al. "Language models are few-shot learners." Advances in neural information processing systems 33 (2020): 1877-1901.

## Mini-GPT：一个本地项目

在算力资源极为有限的条件下，最大限度遵照 GPT-3 的模型架构去复刻一个微型的 GPT 项目。

**参数规模区别：**

| 模型 | Mini-GPT | GPT-3 |
| :--- | :--- | :--- |
| token 嵌入维度 | 512 | 12288 |
| 注意力层数 | 6 | 96 |
| 最大上下文长度 | 256 | 2048 |

**其他区别：**

- 位置编码方式：参考 GPT-2 将位置嵌入当作 token 嵌入的偏置，没有引入RoPE；
- 模型输出层：由于参数量过小，输出层和嵌入层进行权重绑定，以防止模型完全失去泛化效果。但因为 GPT-3 的参数规模极大，所以输出层是一个独立的矩阵，这样可以增大模型容量，模型学习能力进一步增强。

**数据流动的全过程：**

1. 输入：\( X \in \mathbb{R}^{B \times N \times d} \), $d = d_{model}$
   （包含 token embedding + positional encoding）

2. 多头自注意力投影
   \( Q = X W^Q, \ K = X W^K, \ V = X W^V \)  
   其中 \( W^Q, W^K, W^V \in \mathbb{R}^{d \times (h d_k)}, \ W^O \in \mathbb{R}^{(h d_k) \times d} \), \( d_k = d / h \)

3. reshape + transpose：
   \( Q, K, V \to \mathbb{R}^{B \times h \times N \times d_k} \),  
   \( K^\top \to \mathbb{R}^{B \times h \times d_k \times N} \)

4. LayerNorm 归一化（Pre-Attention）
   \( X = \text{LayerNorm}(X) \)

5. 注意力得分
   \( \text{scores} = \frac{Q K^\top}{\sqrt{d_k}} \in \mathbb{R}^{B \times h \times N \times N} \)

6. Causal masking
   \( \text{mask} \in \mathbb{R}^{N \times N} \): 上三角（不含对角线）为 \(-\infty\)，其余为 0  
   \( \text{scores} = \text{scores} + \text{mask} \)（广播到 B×h）

7. 注意力权重  
   \( A = \text{softmax}(\text{scores}, \dim=-1) \in \mathbb{R}^{B \times h \times N \times N} \)

8. 获取每个头的输出并拼接
   \( \text{head}_i = A_i V_i \in \mathbb{R}^{B \times N \times d_k} \)  
   \( \text{MultiHeadOut} = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O \in \mathbb{R}^{B \times N \times d} \)

9. 残差连接 + LayerNorm 归一化
   \( X_{\text{att}} = \text{LayerNorm}(X + \text{MultiHeadOut})\in \mathbb{R}^{B \times N \times d} \)

10. Position-wise Feed-Forward Network（两层 MLP）  
   \( \text{FFN}(X_{\text{att}}) = (\text{GELU}(X_{\text{att}}W_1 + b_1))W_2 + b_2 \)  
   其中 \( W_1 \in \mathbb{R}^{d \times d_{ff}}, \ W_2 \in \mathbb{R}^{d_{ff} \times d} \)

11. 多层堆叠  
    \( H^{(0)} = X,~ H^{(\ell)} = \text{TransformerBlock}(H^{(\ell-1)}), ~\ell = 1\dots h \)

12. LM Head 输出
    \( \text{logits} = (\text{LayerNorm}(H^{(L)})) W^{\text{LM}} + b^{\text{LM}} \)  
    其中 \( W^{\text{LM}} \in \mathbb{R}^{d \times |V|} \)

13. 预测（训练时用交叉熵，推理时取最后一位）
    \( P(x_{t+1} | x_{<t+1}) = \text{softmax}(\text{logits}_{t}) \)

## LLM 的瓶颈

注意力公式：

\[
\text{Attention}(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

其中，$Q, K, V\in \mathbb{R}^{N\times d}$
$\Rightarrow QK^T\in \mathbb{R}^{N\times N}$, which is $O(N^2)$
$\Rightarrow \text{softmax}(QK^T)=S$, which is $O(N^2)$
$\Rightarrow SV$, which is $O(N^2)$

**开销：**

- 计算量（较大）
- 显存占用（极大）：训练过程中 attention score 的存储
- IO 消耗（极大）：softmax 运算过程中的读写

**改进：**

- 推理：KV-cache, paged attention
- 训练：flash attention
- 算法：sparse attention, linear attention

## Mini-GPT 参数统计

Mini-GPT 模型的参数量是 44.78M，由于我们已经深入了解过模型的基础架构，现在完全可以计算出一个精确到个位数的准确结果：

> 模型嵌入层（token嵌入+位置嵌入）：$50257 \times 512 + 256 \times 512 = 25,862,656$
> 归一化层（LayerNorm1 权重+偏置）：$512+512=1024$
> 注意力层（3个投影矩阵+1个注意力输出矩阵）：$4\times (512\times 512+512) = 1,050,624$
> 归一化层（LayerNorm2 权重+偏置）：$512+512=1024$
> 前馈神经网络（升维矩阵 + 降维矩阵）：$((4\times 512)\times 512 + 4\times 512) + (512\times(4\times 512) + 512) = 2,099,712$
> 全部注意力参数（6个模块总参数）：$6\times(1024+1,050,624+1024+2,099,712) = 18,914,304$
> 归一化层（Fianl LayerNorm 权重+偏置）：$512+512=1024$
> 模型输出层：和嵌入层共用同一套参数 0

总计：$25,862,656 + 18,914,304 + 1024 = \mathbf{44,777,984}$
