# 推荐系统多任务学习进阶：ESM2 多路径全空间建模深度解析

## 一、 业务痛点：从“单线漏斗”到“复杂行为网络”

早期的 ESMM 模型成功解决了 `曝光 -> 点击 -> 转化` 这一两阶段标准漏斗的样本选择偏差（SSB）问题。
但在真实的电商工业场景中，用户点击后到最终购买前，往往会发生大量复杂的中间行为，形成了一张多路径的网络（如：`曝光 -> 点击 -> 加购 -> 购买`，或 `曝光 -> 点击 -> 收藏 -> 购买` 等）。

<div align="center">
  <img src="https://datawhalechina.github.io/fun-rec/_images/esm2_seq.png" width="400" />
</div>

**传统 ESMM 的局限**：直接跨过这些中间高频行为去预估购买，浪费了大量极其宝贵的用户意图信号。

---

## 二、 破局第一步：行为抽象与概率树构建

<div align="center">
  <img src="https://datawhalechina.github.io/fun-rec/_images/esm2_seq_2.png" width="400" />
</div>

为了对复杂的行为网络进行数学建模，ESM2 首先对用户的点击后行为进行了极简抽象分类：
* **DAction (决定性行为, Deterministic Action)**：离转化最近的核心动作（如：加购、收藏）。
* **OAction (其他行为, Other Action)**：除了 DAction 和购买外的普通动作（如：浏览详情、看评论）。用户点击后，要么是 DAction，要么是 OAction，两者互斥。

基于此，ESM2 构建了一棵“概率树”，并设立了 4 个独立的 Tower 来分别预测 4 个基础条件概率：
1. **$y_1 = P(\text{点击} | \text{曝光})$**：经典的 CTR（点击率）。
2. **$y_2 = P(\text{DAction} | \text{点击})$**：点击后，发生加购/收藏的概率。
3. **$y_3 = P(\text{购买} | \text{DAction})$**：加购/收藏后，最终付款的概率。
4. **$y_4 = P(\text{购买} | \text{OAction})$**：没加购/收藏（只是随便看看），但最终付款的概率（冲动消费）。

<div align="center">
  <img src="https://datawhalechina.github.io/fun-rec/_images/esm2.png" width="400" />
</div>

ESM2模型 (Wen et al., 2020) 有四个塔，分别用来预测上述的$y_1$ $y_2$ $y_3$和 $y_4$，对于这四个塔的输出并不是算4个Loss，而是分别计算曝光->点击、曝光->决定行为和曝光->购买这三个Loss。可以很明显的看出，这三个Loss都是在曝光空间上计算的，和ESMM在曝光空间优化CVR有着异曲同工之处。

## 三、 核心引擎：三大全空间损失函数 (Loss) 深度拆解

ESM2 的精髓在于：**无论预测哪个中间节点的概率，算 Loss 时全都拉到宏观的“曝光空间”下，使用标准的二元交叉熵 (BCELoss) 进行联合优化。** 彻底规避了训练/测试空间断裂的 SSB 问题。

标准 $\text{BCELoss}$ 展开式为：$\text{BCELoss}(y, p) = - [ y \log(p) + (1 - y) \log(1 - p) ]$

### 1. $L_{ctr}$：点击率预估损失
* **业务含义**：在全量曝光样本中，校验用户是否点击。
* **数学展开**：
  $$L_{ctr} = -\frac{1}{N} \sum_{i=1}^N \left[ y_{isClick}^i \log(y_1^i) + (1 - y_{isClick}^i) \log(1 - y_1^i) \right]$$

### 2. $L_{ctavr}$：点击且发生决定性行为 (DAction) 损失
* **业务含义**：在全量曝光样本中，校验用户最终是否加购/收藏。
* **物理机制**：要加购，必须“先点击 ($y_1$) 且点击后加购 ($y_2$)”，预测联合概率为 $y_1 \cdot y_2$。$y_2$ 塔通过这个乘法节点，在全量曝光数据下获得了梯度更新。
* **数学展开**：
  $$L_{ctavr} = -\frac{1}{N} \sum_{i=1}^N \left[ y_{isDAction}^i \log(y_1^i \cdot y_2^i) + (1 - y_{isDAction}^i) \log(1 - y_1^i \cdot y_2^i) \right]$$

### 3. ⭐️ $L_{ctcvr}$：最终转化损失 (多路径融合的艺术)
* **业务含义**：在全量曝光样本中，校验这单最终是否成交。
* **物理机制**：用户从曝光到购买有两条“通关路径”：
  * 路径A (按部就班)：`曝光 -> 点击 -> DAction -> 购买`，概率为 $y_1 \cdot y_2 \cdot y_3$。
  * 路径B (冲动消费)：`曝光 -> 点击 -> OAction -> 购买`，概率为 $y_1 \cdot (1 - y_2) \cdot y_4$。
  最终转化的总概率是这两条路径概率的加和：$pCTCVR = y_1(y_2 \cdot y_3 + (1 - y_2) y_4)$。
* **数学展开**：
  $$L_{ctcvr} = -\frac{1}{N} \sum_{i=1}^N \left[ y_{isPurchase}^i \log\Big(pCTCVR_i\Big) + (1 - y_{isPurchase}^i) \log\Big(1 - pCTCVR_i\Big) \right]$$
  *(将上述 $pCTCVR$ 整体代入公式)*
* **反向传播直觉**：如果用户没加购但买单了，$(1-y_2)$ 接近 1，损失函数的梯度会顺着数学公式自动流向 $y_4$ 塔，逼迫模型调高 $y_4$（无 DAction 下的转化率）的预测值。

---

## 四、 最终目标函数与工程价值

最终的 Loss 是上述三个全空间 Loss 的加权融合：
$$L_{final} = w_{ctr} \cdot L_{ctr} + w_{ctavr} \cdot L_{ctavr} + w_{ctcvr} \cdot L_{ctcvr}$$

**工程与业务价值：**
1. **消灭 SSB (样本选择偏差)**：所有任务的 Loss 都在宏观的曝光空间下计算，所见即所得。
2. **缓解数据稀疏性**：底层的 Shared Embedding 和浅层特征提取网络，能够利用海量的点击 ($y_1$) 和加购 ($y_2$) 数据进行充分的参数更新，从而极大地帮助了处于链路极深处、样本极其稀疏的“购买”任务 ($y_3, y_4$)。
3. **架构通用性**：ESM2 证明了“概率拆解 + 全空间组装”的方法论可以无限扩展。无论未来业务线增加多少种交互行为（如：分享、评论、打赏），只需扩展概率树的分支并合并最终的联合概率即可。