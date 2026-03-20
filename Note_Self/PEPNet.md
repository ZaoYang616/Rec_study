# 推荐系统多场景多任务进阶：PEPNet 动态权重建模深度解析

## 一、 业务痛点：双重跷跷板效应 (Double Seesaw)

在复杂的工业级推荐系统（如快手短视频）中，模型常常面临两个维度的剧烈冲突：
1. **场景跷跷板 (Domain Seesaw)**：例如“单列上下滑”与“双列点选”场景的数据分布差异极大，混合训练时底层 Embedding 难以兼顾，表征无法对齐。
2. **任务跷跷板 (Task Seesaw)**：同一场景下需同时预测“点赞”、“关注”、“完播”等相互依赖又可能互斥的多个目标，导致目标相互抑制。

**破局思路（动态权重建模）**：不再像 MMoE/STAR 那样构建物理隔离的参数网络，而是让模型核心参数共享，通过动态生成与“场景/用户”高度相关的**权重系数（调音台旋钮）**，来动态缩放（Modulate）共享参数的激活值。

<div align="center">
  <img src="https://datawhalechina.github.io/fun-rec/_images/pepnet.png" width="600
  " />
</div>

---

## 二、 核心底层引擎：Gate NU (动态调音台)

<div align="center">
  <img src="https://datawhalechina.github.io/fun-rec/_images/pepnet.png" width="600
  " />
</div>

PEPNet 所有的个性化能力，都建立在一个极其轻量级的门控单元 **Gate NU (Gate Network Unit)** 之上。它的作用是根据输入的先验环境信息（如场景 ID、用户画像），输出一个特征缩放系数 $\delta$。

**公式表达 (3.5.9)：**
$$\mathbf{x}' = \text{ReLU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)$$
$$\delta = \gamma \cdot \text{Sigmoid}(\mathbf{x}'\mathbf{W}_2 + \mathbf{b}_2) \in [0, \gamma]$$

* **$\mathbf{x}$**：先验特征（即“调节依据”，告诉门控当前处于什么环境）。
* $\mathbf{x}'$：通过第一层全连接层 (ReLU) 降维提取特征。
* $\text{Sigmoid}(\dots)$：通过第二层全连接层，输出一个介于 0 到 1 之间的概率值。
* **$\gamma$ (缩放强度)**：论文经验值设为 **2**。这是极其绝妙的工程设计：因为 Sigmoid 输出在 0~1 之间，如果直接相乘只能“缩小”特征；乘上 2 之后，输出 $\delta$ 的范围变为 `[0, 2]`。
  * 当 $\delta = 1$ 时：特征保持原样。
  * 当 $\delta < 1$ 时：特征被抑制。
  * 当 $\delta > 1$ 时：特征被放大增强。

### GateNU实现代码

``` python
class GateNU(tf.keras.layers.Layer):
    """
    两层门控网络（NU）：用于为不同分支/专家动态生成权重或系数。
    结构：Dense(ReLU) -> Dense(Sigmoid) -> gamma 缩放。
    """
    def __init__(self,
                 hidden_units,
                 gamma=2.,
                 l2_reg=0.):
        assert len(hidden_units) == 2
        self.gamma = gamma
        self.dense_layers = [
            tf.keras.layers.Dense(hidden_units[0], activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),  # 第一层：非线性特征提取
            tf.keras.layers.Dense(hidden_units[1], activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(l2_reg)) # 第二层：输出 (0,1) 门值
        ]
        super(GateNU, self).__init__()

    def call(self, inputs):
        output = self.dense_layers[0](inputs)  # [B, hidden_units[0]]
        # 乘以 gamma 对 Sigmoid 输出进行缩放。
        output = self.gamma * self.dense_layers[1](output)  # [B, hidden_units[1]]
        return output
```

---

## 三、 底层个性化：EPNet (场景感知的 Embedding)

<div align="center">
  <img src="https://datawhalechina.github.io/fun-rec/_images/pepnet.png" width="600
  " />
</div>

EPNet 的目标是解决“场景跷跷板”。它在底层共享 Embedding 的基础上，通过 Gate NU 为**当前样本的场景**生成一个维度相同的系数向量，对 Embedding 进行元素级缩放。

**1. 场景个性化门控计算 (公式 3.5.10)：**
$$\delta_{domain} = \mathcal{U}_{ep}(E(\mathcal{F}_d) \oplus (\oslash(E)))$$
* **$E(\mathcal{F}_d)$**：场景先验特征的 Embedding（例如“双列场景”的向量表示）。
* **$E$**：全盘共享的底层原始特征 Embedding（如“商品ID”）。
* **$\oslash$ (Stop Gradient)**：**梯度阻断**！极其关键的保护机制，强制切断门控网络误差向共享 Embedding 的反向传播，防止特征被门控网络污染。
* **$\oplus$ (Concat)**：将场景特征与共享特征拼接，送入 $\mathcal{U}_{ep}$ (即 Gate NU 模块)。
* $\mathcal{U}_{ep}$：就是刚刚讲的 Gate NU 调音台！
* **$\delta_{domain}$**：输出**一个**与共享 Embedding 维度完全一致的缩放系数向量。

**2. 场景个性化 Embedding 生成 (公式 3.5.11)：**
$$O_{ep} = \delta_{domain} \otimes E$$
* **$\otimes$ (Element-wise Product)**：元素级相乘。
* **物理机制**：EPNet **并不会**把所有场景的 Embedding 都算出来拼接，而是“来什么场景的样本，就动态生成一套针对该场景的放大镜（权重向量），去修剪这一个样本的共享 Embedding”，实现极低参数量下的场景个性化。
* 输出 $O_{ep}$：已经被当前场景“开过光”的个性化 Embedding。

### EPNet实现代码
```python
class EPNet(tf.keras.layers.Layer):
    def __init__(self,
                 l2_reg=0.,
                 **kwargs):
        self.l2_reg = l2_reg
        self.gate_nu = None
        super(EPNet, self).__init__( **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        shape1, shape2 = input_shape
        self.gate_nu = GateNU(hidden_units=[shape2[-1], shape2[-1]], l2_reg=self.l2_reg)

    def call(self, inputs, *args, **kwargs):
        domain, emb = inputs
        # stop_gradient 阻断系数支路对 emb 的反向梯度，避免过度耦合。
        return self.gate_nu(tf.concat([domain, tf.stop_gradient(emb)], axis=-1)) * emb  # 输出形状 [B, D_emb]
```
---

## 四、 顶层个性化：PPNet (用户与任务感知的 DNN 参数)

<div align="center">
  <img src="https://datawhalechina.github.io/fun-rec/_images/pepnet.png" width="600
  " />
</div>

经过 EPNet，特征已经具备了“场景感知”能力。流向顶层多任务塔时，PPNet 负责解决“任务跷跷板”问题。它根据当前用户的特性，对每一层 DNN 塔的神经元进行动态缩放。

**1. 提取用户先验与任务门控 (公式 3.5.12)：**
$$O_{prior} = E(\mathcal{F}_u) \oplus E(\mathcal{F}_i) \oplus E(\mathcal{F}_a)$$
$$\delta_{task} = \mathcal{U}_{pp}(O_{prior} \oplus (\oslash(O_{ep})))$$
* $O_{prior}$ 汇聚了用户、Item 和作者的先验特征（Persona）。
* 将其与底层传上来的特征 $O_{ep}$（同样加了梯度阻断 $\oslash$）拼接，送入 PPNet 的 Gate NU，输出针对当前任务的缩放系数 $\delta_{task}$。

**2. 逐层调节 DNN 神经元 (公式 3.5.13)：**
$$\mathbf{O}_{pp}^{(l)} = \delta_{task}^{(l)} \otimes \mathbf{H}^{(l)}$$
$$\mathbf{H}^{(l+1)} = f(\mathbf{O}_{pp}^{(l)}\mathbf{W}^{(l)} + \mathbf{b}^{(l)})$$
* $\mathbf{H}^{(l)}$：第 $l$ 层 DNN 的原始输入。
* **核心动作**：在进行下一层的矩阵乘法前，先拿 $\delta_{task}^{(l)}$ 与本层输入做**元素级相乘**。
* **物理机制**：相当于根据不同的用户偏好，给任务塔里每个神经元“动态调薪”。复用同一套物理参数 $\mathbf{W}^{(l)}$，却通过改变流入数据的激活强度，实现了千人千面的计算逻辑。
* **大白话：** 就相当于根据当前是张三还是李四，在给任务塔里的每一个神经元动态发工资。如果张三是个“重度点赞狂”，$\delta_{task}$ 就会把这一层负责捕捉“点赞意图”的神经元激活值强行放大；如果是李四，就缩小。
* **结果：** 虽然所有任务依然在复用这一套物理参数 $\mathbf{W}^{(l)}$，但由于流入的特征被动态“整容”了，实际上网络产生了个性化的计算逻辑。

### PPNet实现代码
```python
class PPNet(tf.keras.layers.Layer):
    # 核心：用 persona 生成逐层、逐塔的门控系数，对输出按维度缩放
    def __init__(self,
                 multiples,
                 hidden_units,
                 activation,
                 dropout=0.,
                 l2_reg=0.,
                 **kwargs):
        self.hidden_units = hidden_units
        self.l2_reg = l2_reg
        self.multiples = multiples
        # 每个塔一份同结构的层
        self.dense_layers = [
            [tf.keras.layers.Dense(u, activation=activation,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
             for u in hidden_units]
            for _ in range(multiples)
        ]
        self.dropout_layers = [
            [tf.keras.layers.Dropout(dropout) for _ in hidden_units]
            for _ in range(multiples)
        ]
        self.gate_nu = []
        super(PPNet, self).__init__( **kwargs)

    def build(self, input_shape):
        # 为每层生成 gate：输出维度 units*multiples，后续按塔切片
        self.gate_nu = [
            GateNU([units * self.multiples, units * self.multiples], l2_reg=self.l2_reg)
            for units in self.hidden_units
        ]

    def call(self, inputs, training=None, **kwargs):
        inputs, persona = inputs

        # 先计算各层 gate（persona ⊕ stop_gradient(inputs)）
        gate_list = []
        concat_in = tf.concat([persona, tf.stop_gradient(inputs)], axis=-1)
        for i, gate in enumerate(self.gate_nu):
            g = gate(concat_in)                     # [B, units*multiples]
            g = tf.split(g, self.multiples, axis=1) # 每塔 [B, units]
            gate_list.append(g)

        # 按塔前向：逐层 Dense 后用 gate 做逐维调制
        outputs = []
        for n in range(self.multiples):
            x = inputs
            for i in range(len(self.hidden_units)):
                x = gate_list[i][n] * self.dense_layers[n][i](x)
                x = self.dropout_layers[n][i](x, training=training)
            outputs.append(x)
        return outputs
```

---

## 五、 深度拷问与工程实践 (Q&A)

**Q1: PPNet 的 Gate NU 是怎么给不同任务分配 $\delta_{task}$ 的？**
> **工程提效**：在实际代码中（如 TensorFlow 的实现），模型并不是循环调用 $K$ 次 Gate NU。而是让 Gate NU **一次性**输出一个超级长的权重向量（长度为 `单塔维度 × 任务数`），随后使用 `tf.split` 将其切片，精准分发给不同的任务塔，最大化利用 GPU 的并行矩阵运算能力。

**Q2: 为什么 PPNet 必须在每一层 DNN 网络都加 Gate 进行调节？只在底层加不行吗？**
> **深度抽象机制**：深度神经网络是逐层进行特征抽象的（底层提取具象特征如类别，高层提取抽象特征如深层意图）。逐层加入 Gate NU 就像一个**“多频段音频均衡器 (EQ)”**。它赋予了模型在每一个抽象级别上独立调节信号强弱的权力。针对不同的任务（如点赞 vs 完播），模型可以在浅层放大视觉特征，在深层放大社交意图特征，从而达到极致的个性化效果。