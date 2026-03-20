# 推荐系统多任务学习进阶：PLE 模型与 CGC 架构深度解析

## 一、 痛点分析：MMoE 的根本性局限（为什么需要 PLE？）

MMoE 虽然通过“专属门控”缓解了任务冲突，但其架构存在一个根本性局限：**所有专家对所有任务门控均可见（软隔离）**。这种设计在实践中面临两大挑战：

1. **负迁移未彻底根除（干扰路径未切断）**：
   * 即使某个专家（如 $e_m$）被任务 $i$ 高度加权而被任务 $j$ 忽略，由于 $e_m$ 依然是任务 $j$ 门控的可选项，任务 $j$ 的梯度在反向传播时仍会流经 $e_m$。当任务冲突强烈时，这种“潜在通路”仍可能导致共享表征被污染。
   * 专家角色模糊，一个专家可能同时承载共享信息和多个任务的特定信息，成为冲突的“重灾区”。
2. **门控决策负重**：
   * 门控需要在所有 $K$ 个专家中分配权重，当专家数量庞大时，门控网络面临高维决策问题，易导致训练不稳定或陷入次优解。
   * 门控需要“费力”地从包含混杂信息的专家池中筛选有用信息，增加了学习难度。

---

## 一、破局核心—— CGC 单元的数学解剖：物理切断干扰路径
<div align="center">
  <img src="https://datawhalechina.github.io/fun-rec/_images/cgc.png" width="400" />
</div>

在 CGC 中，专家（Experts）被进行了严格的“职责强制分离”：
1. **共享专家 (C-Experts)**：负责提取所有任务的共性。设数量为 $M$，输出向量记为 $\mathbf{c}_1, \mathbf{c}_2, \dots, \mathbf{c}_M$。
2. **任务专属专家 (T-Experts)**：负责提取特定任务的特性。设任务 $t$ 有 $N_t$ 个专属专家，输出向量记为 $\mathbf{t}_t^1, \mathbf{t}_t^2, \dots, \mathbf{t}_t^{N_t}$。

### 1. 门控打分机制 (公式 3.4.3)
任务 $t$ 的专属门控网络根据输入特征 $\mathbf{x}$，计算其对各个可用专家的注意力权重：
$$g_t(\mathbf{x}) = \text{softmax}(\mathbf{W}_t \cdot \mathbf{x} + \mathbf{b}_t)$$
* $\mathbf{W}_t, \mathbf{b}_t$ 为任务 $t$ 门控的专属参数。
* $g_t(\mathbf{x})$ 输出一个概率分布向量。

### 2. ⭐️ 硬隔离加权融合 (公式 3.4.4) 
任务 $t$ 拿到门控权重后，进行特征融合得到 $\mathbf{h}_t$：
$$\mathbf{h}_t = \sum_{k=1}^M g_{t,k} \cdot \mathbf{c}_k + \sum_{j=1}^{N_t} g_{t,M+j} \cdot \mathbf{t}_t^j$$

**【核心物理意义解读】**：
* 公式左半部分是对 **$M$ 个共享专家**的加权求和；右半部分是对 **$N_t$ 个本任务专属专家**的加权求和。
* **隔离的本质**：仔细观察求和的边界，该融合公式**严格排除了其他任务 $s$ ($s \neq t$) 的专属专家**。在反向传播计算梯度时，任务 $t$ 的 Loss 绝对无法影响到其他任务的专属专家参数。这种数学层面边界的框定，在物理链路上彻底切断了多任务间的相互干扰（负迁移）。

---

## 二、 从 CGC 到 PLE：数学公式的纵向堆叠
<div align="center">
  <img src="https://datawhalechina.github.io/fun-rec/_images/ple.png" width="400" />
</div>

单层 CGC 提取高阶非线性特征的能力有限。PLE 模型借鉴了深层网络逐层抽象的思想，将多个 CGC 单元像千层饼一样纵向堆叠，实现**渐进式知识提取**。

### 1. 输入层 (第 $l=1$ 层)
* **输入**：最底层的原始特征 $\mathbf{x}$。
* **输出**：经过第一层 CGC 模块，每个任务获得初步的融合表征 $\mathbf{h}_t^{(1)}$。同时，收集该层所有专家（共享+专属）的输出，拼接作为下一层的输入。

### 2. 中间层 (第 $l \ge 2$ 层)
* **输入**：第 $l-1$ 层所有专家输出的拼接体。设上一层总专家数为 $E^{(l-1)}$，则输入维度扩大。
* **处理**：在这个包含上一层丰富特征的输入上，构建一套全新的 CGC 模块（新的专家组与新门控 $g_t^{(l)}$），再次进行显式的知识分离与融合。

### 3. 输出层 (第 $L$ 层)
* 在最顶层，直接拿到各任务经历 $L$ 次提纯后的专属融合向量 $\mathbf{h}_t^{(L)}$。
* **最终预测 (公式 3.4.5)**：将其送入各自的任务专属塔 $f_t$，得出预测结果：
$$\hat{y}_t = f_t(\mathbf{h}_t^{(L)})$$
---

## 四、 工业级代码实践拆解 (Keras/TensorFlow)

### 1. CGC 核心模块网络 (`cgc_net`)
该函数实现了 CGC 中“任务专家”与“共享专家”的硬隔离融合。

```python
def cgc_net(
        input_list,
        task_num,
        task_expert_num,
        shared_expert_num,
        task_expert_dnn_units,
        shared_expert_dnn_units,
        task_gate_dnn_units,
        shared_gate_dnn_units,
        leval_name=None,
        is_last=False):
    """CGC（共享专家 + 任务门控）核心结构（简化版）
    - 每个任务：拥有若干 Task-Experts；
    - 全局：拥有若干 Shared-Experts；
    - 每个任务 Gate 产生 softmax 权重，对其 Task-Experts 与 Shared-Experts 加权融合；
    - 若非最后一层：再用 Shared-Gate 融合所有任务的 Task-Experts 与 Shared-Experts，供下一层共享使用。
    input_list：为方便处理，给每个任务复制一份相同输入，最后一个为共享输入。
    """

    # 任务专家：每个任务创建 task_expert_num 个专家
    task_expert_list = []
    for i in range(task_num):
        task_expert_list.append([
            DNNs(task_expert_dnn_units, name=f"{leval_name}_task_{i}_expert_{j}")(input_list[i])
            for j in range(task_expert_num)
        ])

    # 共享专家：创建 shared_expert_num 个专家（共享输入使用 input_list[-1]）
    shared_expert_list = [
        DNNs(shared_expert_dnn_units, name=f"{leval_name}_shared_expert_{i}")(input_list[-1])
        for i in range(shared_expert_num)
    ]

    # 任务门控与融合：对当前任务的（Task + Shared）专家集合进行 softmax 加权求和
    cgc_outputs = []
    fusion_expert_num = task_expert_num + shared_expert_num
    for i in range(task_num):
        cur_experts = task_expert_list[i] + shared_expert_list
        experts = tf.keras.layers.Lambda(lambda xs: tf.stack(xs, axis=1))(cur_experts)  # [B, E, D]

        gate_hidden = DNNs(task_gate_dnn_units, name=f"{leval_name}_task_{i}_gate")(input_list[i])
        gate_weights = tf.keras.layers.Dense(fusion_expert_num, use_bias=False, activation='softmax')(gate_hidden)  # [B, E]

        # 加权融合：einsum('be,bed->bd') == sum_e w_e * expert_e
        fused = tf.keras.layers.Lambda(lambda x: tf.einsum('be,bed->bd', x[0], x[1]))([gate_weights, experts])
        cgc_outputs.append(fused)

    # 若非最后一层：共享门控融合所有任务专家与共享专家，作为下一层共享输入
    if not is_last:
        # 展平所有任务的专家 + 共享专家
        all_task_experts = [e for task in task_expert_list for e in task]
        cur_experts = all_task_experts + shared_expert_list
        experts_all = tf.keras.layers.Lambda(lambda xs: tf.stack(xs, axis=1))(cur_experts)  # [B, E_all, D]
        cur_expert_num = len(cur_experts)

        shared_gate_hidden = DNNs(shared_gate_dnn_units, name=f"{leval_name}_shared_gate")(input_list[-1])
        shared_gate_weights = tf.keras.layers.Dense(cur_expert_num, use_bias=False, activation='softmax')(shared_gate_hidden)  # [B, E_all]
        shared_fused = tf.keras.layers.Lambda(lambda x: tf.einsum('be,bed->bd', x[0], x[1]))([shared_gate_weights, experts_all])
        cgc_outputs.append(shared_fused)

    return cgc_outputs
```

### 2. PLE 宏观架构构建 (build_ple_model)
通过循环将 cgc_net 堆叠起来，形成渐进式提取网络。
```python
def build_ple_model(
        feature_columns,
        task_name_list,
        ple_level_nums=1,
        task_expert_num=4,
        shared_expert_num=2,
        task_expert_dnn_units=[128, 64],
        shared_expert_dnn_units=[128, 64],
        task_gate_dnn_units=[128, 64],
        shared_gate_dnn_units=[128, 64],
        task_tower_dnn_units=[128, 64],
        ):
    # 1) 输入与嵌入：构建输入层/分组嵌入，拼接为 PLE 的共享输入
    input_layer_dict = build_input_layer(feature_columns)
    group_embedding_feature_dict = build_group_feature_embedding_table_dict(
        feature_columns, input_layer_dict, prefix="embedding/"
    )
    dnn_inputs = concat_group_embedding(group_embedding_feature_dict, 'dnn')

    # 2) 级联 PLE（CGC）层：每层包含“任务专家 + 共享专家 + 门控”，最后一层仅输出任务特征
    task_num = len(task_name_list)
    ple_input_list = [dnn_inputs] * (task_num + 1)  # 前 task_num 为各任务输入，末尾为共享输入
    for i in range(ple_level_nums):
        is_last = (i == ple_level_nums - 1)
        ple_input_list = cgc_net(
            ple_input_list,
            task_num,
            task_expert_num,
            shared_expert_num,
            task_expert_dnn_units,
            shared_expert_dnn_units,
            task_gate_dnn_units,
            shared_gate_dnn_units,
            leval_name=f"cgc_level_{i}",
            is_last=is_last
        )

    # 3) 任务塔与输出：将各任务特征送入塔 DNN，得到每个任务的概率输出
    task_output_list = []
    for i in range(task_num):
        task_logit = DNNs(task_tower_dnn_units + [1])(ple_input_list[i])
        task_prob = PredictLayer(name="task_" + task_name_list[i])(task_logit)
        task_output_list.append(task_prob)

    # 4) 构建模型：输入为所有原始输入层，输出为各任务概率
    model = tf.keras.Model(inputs=list(input_layer_dict.values()), outputs=task_output_list)
    return model
    ```