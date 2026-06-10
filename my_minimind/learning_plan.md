# MiniMind 手写复现学习路线

这个路线按“先主干、再对话、再工程化、再高级方法”的顺序推进。原项目目录保持不动，作为参考；自己的实现统一放在 `my_minimind/` 下。

## 第一轮：写出最小 GPT/LLM

目标是从 0 写出一个能完成语言模型训练闭环的 Dense 版小模型。完成后应该能够用文本数据做预训练，并用最简单的生成逻辑输出文本。

`model/model_minimind.py` 采用“贴近原项目结构的学习版”：类名、主要函数名、`forward` 参数风格尽量和原项目 `model/model_minimind.py` 保持一致，方便逐行对照学习；但第一轮只实现 Dense 主干和 Causal LM loss，不提前实现 MoE、KV cache、复杂采样等后续轮次内容。

建议文件结构：

```text
my_minimind/
  model/
    model_minimind.py
  dataset/
    lm_dataset.py
  trainer/
    train_pretrain.py
  scripts/
    eval_llm.py
```

需要实现的内容：

- `model/model_minimind.py`
  - `MiniMindConfig`
  - `RMSNorm`
  - RoPE 位置编码
  - Causal Self-Attention
  - FeedForward
  - Transformer Block
  - `MiniMindModel`
  - `MiniMindForCausalLM`
  - Causal LM loss

- `dataset/lm_dataset.py`
  - `PretrainDataset`
  - 文本读取
  - tokenizer 编码
  - `bos/eos/pad` 处理
  - `labels` 构造
  - pad 位置设置为 `-100`

- `trainer/train_pretrain.py`
  - 加载 tokenizer
  - 创建模型
  - 创建 `PretrainDataset`
  - 创建 `DataLoader`
  - `AdamW` 优化器
  - 单卡训练循环
  - 打印 loss
  - 保存 `pretrain` 权重

- `scripts/eval_llm.py`
  - 加载模型权重
  - 输入 prompt
  - tokenizer 编码
  - greedy generate
  - decode 输出文本

第一轮完成标志：

- 随机 `input_ids` 可以 forward
- `logits` shape 正确
- `loss` 是正常标量
- 小数据预训练可以跑通
- 可以用保存的权重做简单生成

## 第二轮：加入对话能力

目标是在第一轮预训练模型的基础上加入 SFT，让模型从“文本续写”变成“能按 user/assistant 格式对话”。

建议文件结构：

```text
my_minimind/
  dataset/
    lm_dataset.py
  trainer/
    train_sft.py
  scripts/
    chat_cli.py
```

需要实现的内容：

- `dataset/lm_dataset.py`
  - `SFTDataset`
  - 多轮对话样本读取
  - chat prompt 拼接
  - assistant-only loss mask
  - user/system 部分 label 设置为 `-100`
  - pad 部分 label 设置为 `-100`

- `trainer/train_sft.py`
  - 加载第一轮得到的 `pretrain` 权重
  - 创建 `SFTDataset`
  - 复用单卡训练循环
  - 继续训练模型
  - 保存 `full_sft` 权重

- `scripts/chat_cli.py`
  - 加载 `full_sft` 权重
  - 命令行输入用户问题
  - 按 chat 格式组织 prompt
  - 调用模型生成回答
  - 维护简单多轮历史

第二轮完成标志：

- SFT 样本的 label mask 可以解释清楚
- 只对 assistant 回复计算 loss
- 可以从 `pretrain` 权重继续训练
- 命令行里可以进行简单多轮对话

## 第三轮：补工程化训练和推理优化

目标是把前两轮的“能跑”升级成“更像真实训练代码”，让训练更稳定、可恢复，推理更灵活。

建议文件结构：

```text
my_minimind/
  trainer/
    trainer_utils.py
    train_pretrain.py
    train_sft.py
  model/
    model_minimind.py
  scripts/
    eval_llm.py
    chat_cli.py
```

需要实现的内容：

- `trainer/trainer_utils.py`
  - 随机种子设置
  - device 选择
  - 学习率调度
  - checkpoint 保存
  - checkpoint 恢复
  - 模型参数统计

- `trainer/train_pretrain.py`
  - 梯度累积
  - 梯度裁剪
  - 混合精度训练
  - resume 训练
  - 更清晰的日志输出

- `trainer/train_sft.py`
  - 梯度累积
  - 梯度裁剪
  - 混合精度训练
  - resume 训练
  - 从指定权重继续训练

- `model/model_minimind.py`
  - `attention_mask`
  - `past_key_values`
  - KV cache
  - generate 时复用 KV cache

- `scripts/eval_llm.py`
  - temperature
  - top-k sampling
  - top-p sampling
  - repetition penalty
  - 最大生成长度控制

- `scripts/chat_cli.py`
  - 多轮历史长度控制
  - 采样参数配置
  - 更稳定的停止条件

第三轮完成标志：

- 训练中断后可以继续
- 可以选择从不同权重开始训练
- 支持混合精度和梯度累积
- 生成时支持采样参数
- KV cache 生效，逐 token 生成更快

## 第四轮 A：高级微调和模型结构

目标是复现 MiniMind 中更进阶但仍适合手写学习的机制，包括 LoRA、DPO 和 MoE。

建议文件结构：

```text
my_minimind/
  model/
    model_minimind.py
    model_lora.py
  dataset/
    lm_dataset.py
  trainer/
    train_lora.py
    train_dpo.py
```

需要实现的内容：

- `model/model_lora.py`
  - `LoRA` 模块
  - 给 Linear 层挂载 LoRA 分支
  - 冻结原模型参数
  - 保存 LoRA 权重
  - 加载 LoRA 权重
  - 合并 LoRA 到基础权重

- `trainer/train_lora.py`
  - 加载 `full_sft` 基础模型
  - 注入 LoRA
  - 只训练 LoRA 参数
  - 保存 LoRA 权重

- `dataset/lm_dataset.py`
  - `DPODataset`
  - chosen/rejected 样本读取
  - chosen/rejected 的 token 和 mask 构造

- `trainer/train_dpo.py`
  - policy model
  - reference model
  - token log probability 计算
  - DPO loss
  - DPO 训练循环
  - 保存 `dpo` 权重

- `model/model_minimind.py`
  - `MOEFeedForward`
  - router gate
  - top-k expert routing
  - expert 权重聚合
  - MoE auxiliary loss
  - Dense/MoE 配置切换

第四轮 A 完成标志：

- LoRA 只更新少量新增参数
- LoRA 权重可以保存、加载、合并
- DPO 能读取 chosen/rejected 数据并计算 loss
- Dense 和 MoE 都能 forward
- MoE 训练时能产生 auxiliary loss

## 第四轮 B：强化学习和工具调用

目标是学习 SFT/DPO 之后更复杂的后训练方法，包括 PPO、GRPO、CISPO、Tool Use 和 Agentic RL。

建议文件结构：

```text
my_minimind/
  dataset/
    lm_dataset.py
  trainer/
    rollout_engine.py
    train_grpo.py
    train_ppo.py
    train_agent.py
```

需要实现的内容：

- `dataset/lm_dataset.py`
  - `RLAIFDataset`
  - `AgentRLDataset`
  - RL prompt 构造
  - 工具调用样本读取
  - ground truth 字段读取

- `trainer/rollout_engine.py`
  - prompt 批量生成
  - 记录 response token
  - 记录 old log probability
  - 返回 completion mask
  - 统一 rollout 结果结构

- `trainer/train_grpo.py`
  - 同一 prompt 生成多个回答
  - reward 计算
  - group reward mean/std
  - advantage 计算
  - KL penalty
  - GRPO loss
  - CISPO loss 变体

- `trainer/train_ppo.py`
  - actor model
  - critic model
  - reward 计算
  - value 估计
  - GAE advantage
  - PPO clipped loss
  - value loss
  - KL penalty

- `trainer/train_agent.py`
  - tool call 解析
  - tool 执行
  - 多轮 rollout
  - 延迟 reward
  - 格式奖励
  - ground truth 校验奖励
  - Agentic RL 训练循环

第四轮 B 完成标志：

- 能完成一次 rollout 并拿到 response token/logprob
- GRPO 能基于同一 prompt 的多条回答计算 group advantage
- PPO 能同时训练 actor 和 critic
- Tool Use 能解析并执行工具调用
- Agentic RL 能根据多轮轨迹计算奖励

## 推荐推进顺序

优先顺序建议如下：

```text
第一轮 -> 第二轮 -> 第三轮 -> 第四轮 A -> 第四轮 B
```

如果想进一步压缩学习路线，可以先走：

```text
Dense 主干 -> SFT 对话 -> 工程化训练 -> LoRA -> DPO
```

这条路线最适合建立对 MiniMind 主线的理解。
