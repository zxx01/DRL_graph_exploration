# DRL图探索训练系统

这个系统使用深度强化学习进行图探索任务的训练。

## 配置和训练

### 基本用法

1. 复制一个示例配置文件作为起点：
   ```bash
   cp example_configs/dqn_noisy.py config.py
   ```

2. 根据需要编辑`config.py`文件中的参数

3. 运行训练：
   ```bash
   python combined_train.py
   ```

### 列出可用GPU

如果你想查看系统中的可用GPU，可以设置`config.py`中的`LIST_GPUS = True`，然后运行：

```bash
python combined_train.py
```

这将显示所有可用的GPU及其详细信息。

### 配置参数说明

主要配置参数说明：

| 参数名 | 说明 | 可选值 |
|--------|------|--------|
| TRAINING_METHOD | 训练方法 | "DQN", "A2C" |
| MODEL_NAME | 模型类型 | "GCN", "DuelingGCN", "NoisyGCN", "g-U-Net", "GG-NN" |
| USE_DOUBLE_DQN | 是否使用Double DQN | True, False (仅DQN有效) |
| EXPLORATION_METHOD | 探索策略 | "noisy", "epsilon", "bayesian", None |
| CONTINUE_TRAINING | 是否继续之前的训练 | True, False |
| EPOCHS | 要运行的epoch数量 | 整数或None(使用默认值) |
| GPU_ID | 指定使用的GPU ID | 整数或None(自动选择) |
| LIST_GPUS | 是否只列出GPU然后退出 | True, False |

### DQN特定参数

DQN训练方法的特定参数，在`DQN_PARAMS`字典中配置：

| 参数名 | 说明 | 可选值 |
|--------|------|--------|
| use_priority_buffer | 是否使用优先经验回放缓冲区 | True(优先经验回放), False(普通经验回放) |
| buffer_size | 经验回放缓冲区大小 | 整数，如100000 |
| batch_size | 训练批次大小 | 整数，如32, 64, 128 |

### 示例配置

系统提供了几个示例配置文件：

1. `example_configs/dqn_noisy.py` - 使用NoisyGCN的DQN训练(优先经验回放)
2. `example_configs/dqn_standard.py` - 使用GCN的DQN训练(普通经验回放)
3. `example_configs/a2c_gcn.py` - 使用GCN的A2C训练

## 文件结构

- `combined_train.py` - 训练主程序
- `config.py` - 训练配置文件
- `example_configs/` - 示例配置文件目录
- `Networks.py` - 网络模型定义
- `policy.py` - 强化学习策略实现

## 训练路径

默认情况下，训练数据和模型保存在以下位置：

- 训练对象: `../data/training_object_data/{METHOD}_{MODEL}/`
- 日志: `../data/torch_logs/{METHOD}_{MODEL}/`

可以通过修改`DATA_PATH`参数来更改基础路径。 