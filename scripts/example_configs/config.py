"""
DQN NoisyGCN 训练配置示例
使用方法: 复制并修改此文件，然后在训练前将其重命名为config.py
"""

# 训练方法: "DQN" 或 "A2C"
TRAINING_METHOD = "DQN"

# 模型类型: "GCN", "DuelingGCN", "NoisyGCN", "g-U-Net", "GG-NN"
MODEL_NAME = "NoisyGCN"

# 是否使用Double DQN (仅对DQN方法有效)
USE_DOUBLE_DQN = True

# 探索策略: "noisy", "epsilon", "bayesian", None表示自动选择
EXPLORATION_METHOD = "noisy"

# 是否继续之前的训练
CONTINUE_TRAINING = False

# 要运行的epoch数量，None表示使用训练对象的默认配置
EPOCHS = None

# 指定GPU ID，None表示自动选择可用的GPU
GPU_ID = None

# 是否只是列出可用的GPU然后退出
LIST_GPUS = False

# DQN训练参数
DQN_PARAMS = {
    # 是否使用优先经验回放缓冲区(True: 优先经验回放, False: 普通经验回放)
    'use_priority_buffer': True,
    # 经验回放缓冲区大小
    'buffer_size': 1e5,
    # 训练批次大小
    'batch_size': 128,
}

# 模型架构参数
MODEL_PARAMS = {
    'in_channels': 5,
    'hidden_channels': 1000,
    'out_channels': 1000,
    'depth': 3
}

# 训练数据路径
DATA_PATH = '../data'

# 获取配置字典，可直接传递给TrainingManager
def get_config():
    return {
        'training_method': TRAINING_METHOD,
        'model_name': MODEL_NAME,
        'use_double_dqn': USE_DOUBLE_DQN,
        'exploration_method': EXPLORATION_METHOD,
        'continue_training': CONTINUE_TRAINING,
        'epochs': EPOCHS,
        'gpu_id': GPU_ID,
        'list_gpus': LIST_GPUS,
        'dqn_params': DQN_PARAMS,
        'model_params': MODEL_PARAMS,
        'data_path': DATA_PATH
    } 