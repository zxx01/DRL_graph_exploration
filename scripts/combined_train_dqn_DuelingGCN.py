#!/usr/bin/env python3
import os
import pickle
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import Networks
from policy import DeepQ, A2C
import example_configs.dqn_duelinggcn

class TrainingManager:
    """训练管理器：整合训练初始化和执行功能"""
    def __init__(self, config=None):
        # 设置配置
        self.config = config or self._default_config()
        
        # 设置GPU设备
        if 'gpu_id' in self.config and self.config['gpu_id'] is not None:
            gpu_id = self.config['gpu_id']
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                self.device = torch.device(f"cuda:{gpu_id}")
                print(f"使用GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            else:
                if not torch.cuda.is_available():
                    print(f"警告: 未检测到CUDA设备，将使用CPU进行训练")
                elif gpu_id >= torch.cuda.device_count():
                    print(f"警告: 指定的GPU ID {gpu_id} 超出可用范围(0-{torch.cuda.device_count()-1})，将使用CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                print(f"使用默认GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("使用CPU进行训练")
        
        self.config['device'] = self.device
        
        # 设置数据路径前缀
        if 'data_path' in self.config:
            self.data_path = self.config['data_path']
        else:
            self.data_path = '../data'
        
        # 设置路径
        self._setup_paths()
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.config['log_path'])

    def _default_config(self):
        """返回默认配置"""
        return {
            'training_method': "DQN",  # DQN 或 A2C
            'model_name': "NoisyGCN",  # GCN, GG-NN, g-U-Net, DuelingGCN, NoisyGCN
            'use_double_dqn': True,
            'exploration_method': "noisy",  # noisy, epsilon, bayesian, None表示自动选择
            'gpu_id': None,  # 默认使用自动选择GPU
            'dqn_params': {
                'use_priority_buffer': True,
                'buffer_size': 1e5,
                'batch_size': 128
            },
            'model_params': {
                'in_channels': 5,
                'hidden_channels': 1000,
                'out_channels': 1000,
                'depth': 3
            },
            'data_path': '../data'
        }

    def _setup_paths(self):
        """设置训练所需的路径并创建目录"""
        # 设置基本路径
        case_path = f"{self.config['training_method']}_{self.config['model_name']}"
        self.config['case_path'] = f"{case_path}/"
        self.config['object_path'] = f'{self.data_path}/training_object_data/{case_path}/'
        self.config['log_path'] = f"{self.data_path}/torch_logs/{case_path}"
        
        # 创建目录
        os.makedirs(self.config['log_path'], exist_ok=True)
        os.makedirs(self.config['object_path'], exist_ok=True)

    def create_model(self, model_type):
        """创建指定类型的模型"""
        model_name = self.config['model_name']
        model_params = self.config['model_params']
        
        # 根据模型类型和训练方法选择合适的模型类
        if model_type == 'dqn_policy' or model_type == 'dqn_target':
            if model_name == "GCN":
                return Networks.GCN()
            elif model_name == "DuelingGCN":
                return Networks.DuelingGCN()
            elif model_name == "NoisyGCN":
                return Networks.NoisyGCN()
            elif model_name == "g-U-Net":
                return Networks.GraphUNet(**model_params)
            elif model_name == "GG-NN":
                return Networks.GGNN()
        elif model_type == 'a2c_policy':
            if model_name == "GCN":
                return Networks.PolicyGCN()
            elif model_name == "g-U-Net":
                return Networks.PolicyGraphUNet(**model_params)
            elif model_name == "GG-NN":
                return Networks.PolicyGGNN()
        elif model_type == 'a2c_value':
            if model_name == "GCN":
                return Networks.ValueGCN()
            elif model_name == "g-U-Net":
                return Networks.ValueGraphUNet(**model_params)
            elif model_name == "GG-NN":
                return Networks.ValueGGNN()
                
        raise ValueError(f"不支持的模型类型组合: {self.config['model_name']}/{model_type}")

    def train(self, continue_training=None, epochs=None):
        """执行训练流程"""
        # 获取配置信息
        method = self.config['training_method']
        model_name = self.config['model_name']
        use_double_dqn = self.config['use_double_dqn']
        exploration_method = self.config['exploration_method']
        object_path = self.config['object_path']
        device = self.config['device']
        
        # 获取DQN特定参数
        dqn_params = self.config.get('dqn_params', {})
        use_priority_buffer = dqn_params.get('use_priority_buffer', True)
        buffer_size = dqn_params.get('buffer_size', 1e5)
        batch_size = dqn_params.get('batch_size', 128)
        
        # 使用配置文件中的参数，如果没有传入
        if continue_training is None:
            continue_training = self.config.get('continue_training', False)
        if epochs is None:
            epochs = self.config.get('epochs', None)

        # 初始化新训练或加载现有训练
        if continue_training:
            print(f"继续训练 - 方法: {method}, 模型: {model_name}")
            
            # 加载训练对象
            with open(f"{object_path}saved_training.pkl", 'rb') as f:
                training = pickle.load(f)
                
            # 加载模型
            if method == "DQN":
                policy_model = self.create_model('dqn_policy').to(device)
                target_model = self.create_model('dqn_target').to(device)
                policy_model.load_state_dict(torch.load(f"{object_path}Model_Policy.pt"))
                target_model.load_state_dict(torch.load(f"{object_path}Model_Target.pt"))
                models = (policy_model, target_model)
            else:  # A2C
                policy_model = self.create_model('a2c_policy').to(device)
                value_model = self.create_model('a2c_value').to(device)
                policy_model.load_state_dict(torch.load(f"{object_path}Model_Policy.pt"))
                value_model.load_state_dict(torch.load(f"{object_path}Model_Value.pt"))
                models = (policy_model, value_model)
        else:
            print(f"初始化新训练 - 方法: {method}, 模型: {model_name}")
            
            # 创建训练对象
            if method == "DQN":
                training = DeepQ(
                    self.config['case_path'], 
                    model_name, 
                    device,
                    use_priority_buffer=use_priority_buffer,
                    buffer_size=buffer_size,
                    batch_size=batch_size
                )
                
                policy_model = self.create_model('dqn_policy').to(device)
                target_model = self.create_model('dqn_target').to(device)
                torch.save(policy_model.state_dict(), f"{object_path}Model_Policy.pt")
                torch.save(target_model.state_dict(), f"{object_path}Model_Target.pt")
                models = (policy_model, target_model)
            else:  # A2C
                training = A2C(self.config['case_path'], device)
                
                policy_model = self.create_model('a2c_policy').to(device)
                value_model = self.create_model('a2c_value').to(device)
                torch.save(policy_model.state_dict(), f"{object_path}Model_Policy.pt")
                torch.save(value_model.state_dict(), f"{object_path}Model_Value.pt")
                models = (policy_model, value_model)
                
            # 保存训练对象
            with open(f"{object_path}saved_training.pkl", 'wb') as f:
                pickle.dump(training, f, pickle.HIGHEST_PROTOCOL)

        # 确定要运行的epoch数量
        if epochs is None:
            epochs = int(training.EXPLORE / training.epoch)
            
        print(f"开始训练 {epochs} 个epochs...")
        
        # 运行训练
        for i in range(epochs):
            print(f"Epoch {i+1}/{epochs}")
            
            # 开始计时
            time_start = time.time()
            
            # 为单个epoch运行训练
            if method == "DQN":
                training.running(
                    models[0],  # policy_model
                    models[1],  # target_model
                    double_dqn=use_double_dqn,
                    exploration_method=exploration_method
                )
            else:  # A2C
                training.running(
                    models[0],  # policy_model
                    models[1]   # value_model
                )
                
            # 计算持续时间
            duration = time.time() - time_start
            print(f"Epoch {i+1} 完成，用时: {duration:.2f}秒")
            
            # 保存训练状态
            with open(f"{object_path}saved_training.pkl", 'wb') as f:
                pickle.dump(training, f)
                
            # 保存模型
            if method == "DQN":
                torch.save(models[0].state_dict(), f"{object_path}Model_Policy.pt")
                torch.save(models[1].state_dict(), f"{object_path}Model_Target.pt")
            else:  # A2C
                torch.save(models[0].state_dict(), f"{object_path}Model_Policy.pt")
                torch.save(models[1].state_dict(), f"{object_path}Model_Value.pt")
                
            # 记录指标
            self._log_metrics()
            
        print("训练完成!")
        
    def _log_metrics(self):
        """记录训练指标到TensorBoard"""
        object_path = self.config['object_path']
        
        # 记录奖励数据
        try:
            reward_data = np.loadtxt(f"{object_path}temp_reward.csv", delimiter=",")
            if reward_data.size > 0:
                if reward_data.ndim == 1 and len(reward_data) >= 2:
                    # 单行数据
                    self.writer.add_scalar('Train/avg_reward', reward_data[1], reward_data[0])
                else:
                    # 多行数据
                    for row in reward_data:
                        if len(row) >= 2:
                            self.writer.add_scalar('Train/avg_reward', row[1], row[0])
        except (IOError, ValueError) as e:
            print(f"警告: 无法加载奖励数据: {e}")
        
        # 记录损失数据
        try:
            loss_data = np.loadtxt(f"{object_path}temp_loss.csv", delimiter=",")
            if loss_data.size > 0:
                if loss_data.ndim == 1 and len(loss_data) >= 2:
                    # 单行数据
                    self.writer.add_scalar('Train/loss', loss_data[1], loss_data[0])
                else:
                    # 多行数据
                    for row in loss_data:
                        if len(row) >= 2:
                            self.writer.add_scalar('Train/loss', row[1], row[0])
        except (IOError, ValueError) as e:
            print(f"警告: 无法加载损失数据: {e}")

def list_available_gpus():
    """列出所有可用的GPU及其信息"""
    if not torch.cuda.is_available():
        print("未检测到可用的GPU。")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个可用的GPU:")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GB
        print(f"  GPU {i}: {gpu_name} (显存: {gpu_mem:.2f} GB)")

def main():
    """主函数"""
    # 从配置文件加载配置
    cfg = example_configs.dqn_duelinggcn.get_config()
    
    # 如果需要列出可用GPU
    if cfg.get('list_gpus', False):
        list_available_gpus()
        return
    
    # 打印配置信息
    print("训练配置:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")
    
    # 创建训练管理器并执行训练
    manager = TrainingManager(cfg)
    manager.train(
        continue_training=cfg.get('continue_training'),
        epochs=cfg.get('epochs')
    )

if __name__ == "__main__":
    main() 