import pickle
import sys
import torch
import Networks

# 获取命令行参数
training_method = sys.argv[1]  # DQN 或 A2C
model_name = sys.argv[2]  # 网络模型类型
use_double_dqn = True  # 默认使用Double DQN

# 如果有第三个参数，表示是否使用Double DQN
if len(sys.argv) > 3:
    use_double_dqn = sys.argv[3].lower() == 'true'

# 如果有第四个参数，表示使用哪种探索策略
exploration_method = None
if len(sys.argv) > 4:
    if sys.argv[4] in ["noisy", "epsilon", "bayesian"]:
        exploration_method = sys.argv[4]

# 设置文件路径
case_path = training_method + "_" + model_name
object_path = '../data/training_object_data/' + case_path + '/'
# 加载pickle文件
full_file_name = object_path + 'saved_training.pkl'
with open(full_file_name, 'rb') as f:
    training = pickle.load(f)

# 选择训练方法
if training_method == "DQN":
    # 加载训练模型
    policy_model_name = object_path + 'Model_Policy.pt'
    target_model_name = object_path + 'Model_Target.pt'
    check_point_p = torch.load(policy_model_name)
    check_point_t = torch.load(target_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 根据模型名称创建对应的网络
    if model_name == "GCN":
        policy_model = Networks.GCN()
        target_model = Networks.GCN()
    elif model_name == "DuelingGCN":
        policy_model = Networks.DuelingGCN()
        target_model = Networks.DuelingGCN()
    elif model_name == "NoisyGCN":
        policy_model = Networks.NoisyGCN()
        target_model = Networks.NoisyGCN()
    elif model_name == "g-U-Net":
        policy_model = Networks.GraphUNet(
            in_channels=5, hidden_channels=1000, out_channels=1000, depth=3)
        target_model = Networks.GraphUNet(
            in_channels=5, hidden_channels=1000, out_channels=1000, depth=3)
    elif model_name == "GG-NN":
        policy_model = Networks.GGNN()
        target_model = Networks.GGNN()
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")
    
    # 加载模型权重
    policy_model.load_state_dict(check_point_p)
    target_model.load_state_dict(check_point_t)
    policy_model.to(device)
    target_model.to(device)
    
    # 打印配置信息
    print(f"继续训练 {training_method} 使用 {model_name} {'(Double DQN)' if use_double_dqn else ''} 探索策略: {exploration_method or '自动选择'}")
    
    # 启动训练
    training.running(policy_model, target_model, double_dqn=use_double_dqn, exploration_method=exploration_method)

elif training_method == "A2C":
    # 加载训练模型
    policy_model_name = object_path + 'Model_Policy.pt'
    value_model_name = object_path + 'Model_Value.pt'
    check_point_p = torch.load(policy_model_name)
    check_point_v = torch.load(value_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == "GCN":
        policy_model = Networks.PolicyGCN()
        value_model = Networks.ValueGCN()
    elif model_name == "g-U-Net":
        policy_model = Networks.PolicyGraphUNet(
            in_channels=5, hidden_channels=1000, out_channels=1000, depth=3)
        value_model = Networks.ValueGraphUNet(
            in_channels=5, hidden_channels=1000, out_channels=1000, depth=3)
    elif model_name == "GG-NN":
        policy_model = Networks.PolicyGGNN()
        value_model = Networks.ValueGGNN()
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")
        
    policy_model.load_state_dict(check_point_p)
    value_model.load_state_dict(check_point_v)
    policy_model.to(device)
    value_model.to(device)
    
    print(f"继续训练 {training_method} 使用 {model_name}")
    training.running(policy_model, value_model)

# 保存训练状态
with open(full_file_name, 'wb') as f:
    pickle.dump(training, f)
