import os
import pickle
import time
import subprocess
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import Networks
from policy import DeepQ, A2C

# setup the training model and method
training_method = "DQN"  # DQN, A2C
model_name = "NoisyGCN"  # GCN, GG-NN, g-U-Net, DuelingGCN, NoisyGCN
# using double DQN
use_double_dqn = True
# 指定探索策略 (可选: "None", "noisy", "epsilon", "bayesian")
exploration_method = "noisy"  # None表示自动选择

# setup local file paths
case_path = training_method + "_" + model_name + "/"
object_path = '../data/training_object_data/' + case_path
log_path = "../data/torch_logs/" + case_path
if not os.path.exists(log_path):
    os.makedirs(log_path)
if not os.path.exists(object_path):
    os.makedirs(object_path)

# tensorboard
writer = SummaryWriter(log_dir=log_path)

# choose training method
if training_method == "DQN":
    # create training object
    training = DeepQ(case_path, model_name)
    # 保存训练对象
    full_file_name = object_path + 'saved_training.pkl'
    with open(full_file_name, 'wb') as f:
        pickle.dump(training, f, pickle.HIGHEST_PROTOCOL)
    # save the model
    policy_model_name = object_path + 'Model_Policy.pt'
    target_model_name = object_path + 'Model_Target.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    policy_model.to(device)
    target_model.to(device)
    torch.save(policy_model.state_dict(), policy_model_name)
    torch.save(target_model.state_dict(), target_model_name)

elif training_method == "A2C":
    # create training object
    training = A2C(case_path)
    # 保存训练对象
    full_file_name = object_path + 'saved_training.pkl'
    with open(full_file_name, 'wb') as f:
        pickle.dump(training, f, pickle.HIGHEST_PROTOCOL)
    # save the model
    policy_model_name = object_path + 'Model_Policy.pt'
    value_model_name = object_path + 'Model_Value.pt'
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
    policy_model.to(device)
    value_model.to(device)
    torch.save(policy_model.state_dict(), policy_model_name)
    torch.save(value_model.state_dict(), value_model_name)

# print(f"开始训练 {training_method} 使用 {model_name} {'(Double DQN)' if use_double_dqn else ''} 探索策略: {exploration_method or '自动选择'}")
# training.running(policy_model, target_model, test=False, double_dqn=use_double_dqn, exploration_method=exploration_method)

# 根据training对象设置epoch数量
if training_method == "DQN":
    epoch_nums = training.EXPLORE / training.epoch
elif training_method == "A2C":
    epoch_nums = training.EXPLORE / training.epoch

time_total = 0
for i in range(int(epoch_nums)):
    cmd = "python3 run_training.py " + training_method + " " + model_name + " " + str(use_double_dqn).lower()
    if exploration_method:
        cmd += " " + exploration_method

    time_start = time.time()
    subprocess.call(cmd, shell=True)
    time_end = time.time()
    duration = time_end - time_start
    time_total = time_total + duration
    print(f"10000 epoches time: {duration} s.")

    temp_reward_data = np.loadtxt(
        object_path + "temp_reward.csv", delimiter=",")
    temp_loss_data = np.loadtxt(object_path + "temp_loss.csv", delimiter=",")
    for j in range(np.shape(temp_reward_data)[0]):
        step_t = temp_reward_data[j][0]
        reward = temp_reward_data[j][1]
        writer.add_scalar('Train/avg_reward', reward, step_t)
    for j in range(np.shape(temp_loss_data)[0]):
        step_t = temp_loss_data[j][0]
        loss = temp_loss_data[j][1]
        writer.add_scalar('Train/loss', loss, step_t)


print(f"1e6 total time: {duration} s.")
